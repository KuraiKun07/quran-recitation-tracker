# train_muq.py - closed-set fine-tune for the letter openings. Run from backend.
#   python train_muq.py                      # lr 1e-5, 6 epochs
#   python train_muq.py --freeze-encoder --epochs 8 --tag muq_fe
# Training set : letter-opening takes that are NOT held out (takes 3, 6, ... are held out) PLUS
#                your normal clips from ayahs that are not in the 20% held-out split (replay, so the
#                model does not forget normal recitation). Targets are "target B" (Uthmani converted
#                to the model's style; for the openings that is the compact form, e.g. the text of 2:1).
# Each epoch is exported to CTranslate2 and scored on
#   (1) held-out letter-opening TAKES: exact match of the whole-clip transcript (beam 4 and greedy)
#   (2) held-out NORMAL ayahs: real errors with the tolerant metric (must not get worse than epoch 0)
# Prefix behaviour (early / flip) is NOT measured here; afterwards run:
#   python muq_eval.py --model runs\<tag>\ep<N>_ct2
import argparse, filecmp, json, math, os, random, shutil, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

from ct2conv import convert
from train import BASE, PROD, augment, decode_rows
from trainutil import (HERE, is_heldout, is_heldout_take, load_manifest, norm, read_wav,
                       to_model_style, tolerant_ops)


def speed(a, rng):
    """speed perturbation 0.85-1.15x (changes tempo and pitch a little)"""
    f = rng.uniform(0.85, 1.15)
    n = max(1600, int(len(a) / f))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--muq-repeat", type=int, default=1, help="repeat each opening clip this many times per epoch")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or ("muq" + ("_fe" if args.freeze_encoder else "") + "_lr%g" % args.lr)
    run = os.path.join(HERE, "runs", tag)
    os.makedirs(run, exist_ok=True)
    logf = open(os.path.join(run, "log.txt"), "a", encoding="utf-8")

    def log(s):
        print(s)
        logf.write(s + "\n")
        logf.flush()

    log("=== run %s | %s | args: %s" % (tag, time.strftime("%Y-%m-%d %H:%M:%S"), vars(args)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rows = load_manifest()
    muq = [r for r in rows if r["kind"] == "muq"]
    normal = [r for r in rows if r["kind"] != "muq"]
    muq_train = [r for r in muq if not is_heldout_take(r)]
    muq_held = [r for r in muq if is_heldout_take(r)]
    n_train = [r for r in normal if not is_heldout(r["surah"], r["ayah"])]
    n_held = [r for r in normal if is_heldout(r["surah"], r["ayah"])]
    n_words = sum(len(norm(r["clean"]).split()) for r in n_held)
    log("train: %d opening clips + %d normal clips | held-out: %d opening takes + %d normal clips (%d words)"
        % (len(muq_train), len(n_train), len(muq_held), len(n_held), n_words))
    if not muq_train or not muq_held:
        raise SystemExit("no letter-opening takes found; record them first")
    bad = [r for r in muq if norm(to_model_style(r["uthmani"])) != norm(r["clean"])]
    log("opening targets whose converted form differs from text_clean after normalisation: %d" % len(bad))

    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tok.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)
    fe = WhisperFeatureExtractor.from_pretrained(BASE, local_files_only=True)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.float32, local_files_only=True)
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32, local_files_only=True)
    model.config.use_cache = False
    model.to("cuda")

    items = []
    for r in muq_train * max(1, args.muq_repeat) + n_train:
        ids = tok(to_model_style(r["uthmani"])).input_ids
        if len(ids) <= 440:
            items.append((r, ids))
    pre = tok.convert_ids_to_tokens(items[0][1][:4])
    if (items[0][1][0] != model.config.decoder_start_token_id or items[0][1][-1] != model.config.eos_token_id
            or pre[3] != "<|notimestamps|>"):
        raise SystemExit("unexpected label prefix/eos; stopping before training on wrong targets")
    audio_cache = {r["file"]: read_wav(r["file"]) for r, _ in items}

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    log("trainable parameters: %.1fM | training items per epoch: %d" % (sum(p.numel() for p in params) / 1e6, len(items)))

    steps_per_ep = math.ceil(len(items) / args.bs)
    total = steps_per_ep * args.epochs
    warm = max(1, int(0.1 * total))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))

    def make_batch(batch):
        feats, dis, labs = [], [], []
        for r, ids in batch:
            a = audio_cache[r["file"]]
            if not args.no_aug:
                a = augment(speed(a, rng), rng)
            feats.append(fe(a, sampling_rate=16000, return_tensors="np").input_features[0])
            dis.append(ids[:-1])
            labs.append([-100] * 3 + ids[4:])
        L = max(len(d) for d in dis)
        eot = model.config.eos_token_id
        di = torch.tensor([d + [eot] * (L - len(d)) for d in dis])
        y = torch.tensor([l + [-100] * (L - len(l)) for l in labs])
        return torch.from_numpy(np.stack(feats)).to("cuda"), di.to("cuda"), y.to("cuda")

    def export(ep):
        hf = os.path.join(run, "hf_last")
        shutil.rmtree(hf, ignore_errors=True)
        model.config.use_cache = True
        model.save_pretrained(hf)
        tok.save_pretrained(hf)
        fe.save_pretrained(hf)
        model.config.use_cache = False
        ct2 = os.path.join(run, "ep%d_ct2" % ep)
        shutil.rmtree(ct2, ignore_errors=True)
        convert(hf, ct2)
        if ep == 1:
            for n in ("config.json", "vocabulary.json"):
                log("  export check: %s identical to production: %s"
                    % (n, filecmp.cmp(os.path.join(PROD, n), os.path.join(ct2, n), shallow=False)))
        return ct2

    def quick_eval(ct2, ep):
        res = decode_rows(ct2, muq_held + n_held)
        k = len(muq_held)
        out = {"epoch": ep}
        with open(os.path.join(run, "heldout_muq_ep%d.txt" % ep), "w", encoding="utf-8") as f:
            for b in (4, 1):
                mo, no = res[b][:k], res[b][k:]
                out["muq%d" % b] = sum(norm(o) == norm(r["clean"]) for r, o in zip(muq_held, mo))
                out["err%d" % b] = sum(len(tolerant_ops(norm(r["clean"]).split(), norm(o).split()))
                                       for r, o in zip(n_held, no))
                if b == 4:
                    for r, o in zip(muq_held, mo):
                        if norm(o) != norm(r["clean"]):
                            f.write("%d:%d take %s  want=%s  got=%s\n" % (r["surah"], r["ayah"], r.get("take", "?"),
                                                                       norm(r["clean"]), norm(o)))
        return out

    def line(o):
        return ("held-out openings exact: beam4 %d/%d, greedy %d/%d | normal held-out real errors: beam4 %d, greedy %d (of %d words)"
                % (o["muq4"], len(muq_held), o["muq1"], len(muq_held), o["err4"], o["err1"], n_words))

    results = []
    o = quick_eval(PROD, 0)
    o["loss"] = None
    results.append(o)
    log("epoch 0 (production): " + line(o))

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(items)))
        random.shuffle(order)
        losses = []
        for i in range(0, len(order), args.bs):
            x, di, y = make_batch([items[j] for j in order[i:i + args.bs]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_features=x, decoder_input_ids=di).logits
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        model.eval()
        torch.cuda.empty_cache()
        ct2 = export(ep)
        o = quick_eval(ct2, ep)
        o["loss"] = sum(losses) / len(losses)
        results.append(o)
        log("epoch %d | loss %.3f | %s | %.0f s" % (ep, o["loss"], line(o), time.time() - t0))
        json.dump(results, open(os.path.join(run, "results.json"), "w"), indent=1)
        torch.cuda.empty_cache()

    base_err = results[0]["err4"] + results[0]["err1"]
    ok = [r for r in results[1:] if r["err4"] + r["err1"] <= base_err + 2]
    pick = max(ok or results[1:], key=lambda r: (r["muq4"] + r["muq1"], -(r["err4"] + r["err1"]), -r["epoch"]))
    log("\nrecommended epoch (most openings right, normal errors within +2 of production): %d -> %s"
        % (pick["epoch"], os.path.join(run, "ep%d_ct2" % pick["epoch"])))
    log("next: python muq_eval.py --model %s" % os.path.join("runs", tag, "ep%d_ct2" % pick["epoch"]))


if __name__ == "__main__":
    main()
