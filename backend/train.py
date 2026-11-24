# train.py - fine-tune whisper-base-ar-quran on your recorded clips. Run from backend.
#   python train.py --target A                      # targets = text_clean (undiacritized)
#   python train.py --target B                      # targets = Uthmani converted to the model's style
#   python train.py --target A --freeze-encoder     # decoder only
# Options: --epochs 8 --lr 1e-5 --bs 8 --no-aug --seed 1 --tag NAME
# After EVERY epoch the checkpoint is exported to CTranslate2 fp16 and scored on the held-out
# ayahs with faster-whisper (the production path): beam 4 (final pass) and greedy (partial passes).
# Output: runs/<tag>/ep<N>_ct2 (one deployable model per epoch), log.txt, results.json,
#         heldout_ep<N>.txt (Arabic, open in VS Code). The console stays ASCII.
import argparse, filecmp, gc, json, math, os, random, shutil, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch                      # before faster_whisper (CUDA DLLs)
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

from ct2conv import convert
from trainutil import HERE, edit_distance, fold_of, is_heldout, load_manifest, norm, read_wav, to_model_style

BASE = "tarteel-ai/whisper-base-ar-quran"
PROD = os.path.join(HERE, "whisper-quran-ct2-fp16")


def target_text(r, kind):
    return r["clean"] if kind == "A" else to_model_style(r["uthmani"])


def augment(a, rng):
    a = a * 10 ** (rng.uniform(-6, 6) / 20)                        # gain +-6 dB
    snr = rng.uniform(25, 45)                                      # light noise, SNR 25-45 dB
    p = float(np.mean(a ** 2)) + 1e-10
    a = a + rng.standard_normal(len(a)).astype(np.float32) * math.sqrt(p / 10 ** (snr / 10))
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def decode_rows(ct2_dir, rows, beams=(4, 1)):
    from faster_whisper import WhisperModel
    m = WhisperModel(ct2_dir, device="cuda", compute_type="float16")
    res = {b: [] for b in beams}
    for r in rows:
        audio = read_wav(r["file"])
        for b in beams:
            segs, _ = m.transcribe(audio, language="ar", beam_size=b, temperature=0.0,
                                   without_timestamps=True, condition_on_previous_text=False,
                                   vad_filter=False,
                                   max_new_tokens=min(440, int(len(audio) / 16000 * 20) + 24),
                                   no_speech_threshold=None, log_prob_threshold=None,
                                   compression_ratio_threshold=None)
            res[b].append(" ".join(sg.text.strip() for sg in segs))
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return res


def score(rows, outs):
    e = n = 0
    for r, o in zip(rows, outs):
        ref = norm(r["clean"]).split()
        e += edit_distance(ref, norm(o).split())
        n += len(ref)
    return e, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["A", "B"], default="A")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--folds", type=int, default=0, help="cross-validation: number of folds (0 = fixed 20%% split)")
    ap.add_argument("--fold", type=int, default=0, help="which fold is held out")
    args = ap.parse_args()

    base = args.tag or (args.target + ("_fe" if args.freeze_encoder else "") + "_lr%g" % args.lr)
    tag = base + ("_f%d" % args.fold if args.folds else "")
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
    if args.folds:
        is_h = lambda r: fold_of(r["surah"], r["ayah"], args.folds) == args.fold
    else:
        is_h = lambda r: is_heldout(r["surah"], r["ayah"])
    held = [r for r in rows if is_h(r)]
    train = [r for r in rows if not is_h(r)]
    held_words = sum(len(norm(r["clean"]).split()) for r in held)
    log("clips: %d train | %d held-out (%d words)" % (len(train), len(held), held_words))
    if len(train) < 40:
        raise SystemExit("too few training clips")

    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tok.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)
    fe = WhisperFeatureExtractor.from_pretrained(BASE, local_files_only=True)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.float32, local_files_only=True)
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32, local_files_only=True)
    model.config.use_cache = False
    model.to("cuda")

    # ---- labels: [sot, ar, transcribe, notimestamps, text..., eot]
    items = []
    for r in train:
        ids = tok(target_text(r, args.target)).input_ids
        if len(ids) <= 440:
            items.append((r, ids))
    pre = tok.convert_ids_to_tokens(items[0][1][:4])
    log("label prefix: %s ... %s" % (pre, tok.convert_ids_to_tokens(items[0][1][-1:])))
    if (items[0][1][0] != model.config.decoder_start_token_id or items[0][1][-1] != model.config.eos_token_id
            or pre[3] != "<|notimestamps|>"):
        raise SystemExit("unexpected label prefix/eos; stopping before training on wrong targets")
    log("target %s: mean %.1f tokens/clip, max %d" % (args.target, sum(len(i) for _, i in items) / len(items),
                                                     max(len(i) for _, i in items)))
    audio_cache = {r["file"]: read_wav(r["file"]) for r, _ in items}

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    log("trainable parameters: %.1fM" % (sum(p.numel() for p in params) / 1e6))

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
                a = augment(a, rng)
            feats.append(fe(a, sampling_rate=16000, return_tensors="np").input_features[0])
            dis.append(ids[:-1])
            labs.append([-100] * 3 + ids[4:])         # no loss on <|ar|><|transcribe|><|notimestamps|>
        L = max(len(d) for d in dis)
        eot = model.config.eos_token_id
        di = torch.tensor([d + [eot] * (L - len(d)) for d in dis])
        y = torch.tensor([l + [-100] * (L - len(l)) for l in labs])
        x = torch.from_numpy(np.stack(feats))
        return x.to("cuda"), di.to("cuda"), y.to("cuda")

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

    def evaluate(ct2, ep):
        res = decode_rows(ct2, held)
        (e4, n), (e1, _) = score(held, res[4]), score(held, res[1])
        with open(os.path.join(run, "heldout_ep%d.txt" % ep), "w", encoding="utf-8") as f:
            for r, o4, o1 in zip(held, res[4], res[1]):
                f.write("%d:%d %s\n  clean : %s\n  beam4 : %s\n  greedy: %s\n" % (
                    r["surah"], r["ayah"], r["kind"], norm(r["clean"]), norm(o4), norm(o1)))
        return e4, e1, n

    results = []
    e4, e1, n = evaluate(PROD, 0)
    log("epoch 0 (production model): held-out WER beam4 %.1f%% (%d/%d) | greedy %.1f%% (%d/%d)"
        % (100 * e4 / n, e4, n, 100 * e1 / n, e1, n))
    results.append({"epoch": 0, "loss": None, "beam4_err": e4, "greedy_err": e1, "words": n})

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
        e4, e1, n = evaluate(ct2, ep)
        ml = sum(losses) / len(losses)
        log("epoch %d | loss %.3f | held-out WER beam4 %.1f%% (%d/%d) | greedy %.1f%% (%d/%d) | %.0f s elapsed"
            % (ep, ml, 100 * e4 / n, e4, n, 100 * e1 / n, e1, n, time.time() - t0))
        results.append({"epoch": ep, "loss": ml, "beam4_err": e4, "greedy_err": e1, "words": n})
        json.dump(results, open(os.path.join(run, "results.json"), "w"), indent=1)
        torch.cuda.empty_cache()

    best = min(results[1:], key=lambda r: (r["beam4_err"] + r["greedy_err"], -r["epoch"]))
    log("\nbest epoch by held-out errors (beam4 + greedy): %d  -> %s" % (best["epoch"],
        os.path.join(run, "ep%d_ct2" % best["epoch"])))
    log("held-out is only %d words: treat differences of a few words as noise." % n)


if __name__ == "__main__":
    main()
