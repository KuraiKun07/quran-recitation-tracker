# phone_convert.py - dev tool (needs torch + transformers + onnx, none of them shipped; a CUDA GPU for the export).
# Builds backend\phone-model\ for phonecheck.py:
#   muaalem-phonemes-fp16.onnx  Muaalem (obadx/muaalem-model-v3_2, MIT) Wav2Vec2-BERT + its PHONEME CTC head only,
#                               exported in float16 (run by onnxruntime-gpu, CUDA)
#   phone_config.json           the 43 CTC labels (0 = blank)
#   phone_fbank.npz             its feature extractor's mel filters + window (the app computes the features in numpy)
#   tajweed-v2-fp32.onnx + tajweed-v2.json   HisnLabs/tajweed-v2 (Apache-2.0) for PCs without a GPU (onnxruntime CPU)
#   python phone_convert.py MUAALEM_DIR TAJWEED_V2_DIR [--out phone-model] [--onnx-lib DIR]
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("muaalem")
    ap.add_argument("tajweed")
    ap.add_argument("--out", default=os.path.join(HERE, "phone-model"))
    ap.add_argument("--onnx-lib", default=None, help="folder with the onnx package, if it is not installed")
    a = ap.parse_args()
    if a.onnx_lib:
        sys.path.append(a.onnx_lib)
    import numpy as np, torch
    import onnx
    from safetensors.torch import load_file
    from transformers import AutoFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel
    cfg = json.load(open(os.path.join(a.muaalem, "config.json"), encoding="utf-8"))
    voc = json.load(open(os.path.join(a.muaalem, "vocab.json"), encoding="utf-8"))["phonemes"]
    n = cfg["level_to_vocab_size"]["phonemes"]
    assert sorted(voc.values()) == list(range(n))
    keep = {k: v for k, v in cfg.items() if not k.startswith("level_to_") and k not in ("architectures", "model_type", "auto_map")}
    enc = Wav2Vec2BertModel(Wav2Vec2BertConfig(**keep)).eval()
    sd = load_file(os.path.join(a.muaalem, "model.safetensors"))
    missing, unexpected = enc.load_state_dict({k[len("wav2vec2_bert."):]: v for k, v in sd.items() if k.startswith("wav2vec2_bert.")}, strict=False)
    if missing or unexpected:
        sys.exit("weights do not fit: missing %s unexpected %s" % (missing[:5], unexpected[:5]))
    head = torch.nn.Linear(cfg["output_hidden_size"], n)
    head.load_state_dict({"weight": sd["level_to_lm_head.phonemes.weight"], "bias": sd["level_to_lm_head.phonemes.bias"]})

    class Phonemes(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.enc, s.head = enc, head
        def forward(s, feats):
            return torch.log_softmax(s.head(s.enc(feats.half())[0]).float(), -1)

    m = Phonemes().eval().half().cuda()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "muaalem-phonemes-fp16.onnx")
    torch.onnx.export(m, (torch.randn(1, 300, 160).cuda(),), fn, input_names=["features"], output_names=["logprobs"],
                      dynamic_axes={"features": {0: "batch", 1: "frames"}, "logprobs": {0: "batch", 1: "frames"}},
                      opset_version=17, dynamo=False)
    onnx.save(onnx.load(fn), fn)                  # one file (float16 fits in 2 GB)
    for f in os.listdir(a.out):                   # external-data leftovers of the export
        if not f.endswith((".onnx", ".json", ".npz")):
            os.remove(os.path.join(a.out, f))
    labels = [None] * n
    for k, i in voc.items():
        labels[i] = k
    json.dump({"labels": labels, "blank": 0, "source": "obadx/muaalem-model-v3_2 (MIT), phoneme head only, float16"},
              open(os.path.join(a.out, "phone_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    fe = AutoFeatureExtractor.from_pretrained(a.muaalem)
    np.savez(os.path.join(a.out, "phone_fbank.npz"), mel=np.asarray(fe.mel_filters, np.float64), window=np.asarray(fe.window, np.float64))
    shutil.copyfile(os.path.join(a.tajweed, "onnx", "model-fp32.onnx"), os.path.join(a.out, "tajweed-v2-fp32.onnx"))
    tc = json.load(open(os.path.join(a.tajweed, "config.json"), encoding="utf-8"))
    json.dump({"phonemes": tc["phonemes"], "riwayat": tc["riwayat"], "source": "HisnLabs/tajweed-v2 (Apache-2.0), onnx/model-fp32.onnx"},
              open(os.path.join(a.out, "tajweed-v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", a.out, sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
