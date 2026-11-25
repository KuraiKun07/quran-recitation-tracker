# Quran Recitation Tracker

**Recite the Quran out loud and see every word checked as you go, on the page of the Madani mushaf, fully offline on your own PC.**

The tracker listens through the microphone, follows your position word by word and colours each word:
green when it was recited correctly, orange when it is probably a mistake, red for a wrong or skipped word.
It is built for memorisation (hifz) practice: the next words stay hidden until you recite them, and you can ask
for a hint, reveal a word or an ayah, retry an ayah or restart the whole range.

Your voice never leaves the computer. Speech recognition runs locally on the processor or, faster, on an NVIDIA GPU.

---

## Features

- **Live word-by-word tracking** on the 604 pages of the Madani mushaf (one or two pages side by side).
- **Mistake detection**: wrong words, skipped words, extra words, and a **pronunciation check** that catches
  commonly confused letters (for example ح/ه, ع/ء, ق/ك, ص/س, ط/ت, ذ/ز) even when the speech model wrote the word
  correctly.
- **Memorisation mode**: the text appears as you recite it; *Hint*, *Reveal word*, *Reveal ayah*, *Retry ayah*,
  *Restart range*. Click an orange or red word to see what was heard.
- **Practice ranges** by surah and ayahs, by page, or by juz / hizb.
- **Find an ayah by voice**: recite a few words and jump straight to that place.
- **Handles real recitation**: a spoken basmala at the start of a surah, the letter openings (الٓمٓ ...),
  connected speech across words (hamzat al-wasl), restarts and going back to repeat an ayah.
- **Offline and private**: one-click Windows packages that install without internet.

## Download and run (Windows 10/11, 64-bit)

1. Open the [**Releases**](../../releases) page and download one zip:
   - `QuranTracker-<version>-gpu.zip` for a PC with an NVIDIA graphics card (faster), or
   - `QuranTracker-<version>-cpu.zip` for any other PC (smaller; works everywhere).

   The gpu zip is larger than GitHub's 2 GB file limit, so it comes in parts (`.zip.001`, `.zip.002`) with a
   `Join-gpu-zip.bat`: download them into one folder and double-click the .bat to rebuild the zip.
2. Extract the **whole** folder to a short path such as `C:\QuranTracker`.
3. Double-click **`Start.bat`**. The first start sets everything up from the included files
   (a few minutes, no internet needed); later starts are quick.
4. The app opens in your browser (Chrome or Edge). Allow the microphone and start reciting.

Keep the black window open while you use the app; close it to stop. If something goes wrong, the `logs` folder
next to `Start.bat` has the details.

## How it works

```mermaid
flowchart LR
    Mic[Microphone<br/>browser AudioWorklet] -->|16 kHz audio over WebSocket| Buf[Utterance buffer]
    Buf -->|every 0.2-0.4 s| ASR[Speech model<br/>Whisper fine-tuned on Quran<br/>faster-whisper / CTranslate2]
    ASR -->|transcript| Align[Aligner<br/>dynamic programming against<br/>the Uthmani text]
    Align --> Rules[Commit rules<br/>two agreeing passes for a mistake,<br/>look-back, resync]
    Rules -->|green / orange / red| Page[Mushaf page<br/>in the browser]
    Buf -->|end of each utterance| Phone[Pronunciation check<br/>phoneme model, no language model]
    Phone -->|confused letter heard: orange| Page
```

- Whisper is not a streaming model, so the server re-transcribes the growing utterance several times per second.
  Correct words are painted at once; a mistake is only painted when two passes agree on it and a correct word
  follows, so half-heard words are not flagged. A pause ends the utterance with a final, beam-searched pass.
- The aligner matches the transcript against the canonical text with tolerance for the speech model's quirks
  (weak letters, split or fused words, connected speech).
- The **pronunciation check** listens to each utterance again with a phoneme model when you pause. A phoneme model
  has no language model, so unlike Whisper it does not "autocorrect" a wrong letter. Its phonetic script is aligned
  with the expected one (Quran phonetic script), and a word painted green turns orange when a commonly confused
  letter was clearly heard instead. GPU: Muaalem (Wav2Vec2-BERT, ONNX float16, CUDA); processor: HisnLabs
  tajweed-v2 (Whisper-base encoder). Only confused-letter swaps are used: the models' other disagreements are
  mostly noise. It also works the other way: a word Whisper flagged (orange, red or skipped) that the phoneme model
  heard exactly as expected, judged strictly (every sound, at any confidence), turns green again.
- The words recited just before an utterance are given to the model as a context prompt.

## Accuracy

Measured with the project's benchmark (`backend/bench.py`) on 240 recorded sessions: 1,508 words recited
correctly and 46 deliberate mistakes.

| | GPU | Processor |
|---|---|---|
| Correctly recited words shown green | **98.4 %** | **97.9 %** |
| Deliberate mistakes caught (orange or red) | **38 of 46** | **32 of 46** |
| Pronunciation check per utterance (median) | 243 ms | 152 ms |

Before the pronunciation check (app 1.1) both editions showed 97.6 % of correct words green; the GPU edition caught
23 of the 46 mistakes and the processor edition 10.
The speech model was trained on correct recitation, so for many mistakes it writes the correct word anyway; the
phoneme model hears the letter that was actually said. The benchmark's mistakes are mostly confused letters and 46
is a small set: a larger recorded set of mistakes is the next step for measuring.

## Project layout

| Path | What |
|---|---|
| `client-web/index.html` | the whole web app (mushaf pages, microphone, colours, memorisation tools) |
| `backend/server.py` | FastAPI WebSocket server: audio -> speech model -> alignment -> commit rules |
| `backend/alignment.py` | word alignment against the Quran text |
| `backend/phonecheck.py`, `third_party/imam.py` | pronunciation check (phoneme model + HisnLabs rule engine) |
| `backend/phone_convert.py` | builds `backend/phone-model` (ONNX export of Muaalem, copy of tajweed-v2) |
| `backend/ayah_finder.py`, `basmala.py`, `muqattaat.py`, `mushaf.py`, `cudaprep.py` | find by voice, basmala, letter openings, page layout, GPU detection |
| `backend/quran.db` | Quran text and Madani page/line layout (`build_db.py` rebuilds it) |
| `backend/kit/` | Windows release package: launcher, notices, `fetch_offline.py`, `make_package.py` |
| `backend/bench.py`, `replay.py`, `passdiff.py`, `bench_labels.json` | benchmark and regression gate |
| `backend/sim_live.py`, `tape_*.py`, `lettercheck_probe.py`, `*_eval.py` | analysis tools |
| `backend/train*.py`, `ct2conv.py`, `record_server.py` | model training and data recording |
| `scripts/release.ps1` | publish a new GitHub release |

## Development

Requirements: Windows, Python 3.12 (64-bit); an NVIDIA GPU is optional.

```powershell
pip install -r backend\kit\requirements.lock.txt
pip install -r backend\kit\requirements-cuda.txt     # only with an NVIDIA GPU
```

The speech model is not stored in git. Take the `app\backend\model` folder from a release zip and copy it to
`backend\model` (or point `CT2_DIR` at any CTranslate2 model folder). The pronunciation models are not in git
either: copy `app\backend\phone-model` from a gpu release zip to `backend\phone-model` (or build it with
`backend\phone_convert.py`). The GPU check needs onnxruntime-gpu 1.23 (CUDA 12) installed apart from the app's
onnxruntime, e.g. `pip install --no-deps --target C:\ortgpu onnxruntime-gpu==1.23.2` and `PHONE_ORT_DIR=C:\ortgpu`;
without it the processor model is used. Then:

```powershell
cd backend
uvicorn server:app          # open http://localhost:8000/
```

The code defaults are the production settings. Useful environment variables: `HW_DEVICE=cpu` (force the processor),
`PHONE_CHECK=0` (no pronunciation check), `PHONE_T` (its confidence threshold),
`SILENCE_RMS` (raise it if room noise keeps utterances from ending), `TAPE=1` (record sessions to `backend\tapes`
for replay; recordings never leave the PC and are excluded from git).

### Testing a change

Stop the server first (the benchmark uses the GPU). It needs recorded sessions, which are local only.

```powershell
cd backend
python bench.py NAME [--asr] [--set KEY=VALUE]     # score card: false flags, mistakes caught, speed
python bench.py --compare prod_asr NAME             # every word that changed
python passdiff.py prod_asr NAME                    # strict: every pass identical (for clean-ups)
```

A change is kept only if no line of the score card gets worse.

## Releasing

```powershell
cd backend
python kit\fetch_offline.py                         # once: Python, VC++ runtime and wheels -> kit\offline_cache
cd ..
.\scripts\release.ps1 -Tag v0.2.0                    # build both zips, tag, publish the GitHub release
```

See `scripts/release.ps1` for what it checks before publishing.

## A practice aid, not a teacher

The colours can be wrong. The app is meant for self-practice and review; it does not replace a qualified Quran
teacher (shaykh / muqri'), and it must not be used to certify a reciter.

## Privacy

Audio is processed in memory on your computer. Nothing is uploaded, and the app needs no account and no internet
connection. Session recording (`TAPE=1`) is a developer option and is off by default.

## Credits

- Pronunciation check: [obadx/muaalem-model-v3_2](https://huggingface.co/obadx/muaalem-model-v3_2) (MIT,
  [paper](https://arxiv.org/abs/2509.00094)), [HisnLabs/tajweed-v2](https://huggingface.co/HisnLabs/tajweed-v2)
  and its rule engine (Apache 2.0), and the [quran-transcript](https://github.com/obadx/quran-transcript)
  phonetizer (MIT).
- Speech model derived from [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran)
  (Apache 2.0), itself a fine-tune of [OpenAI Whisper](https://github.com/openai/whisper); further fine-tuned for
  this project.
- Quran text and page layout from [zonetecde/mushaf-layout](https://github.com/zonetecde/mushaf-layout).
- Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2)
  and [FastAPI](https://fastapi.tiangolo.com/).

Full third-party notices: [`backend/THIRD_PARTY_NOTICES.txt`](backend/THIRD_PARTY_NOTICES.txt).

## License

No license has been chosen for this project's own code yet; all rights reserved.
