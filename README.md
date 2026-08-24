# Épée ASL Toolkit

**Python toolkit for the [CLERC Épée](https://huggingface.co/datasets/CLERC-DATA/epee) American Sign Language keypoint corpus - load, inspect, visualize and benchmark multi-signer ASL data.**

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CLERC--DATA%2Fepee-yellow?style=flat-square)](https://huggingface.co/datasets/CLERC-DATA/epee)
[![Code License](https://img.shields.io/badge/Code-MIT-green?style=flat-square)](LICENSE)
[![Data License](https://img.shields.io/badge/Data-CC%20BY--NC--SA%204.0-blue?style=flat-square)](https://huggingface.co/datasets/CLERC-DATA/epee)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)](https://www.python.org/)

Épée is an open, AI-grade ASL corpus: **1,200 clips, 71 minutes of signing, 6 native Deaf signers**,
released as MediaPipe keypoints with expert gloss segmentation. Five of the six signers cover a
**parallel grid of 201 phrases** - the same sentence, signed by each of them - which makes it one of
the few public sign language datasets where signer variation can be isolated from phrase variation.
The sixth signer contributes 200 declarative sentences on a separate prompt set, a second register.

This repo is the code that makes it usable in five minutes.

---

## Install

```bash
git clone https://github.com/melouxflorian-jpg/epee-asl-toolkit
cd epee-asl-toolkit
pip install -r requirements.txt
```

Clips are downloaded lazily from the Hub and cached, so nothing is pulled until you ask for it.

## Quickstart

```python
import epee

clip = epee.load_clip("clerc_v03_0002")

clip.text_en           # "Do you see?"
clip.signer_id         # "ALPHA"
clip.keypoints.shape   # (158, 128, 3)  - frames × landmarks × (x, y, z)
clip.glosses           # ['D-O', 'YOU', 'SEE', 'QUESTION']

clip.slr_keypoints()   # (158, 118, 3) - lower body dropped, the usual SLR input
clip.hands()           # (158, 42, 3)  - both hands only

for seg in clip.segments:
    start, end = clip.segment_frames(seg)   # gloss boundaries as frame indices
    window = clip.segment_keypoints(seg)
```

The parallel structure, which is the point of this corpus:

```python
clips = epee.parallel_group(7)   # phrase #7, every signer on the grid
[(c.signer_id, round(c.duration_s, 2)) for c in clips]
# [('ALPHA', 7.1), ('BRAVO', 3.47), ('CHARLIE', 3.42), ('DELTA', 3.97), ('ECHO', 1.87)]
```

Pairing goes through the `phrase_id` column, never through arithmetic on clip ids: block offsets
change between releases and the grid is not perfectly rectangular. `epee.grid_coverage()` tells you
how many signers rendered each phrase before you assume a group size.

---

## What signer variation actually looks like

`python examples/02_signer_variability.py`

Phrase #7, *“How old are you?”*, signed by five native Deaf signers:

| Signer | Duration | Gloss sequence | Signing space (h × v) |
|---|---|---|---|
| ALPHA | 7.10s | `HOW OLD YOU QUESTION OLD YOU QUESTION` | 2.02 × 2.96 |
| BRAVO | 3.47s | `YOU OLD` | 2.32 × 2.63 |
| CHARLIE | 3.42s | `OLD YOU` | 1.84 × 2.50 |
| DELTA | 3.97s | `OLD QUESTION OLD YOU` | 0.86 × 1.34 |
| ECHO | 1.87s | `OLD YOU` | 1.85 × 2.94 |

Same phrase, same language, same recording protocol. A **3.8× spread in duration**, a **2.7× spread in
horizontal signing space**, and gloss orders that differ between signers - `YOU OLD` versus `OLD YOU`.

This is the variance a model trained on one signer never sees, and it is why single-signer sign
language datasets do not transfer.

![cross-signer benchmark](https://huggingface.co/datasets/CLERC-DATA/epee/resolve/main/benchmark.png)

---

## The benchmark this corpus was built to support

From the dataset's own [BENCHMARK.md](https://huggingface.co/datasets/CLERC-DATA/epee/blob/main/BENCHMARK.md) - a small BiLSTM, tested on signers held
**entirely outside** the training set:

| Training signers | Accuracy on an unseen signer | Macro-F1 |
|---|---|---|
| 1 | 29% | 0.17 |
| 2 | 47% | 0.30 |
| 3 | 57% | 0.39 |
| 4 | 63% | 0.47 |
| 5 | 65% | 0.51 |
| **6** | **69%** | **0.57** |

Chance is 4.2%. A signer the model *has* seen scores 73%, so the stranger gap closes from 43 points
at one training signer to **3 points at six**. The curve has not flattened, in signers or in data.

**Sign language AI is data-bound, not model-bound.** A bigger model does not close the
one-signer-to-six-signer gap, because it is a property of the data.

---

## API

| | |
|---|---|
| `load_metadata()` | master index as a DataFrame, one row per clip |
| `load_clip(clip_id)` | a single `Clip` - keypoints, glosses, timings |
| `load_clips(signer=..., limit=...)` | bulk load, optionally filtered |
| `parallel_group(n)` | phrase *n* as signed by every signer on the grid |
| `grid_coverage()` | how many signers rendered each grid phrase |
| `normalize_by_shoulders(kp)` | remove body size and camera framing |
| `resample(seq, n)` | fixed-length sequences for batching |
| `base_gloss(g)` | collapse variants (`SIGN_2` → `SIGN`), keep directionals |
| `download_all()` | fetch the whole corpus at once |
| `visualize.plot_clip_strip(clip)` | filmstrip of a clip |
| `visualize.plot_signer_comparison(clips)` | one phrase, every signer, aligned |

### Keypoint layout - 128 landmarks per frame

| Indices | Region | Source |
|---|---|---|
| 0–20 | Left hand | MediaPipe Hands |
| 21–41 | Right hand | MediaPipe Hands |
| 42–53 | Upper body | MediaPipe Pose |
| 54–63 | Lower body | MediaPipe Pose - extrapolated, clips are framed waist-up |
| 64–91 | Eyes + mouth | MediaPipe Face - privacy-preserving subset |
| 92–127 | Head silhouette | MediaPipe FaceMesh `FACE_OVAL` - outline only |

No internal facial features are included beyond eyes and mouth. `(0, 0, 0)` marks an undetected
landmark. Coordinates are MediaPipe image-normalized space and are **not** clipped to `[0, 1]`.

**Skeletons are body data.** Removing the video removes the face, not the body: limb proportions act
as a soft biometric, and the pseudonyms are stable identities rather than anonymity. Using this data
to identify or track individuals is prohibited under the dataset licence. See the dataset card's
ethical considerations.

---

## Licensing

Code in this repo is **MIT**. The dataset is **CC BY-NC-SA 4.0** - free for research and education,
not for commercial use.

The public release ships keypoints and annotations only; source video stays proprietary. Commercial
licensing and access to the full multi-signer corpus: **[florian@clerc.io](mailto:florian@clerc.io)**.

## Citation

```bibtex
@misc{clerc_epee_2026,
  title  = {CLERC Épée v0.3: A Multi-Signer ASL Keypoint Corpus},
  author = {Meloux, Florian},
  year   = {2026},
  url    = {https://huggingface.co/datasets/CLERC-DATA/epee}
}
```

---

## About CLERC

[CLERC](https://clerc.io) builds the data infrastructure layer for sign language in multimodal AI.
Text and images have a data layer; sign language does not. We produce native-signer ASL and LSF
corpora with expert annotation and a versioned schema - open pilot releases for research, tiered
licensing for foundation model labs.

🌐 [clerc.io](https://clerc.io) · 🤗 [Hugging Face](https://huggingface.co/CLERC-DATA) · ✉️ [florian@clerc.io](mailto:florian@clerc.io)

<sub>Topics: sign language dataset · American Sign Language · ASL · sign language recognition · SLR · sign language translation · gloss annotation · MediaPipe keypoints · pose estimation · multimodal AI · Deaf accessibility · machine learning dataset</sub>
