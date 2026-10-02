# AHA — dense answer–onset grounding for TMVP surgical video QA

Given a full-length thoracoscopic mitral valve repair video, a question about which instrument is visible when a surgical phase begins, and four options, the model returns **an answer and the second it cites as evidence**. Protocol version: AHA-M1-v2.0, fixed in [proposal.md](proposal.md).

**There are no trained-model results yet.** The code is tested on CPU; feature extraction and training need a GPU server (see [RUNBOOK.md](RUNBOOK.md)).

## Method in one paragraph

Every second of the video is encoded once by a frozen image encoder that never saw TMVP labels. A dilated temporal encoder produces per-second states with dense phase, tool and onset supervision. A linear-chain workflow CRF over the 12 phases turns the phase scores into the posterior probability that a phase *starts* at each second. For each question, the score of (second t, option a) is the sum of onset evidence for the asked phase at t, tool evidence for option a just after t, and a learned residual pair term; a softmax over all pairs in the video gives the answer and the cited second.

## Read in this order

1. [proposal.md](proposal.md): problem, why v1 was amended, architecture, losses, baselines, evaluation, claim limits.
2. [data/DATASET_CARD.md](data/DATASET_CARD.md): what data is here and what it is not.
3. [src/aha/model.py](src/aha/model.py), [src/aha/crf.py](src/aha/crf.py), [src/aha/losses.py](src/aha/losses.py), [src/aha/engine.py](src/aha/engine.py).
4. [review/VALIDATION.md](review/VALIDATION.md): what has and has not been checked.
5. [RUNBOOK.md](RUNBOOK.md): server setup, extraction, training, retrospective test.

## Layout

| Path | Content |
|---|---|
| `data/` | 57 cases as 1 FPS frame ZIPs, labels, splits, original and derived QA |
| `assets/` | SurgFormer checkpoint and MiniLM (bundled); `assets/encoders/` is filled by `scripts/fetch_encoder.py` |
| `src/aha/` | data staging, derived QA, feature extraction, CRF, model, losses, training, evaluation, baselines, optional VLM comparator |
| `configs/` | `full` (proposed), matched baselines and ablations |
| `scripts/` | server stages, retrospective test, seed summary, packaging |
| `artifacts/` | generated caches: text embeddings, per-backbone features |
| `references/` | literature notes, historical PCJD results, the frozen v1.0 snapshot |

Splits: train 30 cases (100 primary original QA + 208 derived), validation 11 cases (44 + 62), test 16 cases (55 primary + 100 derived). The test cases were already inspected during earlier work, so test results are retrospective.

## Quick check (CPU)

```powershell
$env:PYTHONPATH="${PWD}/src;${PWD}/vendor"
python -m pytest -q
python -m aha audit
```

## Data and licence notes

Frames, labels and questions derive from TMVP-SurgVideo and MedHorizon and stay under their original terms; this repository does not relicense them. The git repository holds source code only: `data/`, `assets/`, `vendor/`, `references/` and `proposal.md` are not in git and are assembled on the server from a private dataset by `scripts/fetch_data.py` (see [RUNBOOK.md](RUNBOOK.md), section 2). See [THIRD_PARTY.md](THIRD_PARTY.md).
