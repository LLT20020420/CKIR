# CKIR

Bounded kernel influence redistribution for frozen CLIP, with three-stage lightweight training and native-score GDA fusion.

Author: **Lingtao Lu（卢令韬）**. Version: **1.1.0rc1**, 24 October 2026.
Project-authored code uses the MIT license. Third-party components retain the licenses in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Quick start

Use Python 3.12 with NumPy, SciPy, OSQP and PyTorch. The tested versions are in `ENVIRONMENT.json`. The CUDA build recorded in `requirements-recorded.txt` is an environment record; PyTorch must be obtained from the appropriate CPU/CUDA wheel source. `requirements-core.txt` lists the remaining packages. No installation or download occurs on import.

From this directory:

```powershell
python inspect_package.py
python run.py demo --device cpu --output "D:/ckir-work/demo"
```

The demo generates a small synthetic input, runs all three training stages, and writes separate CKIR, GDA and fused scores. Choose a new output directory. It is a software example, not a reported dataset result. CUDA can also be selected with `--device cuda`.

## Prepare one recorded episode

`data/episodes.json` contains the ordered classes, query IDs, 22 support draws and all 88 clean/noisy label conditions used by the recorded study. Image and frozen-text feature assets are referenced by filename, size and SHA-256; they are not duplicated in the code archive.

Given the three assets listed in [ASSETS.md](ASSETS.md):

```powershell
python run.py prepare --feature-cache "D:/ckir-assets/pets-features.npz" --template "D:/ckir-assets/template.pt" --cupl "D:/ckir-assets/cupl.pt" --phase pilot --seed 2000 --label-version 1 --output "D:/ckir-work/episode"
```

The filenames above may be chosen freely; content hashes must match the recorded index. This command creates `episode.json` and numeric `episode.npz`. Query ground-truth labels are not needed. For other cached-feature data, create a manifest using [INPUTS.md](INPUTS.md).

## Train the retained CKIR recipe

```powershell
python run.py train --stage all --manifest "D:/ckir-work/episode/episode.json" --config configs/retained.json --device cuda --output "D:/ckir-work/training"
```

This writes `text.npz`, `head.npz`, `joint.npz` and matching JSON records. Each stage can also be run explicitly:

```powershell
python run.py train --stage text --manifest "D:/ckir-work/episode/episode.json" --device cuda --output "D:/ckir-work/text.npz"
python run.py train --stage head --manifest "D:/ckir-work/episode/episode.json" --init "D:/ckir-work/text.npz" --device cuda --output "D:/ckir-work/head.npz"
python run.py train --stage joint --manifest "D:/ckir-work/episode/episode.json" --init "D:/ckir-work/head.npz" --device cuda --output "D:/ckir-work/joint.npz"
```

The text stage uses alpha 2.44, 30 AdamW epochs and the retained epoch 25. The aligned head uses the final beta/lambda/tau/alpha = 16/1/0.03/2.51 for seven full-support SGD steps. Joint continuation uses seven steps with fresh SGD/AdamW states, gamma fixed at 100, and a shared backward pass before both optimizer updates. CLIP features and kernel geometry stay fixed.

The first-stage loss is CE plus the original SimNL reranking loss plus eight times text consistency. The next two stages use CE of raw CKIR support scores. The full support residual includes self-kernel terms. Stage parameters are explicit in `configs/retained.json`; score APIs require an explicit alpha.

## Inference and fusion

```powershell
python run.py infer --manifest "D:/ckir-work/episode/episode.json" --checkpoint "D:/ckir-work/training/joint.npz" --device cuda --output "D:/ckir-work/scores.npz"
```

Outputs include `ckir_scores`, `gda_scores`, `fused_scores` and their predictions. GDA uses the original unadapted single-prompt text, KS precision estimator and native float32 scores with alpha 0.1. Fusion promotes the GDA scores to float64 and computes `0.95*CKIR + 0.05*GDA` without another softmax, temperature or normalization. The exact implementation uses `1-rho` for the second coefficient.

Checkpoints retain class order, support IDs, a fingerprint of the training arrays, kernel parameters, stage identity and optimizer traces. Mismatched checkpoint/kernel metadata is rejected. Cached geometry is optional; absent caches are recomputed by the bounded solvers, without support labels in the QP.

## Evaluate saved predictions

The previously used query labels and common zero-shot reference are stored separately in `data/evaluation-labels.json`. Training and inference do not load this file. To compute accuracy, benefit, harm and net-correct counts for an explicitly selected saved output:

```powershell
python run.py evaluate --scores "D:/ckir-work/scores.npz"
```

The score-file hash and query order are checked before counting. This command evaluates that saved episode only; the paper's aggregate denominator is obtained by combining the corresponding recorded support/noise conditions.

## Contents and provenance

- `ckir_gda/`: portable runtime, three-stage training, inference, GDA and episode loading.
- `configs/retained.json`: retained stage parameters and fixed 95/5 fusion.
- `data/episodes.json`: recorded IDs, observed labels and asset checksums.
- `archive/CKIR-SYN02-code-v1/` and its ZIP: byte-preserved historical implementation disclosure.
- `provenance/`: original GDA and score-mixing snapshots plus verified source records.
- `tests/`: small mathematical and archived-implementation parity checks.
- `CITATION.cff`: software citation metadata; no repository URL or DOI is invented before upload.

Run `python -m unittest discover -s tests -v` for the software checks. The archived CUDA parity check runs when CUDA is available; the standalone demo supports CPU. Hardware/runtime changes can affect floating-point arithmetic.

This candidate supplies execution from frozen cached features. Raw-image feature extraction and the large experiment assets are separately described in ASSETS.md. The manuscript and its original result files were not changed by packaging.

See [README.zh-CN.md](README.zh-CN.md) for the Chinese overview and [VALIDATION.md](VALIDATION.md) for the completed checks.
