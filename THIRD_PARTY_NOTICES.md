# Third-party notices

The top-level MIT license applies to the CKIR project-authored code by Lingtao Lu. It does not replace the following third-party terms.

## SimNL

Repository: https://github.com/zhangce01/SimNL . Fixed commit: `a0871bcf31a607a7ec8acc99aff82f41ed7eaff3`.

`ckir_gda/losses.py` contains the original `SmoothCrossEntropy` and `adaptive_reranking_loss` declarations. Their function bodies are preserved, with source attribution in the file. The original `main.py` and `utils.py` snapshots remain inside the frozen Code S1 archive.

The fixed upstream README explicitly displays an MIT license badge linking to https://opensource.org/licenses/MIT . That README is reproduced in `provenance/SimNL-README.md`; a reference copy of the MIT permission text is included in `LICENSES/SimNL-MIT-reference.txt`. The fixed source tree did not contain a separate LICENSE/COPYING/NOTICE file. No upstream copyright-holder line is invented here; the original source and README attribution are retained.

## GDA via ProKeR

Repository: https://github.com/ybendou/ProKeR . Fixed local source commit: `f2ecb3d2a0d7e0993f977e550286b34f06015edb`.

`ckir_gda/gda.py` adapts the GDA function from `trainers/gda.py`. The upstream Apache License 2.0 is included unchanged in `LICENSES/ProKeR-Apache-2.0.txt`; the original file is retained as `provenance/gda-proker-unmodified.py`.

Local changes expose a reusable fixed-alpha estimator, retain CPU float32 operations, remove validation-grid selection/printing from the runtime API, and expose the recorded float64 score fusion. The original implementation credits *A Hard-to-Beat Baseline for Training-free CLIP-based Adaptation*, ICLR 2024. Its author-source reference is https://github.com/mrflogs/ICLR24/tree/560fe211700964c915442ff98dbe47dd09656a35 ; the corresponding fixed commit metadata and source hash are retained. The author repository's full training script is referenced rather than redistributed in this candidate.

## Dependencies and assets

NumPy, SciPy, OSQP and PyTorch are dependencies, not vendored in this archive; their licenses remain with their distributions. Oxford-IIIT Pets and CLIP assets have their own terms. See ASSETS.md for the input boundary and recorded checksums.
