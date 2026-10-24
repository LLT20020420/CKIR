# Completed software validation

Checked on 24 October 2026 using the pre-existing recorded runtime in ENVIRONMENT.json and an NVIDIA GeForce RTX2060. No new dependencies were installed.

## Portable execution

A small synthetic CPU example ran through text training, aligned-head training, joint continuation, bounded-kernel inference, native GDA and 95/5 fusion. Its reduced training budget and synthetic dimensions are saved with the example outputs, independently of the retained research recipe.

## Archived-implementation parity

On the same synthetic CUDA inputs:

- Text positive vectors and gradients of the complete first-stage loss matched the archived CKIRText/loss declarations exactly.
- The aligned head's seven updates matched the archived aligned-head fit parameters exactly.
- The joint stage's seven updates matched the archived joint-fit parameters and saved text vectors exactly.
- Native float32 GDA scores matched the archived ProKeR GDA function exactly at fixed alpha 0.1.
- A small bounded quadratic program matched the independent active-set enumeration solution.
- Score APIs require explicit alpha; input, checkpoint and cache checks reject mismatched identities/settings.

## One saved real condition

For the already recorded pilot support seed 2000, noisy-label version 1, the retained final checkpoint and cached geometry were loaded without retraining. Across its 592 existing queries:

- Maximum CKIR score difference from saved scores: 4.912736883966318e-15.
- Maximum GDA score difference: 0.
- CKIR, GDA and fixed95/5 fused predictions all matched the saved records.

No query ground-truth labels were loaded for that comparison, and no accuracy was computed or selected. The original manuscript, original Code S1 ZIP and historical result files were not modified. This check covers the packaged path for one recorded condition; it does not claim a new independent performance experiment or a full raw-image reproduction.

The release integrity manifest fixes every distributed file except the manifest itself. The outer ZIP checksum fixes the complete archive. Author/license metadata were supplied by the author; no public URL or DOI is asserted before upload.
