# Input and checkpoint contract

## Episode JSON

The required fields are `format: ckir-gda-episode-v1`, `episode_id`, `class_names`, `support_ids`, `query_ids`, `support_seed`, `label_version` and `arrays`.

`class_names` is an ordered list. Numeric class index i always denotes `class_names[i]`. Support and query IDs must be unique, disjoint and in the same row order as the corresponding feature matrices. `arrays` contains a relative `path` and the file's actual `sha256`. Paths are resolved against the JSON directory, or the explicit `--data-root`.

`support_seed` and `label_version` define the recorded text-fit RNG seed `234000 + 100*support_seed + label_version`. The recorded index uses version 0 for clean labels and 1, 2, 3 for the three noisy realizations.

## Numeric NPZ fields

| Key | Shape | Meaning |
|---|---|---|
| support | N × d | Frozen support image features, original row order; float64 in CKIR math. |
| query | Q × d | Frozen query image features; no labels required. |
| observed_labels | N | Integer support labels in 0..C-1. |
| template | d × C | Initial positive template text, stored as original float16 values. |
| cupl | d × C | Initial CuPL text, same ordered classes. |
| gda_text | d × C | Original unadapted single-prompt text used by GDA, float32. It is distinct from the learned CKIR text mixture. |

All arrays are numeric and finite; object/pickle arrays are not accepted. The retained real setting is N=296, Q=592, d=1024, C=37. The portable implementation also accepts smaller dimensions for software examples. For GDA, every class must have at least one observed support sample.

The image-feature cache accepted by `prepare` contains `image_features`, `ids`, `text_features` and the original optional `logit_scale`. `text_features` has shape C×d and is transposed into `gda_text`. The `template.pt` and `cupl.pt` assets are tensors loaded with `weights_only=True`. Array memory order is preserved when assembling/loading numeric inputs.

## Optional precomputed geometry

The NPZ may include `text_support_weights`, `text_query_weights`, `final_support_weights` or `final_query_weights`. Support matrices have shape N×N; query matrices N×Q, with columns denoting target samples. A corresponding `cached_geometry.text` or `cached_geometry.final` dictionary in the JSON must specify matching `beta`, `lambda` and `tau` values. Coefficients must obey the configured box bound. Without these fields the package computes coefficients itself.

Only the geometry copies of feature rows are renormalized for the kernel. Probability and GDA computations use the supplied frozen features. Support cross-kernels retain the recorded jitter convention. The kernel QP is label-independent and jointly solves all support coefficients for each target.

## Checkpoints

`text.npz` contains `res_template`, `res_cupl`, `text_features`. The last field is C×d, matching the historical saved learned-text convention. Head and joint checkpoints also contain `weight` (C×d), `bias` (C), `gamma` (C). Gamma is fixed at 100 in this recipe.

The matching `.json` stores `format: ckir-gda-checkpoint-v1`, the actual NPZ hash, stage, kernel configuration, class names, support IDs, episode ID, support fingerprint, full recipe and trace. The fingerprint binds support features, observed support labels and the two initial positive-text arrays. New query rows can therefore be evaluated with the same support/checkpoint while preserving the training identity.

Head training requires a text-stage checkpoint; joint training requires a head-stage checkpoint. Each stage creates fresh optimizer state as in the retained historical recipe. Inference uses the checkpoint's saved learned-text vectors, preserving their original float16 values before float64 scoring. Existing output files are not silently overwritten.
