# Recorded assets

The code archive contains the complete 22-draw/88-condition identifier and label index in `data/episodes.json`. Its `assets` section gives the actual filenames, byte sizes and SHA-256 checksums for:

1. The frozen Pets trainval feature cache: 3680×1024 image features, 37×1024 original single-prompt text features and matching image IDs.
2. The original template-text tensor: 1024×37.
3. The original CuPL-text tensor: 1024×37.

The three assets total 15,822,346 bytes and are retained separately from the code archive. Use `run.py prepare` to verify their hashes and assemble a selected recorded input. No trained checkpoint is required to start the three-stage training command; `infer` requires a checkpoint produced or converted to the documented format.

Dataset source: https://www.robots.ox.ac.uk/~vgg/data/pets/ . The index preserves the study's existing trainval split and never requests the official test split. Raw images, the CLIP RN50 model and feature extraction are not automatically downloaded by this package. Keep third-party data/model terms separate from the software license.

For distribution, the existing feature/text assets can be deposited as a separate input archive, linked from the final code release and the manuscript's Data Availability section. This local candidate does not invent that archive's online location, repository URL or DOI. To use other frozen features, supply your own numeric episode manifest according to INPUTS.md.
