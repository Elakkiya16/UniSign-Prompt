# Dataset assets

This directory combines the original repository data with the supplied revision-2 dataset assets.

## CSL annotations

- `CSL-Daily-groundtruth-train.stm`
- `CSL-Daily-groundtruth-dev.stm`
- `CSL-Daily-groundtruth-test.stm`
- `gloss_dict.npy`
- `Archive.zip` (the supplied annotation bundle)

The STM files are retained in their original format. The training loader consumes CSV manifests with `sample_id`, `video_path` (or `feature_path`), `text_path`, optional `gloss_path`, and annotated `signer_id`. Prepare these manifests and token arrays before selecting the CSL training config; the STM files are not silently converted into signer labels or text/gloss pairs.

## Additional assets

- `how2sign_realigned_test.csv`
- `Frames_Sentence_Level-20260809T140820Z-1-001.zip`
- `Frames_Word_Level-20260809T131313Z-1-001.zip`
- `fsl_chunk_aa`, `fsl_chunk_ab`, `fsl_chunk_ac`, `fsl_chunk_ad`
- `vit_b_16-c867db91.pth`

Existing How2Sign, PHOENIX14T and ISL manifests and vocabularies are retained. Dataset-loading Python modules belong to the updated codebase; the supplied older loader copies do not replace them.

Large archives, chunks and the checkpoint are tracked through `.gitattributes` using Git LFS. When adding this release to a Git checkout, run `git lfs install` before staging. The ViT checkpoint is included as supplied; loading it into PI-ST+ requires an explicit compatible weight mapping.
