# UniSign-Prompt

**Prompt-Space Signer Unlearning with Parameter-Efficient Cross-Lingual Sign Language Translation**

UniSign-Prompt combines a frozen visual encoder with language-conditioned prompts, temporal prompting, sparse video-conditioned routing, and prompt-space signer unlearning. Signer-adaptive prompts are used during annotated training and disabled at inference.

## Architecture

![UniSign-Prompt overview](docs/Figure_1.png)

![Figure 2: UniSign-Prompt architecture](docs/Figure_2.png)

The model includes PI-ST+, a Hierarchical Cross-Lingual Prompt Bank, Temporal-Aware Prompt Injection, a Prompt Routing Mechanism, a Prompt Forgetting Module, and a dual gloss/text decoder. The multi-objective loss combines translation, forgetting, alignment, routing entropy, prompt regularization, and temporal smoothness.

## Setup

Use Python 3.10–3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For METEOR evaluation:

```bash
python -m nltk.downloader wordnet omw-1.4
```

## Datasets

| Dataset | Translation | Config |
|---|---|---|
| How2Sign | ASL → English | `configs/how2sign_config.yaml` |
| RWTH-PHOENIX14T | DGS → German | `configs/rwth_phoenix14t_config.yaml` |
| ISL-CSLTR | ISL → English | `configs/isl_csltr_config.yaml` |
| CSL-Daily | CSL → Chinese | `configs/csl_daily_config.yaml` |

The `datasets/` directory includes the supplied updated metadata, CSL train/dev/test STM annotations, gloss dictionary, frame archives, archive chunks and ViT checkpoint. See [dataset files and formats](datasets/README.md).

Configure manifest paths, tokenizers, vocabulary sizes and pretrained backbone weights in the relevant YAML file. Token arrays use `PAD=0`, `BOS=1`, and `EOS=2`. English uses a shared How2Sign/ISL BPE tokenizer; German and Chinese use separate tokenizers. How2Sign uses text supervision without gloss or signer labels.

```bash
python tools/train_bpe.py --corpora /data/how2sign_train.txt /data/isl_train.txt \
  --output datasets/shared_english_bpe.json --vocab-size 5000
python tools/tokenize_text.py --csv /data/train_transcripts.csv \
  --tokenizer datasets/shared_english_bpe.json --output-dir datasets/text_tokens
```

## Training

Set `training.backbone_checkpoint` to compatible pretrained PI-ST+ weights, or use `--init-checkpoint` for transfer initialization.

```bash
python train.py --config configs/rwth_phoenix14t_config.yaml --seed 42
```

Training uses AdamW, linear learning-rate decay, gradient clipping and validation BLEU-4 checkpoint selection. The encoder stays frozen. Model flags `use_pfm`, `use_tap`, `use_routing` and `use_hclpb` control component ablations. Loss coefficients are configured in YAML.

## Evaluation

```bash
python evaluate_extended.py --checkpoint checkpoints/rwth/seed_42/best.pt \
  --split test --output results/rwth_seed42.json
python evaluate_leakage.py --checkpoint checkpoints/rwth/seed_42/best.pt \
  --output results/rwth_signer.json
```

Translation uses beam size 5 without signer metadata. Outputs include BLEU-1/2/3/4, METEOR, ROUGE-L, Gloss-WER or Text-WER, parameter counts, model latency and prompt activation ratio. The independent signer probe uses pooled deployment features.

```bash
python tools/compare_results.py results/ours.json results/baseline.json --resamples 1000
python tools/summarize_seeds.py results/seed42.json results/seed43.json results/seed44.json
```

Membership analysis is available through `export_membership_losses.py` and `evaluate_membership.py`. It uses a loss-threshold attack fitted on a separate calibration split. `tools/retain_split.py` prepares retained-training manifests; `tools/retraining_gap.py` reports measured ΔSA and ΔMIA.

See [implementation details](docs/MANUSCRIPT_ALIGNMENT.md) for the backbone checkpoint format and module definitions.

## Adding the data to Git

Large dataset files use Git LFS via the included `.gitattributes`. In your Git checkout, enable LFS before staging the updated files:

```bash
git lfs install
git add .gitattributes datasets docs README.md
```

## License

[MIT License](LICENSE.md).
