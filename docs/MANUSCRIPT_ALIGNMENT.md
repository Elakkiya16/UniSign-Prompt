# Implementation details

## Module mapping

| Manuscript equations | Code |
|---|---|
| 1–7: training/deployment mappings | `models/unisign_prompt.py` |
| 8–14: prompt-injected frozen encoder | `models/prompt_injected_sign_transformer.py` |
| 15–22: residual language/signer partitions | `models/hierarchical_prompt_bank.py` |
| 23–28: alignment and temporal prompting | `models/tap_module.py` and model integration |
| 29–33: Top-K routing and entropy | `models/prompt_routing_mechanism.py` |
| 34–37: pooled-feature gradient reversal and decorrelation | `models/prompt_forgetting_module.py` |
| 38–41: causal gloss/text decoding | `models/decoder.py` |
| 42: multi-objective loss | `losses/multi_objective_forgetting_loss.py` |

## Backbone checkpoints

`training.backbone_checkpoint` accepts a tensor state dictionary with exactly the `patch_embed.*`, `input_proj.*` and `layers.*` keys from `model.pi_st`, without the `pi_st.` prefix. Shapes must match the configured model. These parameters stay frozen; per-layer `alpha` and `prompt_residual.*` are trained adaptation parameters.

The supplied `datasets/vit_b_16-c867db91.pth` is preserved as a separate ViT checkpoint. Its keys must be explicitly mapped and architecture compatibility checked before it can serve as a PI-ST+ backbone checkpoint.

## Encoder and decoder

RGB inputs use 16×16 patch embeddings and sinusoidal spatial/temporal positions. A separate feature projection supports precomputed inputs. Segment boundaries use each video's valid frame count. Each segment's prompt-conditioned queries retain global video context.

The default decoder uses two causal layers per active branch. Predicted gloss distributions condition text during training; generated gloss embeddings condition text at inference. Beam search uses no length penalty. These are explicit implementation defaults for details not fixed in the manuscript.

## Routing and ablations

Hard Top-K routing selects five of twenty bank slots by default. Reported activation ratio counts selected bank slots and excludes TAP tokens. Variable effective-activation rates require a separately specified pruning rule.

- `use_pfm=false`: remove adversarial and decorrelation losses.
- `use_tap=false`: use routed prompts without temporal/alignment losses.
- `use_routing=false`: inject the full available bank without routing entropy.
- `use_hclpb=false`: use a static prompt bank; retain pooled-feature adversarial CE when signer labels exist, without partition decorrelation/alignment.

## Evaluation definitions

BLEU uses unsmoothed, case-sensitive NLTK corpus scoring. ROUGE-L uses mean sentence LCS F1. METEOR uses NLTK WordNet. Chinese text uses character tokens. Gloss-WER and Text-WER are reported separately. Latency includes encoding and generation after warm-up, excluding data loading and detokenization.

The independent signer probe is a 256-unit ReLU MLP with dropout 0.3. The membership utility implements a loss-threshold attack on balanced, disjoint calibration/evaluation populations; the exact attack and forget-subset definitions must be held consistent when comparing retraining gaps.
