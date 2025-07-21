
# 🚀 **UniSign-Prompt: Signer Bias Unlearning via Multimodal Prompt Tuning for Cross-Lingual Sign Language Translation**

---

## Overview

**UniSign-Prompt** proposes a novel **prompt-injected architecture** for **continuous sign language translation (SLT)** focused on:
❗ **Signer Bias Unlearning**: minimizes signer-dependent overfitting through adversarial forgetting.

🌐 **Cross-Lingual Generalization**: enables robust transfer across American Sign Language (ASL), German Sign Language (DGS), and Indian Sign Language (ISL).
  
⚡ **Low-Resource Robustness**: superior zero-shot and few-shot ISL performance.

---

## 🎁 **Key Highlights**

📌 **Multimodal Prompt Tuning** using **PI-ST+**, **H-CLPB**, **TAP**, and **PRM** to condition on signer identity, language family, and temporal segments.

✅ **Explicit Signer Bias Unlearning** via **Prompt Forgetting Module (PFM)** applying adversarial forgetting and decorrelation on signer prompts.

✅ **Cross-Lingual Transfer Objective** with **H-CLPB**, enabling scalable alignment across structurally divergent sign languages.

✅ **Temporal-Aware Prompt Adaptation** using **TAP** to handle long sign sequences via temporally adaptive prompts.

✅ **Dynamic Prompt Selection** via **PRM**, reducing inference overhead through Gumbel-softmax-based routing.

✅ **Dual-Branch Decoder** outputs **gloss** and **text** sequences.

✅ **Multi-Objective Forgetting Loss (MOF Loss)** optimizing translation, forgetting, alignment, sparsity, and smoothness.

✅ **Consistent Outperformance** on ASL (How2Sign), DGS (RWTH-PHOENIX14T), and ISL (ISL-CSLTR).

---

## 🖼️ **Architecture Overview**

### 🔷 **Overall System Architecture**
![Overview Architecture](docs/overview_architecture_placeholder.png)

### 🟣 **Detailed Architecture with Module Breakdown**
![Detailed Architecture](docs/detailed_architecture_placeholder.png)

Module references in [`models/`](models/):
- `prompt_injected_sign_transformer.py` → **PI-ST+** (Prompt-Injected Sign Transformer)
- `hierarchical_prompt_bank.py` → **H-CLPB**
- `tap_module.py` → **TAP**
- `prompt_routing_mechanism.py` → **PRM**
- `prompt_forgetting_module.py` → **PFM**
- `decoder.py` → **Dual-Branch Decoder**
- `unisign_prompt.py` → Overall **UniSign-Prompt** model integration

---

## 📂 **Dataset Structure**

Folder structure under `datasets/`:

```plaintext
datasets/
├── how2sign_[train/val/test]_meta.csv
├── rwth_[train/val/test]_meta.csv
├── isl_[train/val/test/zero_shot/few_shot_*]_meta.csv
├── [language]_src_vocab.txt
├── [language]_trg_vocab.txt
```

| Dataset | Language | Train/Val/Test | Signers | Gloss Vocab | Target Language |
| -------- | -------- | --------------- | ------- | ------------ | ---------------- |
| How2Sign | ASL | 29k/3k/3k | 11 | 16k | English |
| RWTH-PHOENIX14T | DGS | 7096/519/642 | 9 | 1066 | German |
| ISL-CSLTR | ISL | 500/100/100 | 7 | 1036 | English |
| ISL Zero-Shot | ISL | 0/100/100 | 7 | 1036 | English |
| ISL Few-Shot | ISL | 50/100/100 | 7 | 1036 | English |

---

## ⚙️ **Setup**

```bash
pip install -r requirements.txt
```

Training:
```bash
python train.py --dataset How2Sign
```

Evaluation:
```bash
python evaluate_extended.py --dataset ISL-CSLTR --split zero_shot --checkpoint checkpoints/ISL-CSLTR_UniSignPrompt_best.pth
```

---

## 🏆 **Results Summary**

### 🔵 **Translation Quality (BLEU-4 ↑, WER ↓)**

| Dataset | BLEU-4 ↑ | WER ↓ |
| -------- | -------- | ------ |
| ASL | 23.0 | 38.4 |
| DGS | 22.3 | 37.4 |
| ISL Zero-Shot | 14.5 | 46.0 |
| ISL Few-Shot | 17.0 | 43.4 |

### 🟣 **Signer Bias Unlearning**

| Dataset | Signer Accuracy ↓ | BLEU-4 Gap ↓ |
| -------- | ----------------- | ------------- |
| DGS | 13.4% | 1.4 |
| ISL Zero-Shot | 18.6% | 1.9 |
| ISL Few-Shot | 16.7% | 1.5 |

### 🟢 **Efficiency**

| Dataset | Params ↓ | Latency ↓ |
| -------- | -------- | --------- |
| ASL | 49.6M | 58.7 ms/frame |
| DGS | 49.6M | 55.2 ms/frame |
| ISL | 49.6M | 53.8 ms/frame |

---

## 📜 **Citation**

```bibtex
@article{unistprompt2025,
  title={Signer Bias Unlearning via Multimodal Prompt Tuning for Cross-Lingual Sign Language Translation},
  author={Elakkiya R},
  year={2025}
}
```

---

## 📝 **License**

MIT License
