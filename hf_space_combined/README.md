---
title: Protein AI - Secondary Structure & Enzyme Classification
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: true
license: mit
---

# 🧬 Protein AI

**Deep learning-powered protein sequence analysis** — predict secondary structure AND enzyme function from amino acid sequences.

### Features

- **🔬 Secondary Structure Prediction** — Per-residue Q3 classification (H/E/C) with **89.8% accuracy** using ESM-2 ensemble
- **🧪 EC Enzyme Classification** — 7-class enzyme prediction with **87.3% accuracy**, trained on 33,331 UniProt sequences

### Models

| Model | Architecture | Performance |
|-------|-------------|-------------|
| SS Prediction | ESM-2 (6-layer Transformer) + CNN-BiLSTM Ensemble | Q3 89.8% |
| EC Classifier | ESM-2 embeddings + MLP | Accuracy 87.3% |

### How to Use

1. Go to the **🧬 二级结构预测** tab to predict per-residue secondary structure
2. Go to the **🧪 EC 酶分类** tab to classify enzyme function
3. Paste an amino acid sequence and click "Predict"

### Citation

If you use this tool, please cite:
- Lin et al. (2023) ESM-2, *Science*
- Bell Zhu (2026) Protein AI

### Links

- [GitHub Repository](https://github.com/jiabeizhu547-tech/Bell-bio-ai)
- [ModelScope Mirror](https://modelscope.cn/studios/BellZhu/protein-ss)
