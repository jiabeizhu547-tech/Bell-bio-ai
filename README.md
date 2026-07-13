# 🧬 Protein Secondary Structure Prediction

> **从氨基酸序列预测蛋白质二级结构（Q3）**
>
> 朱家贝 | 生物医药 × AI | 2026.07

---

## 项目概览

本项目包含四个阶段的工作：

| Phase | 内容 | 模型 | Q3 准确率 |
|-------|------|------|-----------|
| **Phase 1** | 从零搭建 CNN+BiLSTM，在 PDB 真实数据上训练 | V1 (CNN+BiLSTM) | **85.5%** |
| **Phase 2** | 微调 Meta ESM-2 预训练模型，系统对比分析 | ESM-2 (Fine-tuned) | **89.2%** |
| **Phase 3** | Gradio Web 工具，部署至 ModelScope 上线 | ESM-2 (Fine-tuned) | 89.2% |
| **Phase 4** | 模型集成 + 生物规则后处理 + 小论文 | Ensemble (Weighted+Smooth) | **89.8%** |

### Real Benchmark（evaluate.py 实测，相同测试集）

```
模型                        Q3       H F1     E F1     C F1
────────────────────────────────────────────────────────────
V1 (CNN+BiLSTM)            85.5%    0.887    0.780    0.854
V2 (CNN+Attention)         81.7%    0.864    0.706    0.816
ESM-2 (Fine-tuned)         89.2%    0.925    0.846    0.879  ← 最佳单模型
Ensemble (Weighted+Smooth) 89.8%    0.930    0.850    0.885  ← 最佳集成
```

ESM-2 相对 V1 提升 **+3.8% Q3**，所有类别 F1 均有显著提升。
集成（Weighted+Smooth）在 ESM-2 基础上再提升 **+0.52% Q3**，通过加权投票和生物学规则后处理实现。

---

## 项目背景

蛋白质二级结构（α-螺旋 H、β-折叠 E、无规卷曲 C）是理解蛋白质功能和折叠机制的基础。

本项目做了两件事：
1. **从零建模**：用 CNN + BiLSTM 从 PDB 数据学习序列→结构的映射
2. **站在巨人肩上**：微调 Meta 在 2.5 亿条蛋白质序列上预训练的 ESM-2 模型

---

## Phase 3: Web 工具

在线预测工具已部署至 ModelScope，输入序列即可获得逐残基二级结构预测：

🔗 **https://modelscope.cn/studios/BellZhu/protein-ss**

## Phase 4: 方法改进 & 小论文

### 模型集成

通过加权平均（V1 权重 0.4，ESM-2 权重 0.6）+ 生物学规则后处理（最小螺旋/折叠片段 ≥ 3 残基），将 Q3 提升至 **89.8%**，超过任一单模型。

### 小论文

见 [paper.md](paper.md)，包含摘要、引言、方法、结果、讨论、结论和参考文献。

---

## 快速开始

```bash
# 1. 激活虚拟环境
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 训练 V1（从零开始）
python train.py

# 4. 微调 ESM-2（需要 HuggingFace 连接）
python train_esm2.py

# 5. 统一评估所有模型
python evaluate.py

# 6. 单条序列预测
python inference.py
```

---

## 项目结构

```
protein-ai/
├── data/                          # 数据集（785 条 PDB 真实蛋白）
│   ├── real_sequences.fasta       # 氨基酸序列
│   ├── real_structures.fasta      # Q3 结构标签 (H/E/C)
│   ├── demo_sequences.fasta       # 合成数据（用于测试流程）
│   └── demo_structures.fasta
├── models/                        # 训练好的模型
│   ├── best_model.pt              # V1: CNN+BiLSTM (8.3 MB)
│   ├── best_model_v2.pt           # V2: CNN+Attention (14 MB)
│   ├── best_model_esm2.pt         # ESM-2 Fine-tuned (29 MB)
│   ├── training_curve*.png        # 训练曲线
│   └── benchmark_v1_vs_esm2_real.png  # 真实 Benchmark 对比图
├── outputs/3d_vis/                # 3D 结构可视化（5 个蛋白）
├── train.py                       # Phase 1: V1 训练脚本
├── train_v2.py                    # Phase 1: V2 训练脚本（实验性）
├── train_esm2.py                  # Phase 2: ESM-2 微调脚本
├── evaluate.py                    # 统一评估脚本（真实 Benchmark）
├── inference.py                   # 推理封装（供 Phase 3 Web 工具调用）
├── visualize_3d.py                # 3D 结构可视化
├── fetch_pdb_data.py              # PDB 数据批量下载
├── download_data.py               # 合成数据生成器
├── requirements.txt               # Python 依赖
└── README.md
```

---

## 模型架构

### V1: CNN + BiLSTM（从零训练）

```
氨基酸序列
    ↓ Embedding (64-dim)
    ↓ 1D CNN (kernel=7, 128 channels)  ← 捕获局部 motif
    ↓ BiLSTM (2 layers, 128 hidden)    ← 捕获长程依赖
    ↓ Dropout (0.3)
    ↓ Linear → Q3 logits (H/E/C)
```

### ESM-2: 预训练 Transformer + 分类头（微调）

```
氨基酸序列
    ↓ ESM-2 (8M params, 6-layer Transformer)
    ↓   — 在 2.5 亿条蛋白质序列上预训练
    ↓   — 内化了序列-结构-功能知识
    ↓ Dropout → Linear → GELU → Dropout → Linear
    ↓ Q3 logits (H/E/C)
```

---

## 评估指标

- **Q3 Accuracy**：三级分类准确率（>70% 为合格基线，>85% 为优秀）
- **Per-class F1**：每类二级结构的预测质量（兼顾 precision 和 recall）

---

## 技术栈

- PyTorch 2.x
- HuggingFace Transformers（ESM-2）
- scikit-learn / NumPy / Pandas / Matplotlib
- BioPython
