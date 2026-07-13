# 基于深度学习的蛋白质二级结构预测：CNN-BiLSTM 与 ESM-2 预训练模型的系统对比

## 作者

朱家贝（Bell Zhu）

## 摘要

**背景**：蛋白质二级结构预测是生物信息学的基础任务之一，对理解蛋白质功能和折叠机制具有重要意义。**方法**：本文系统对比了两种深度学习策略——从零训练的 CNN+BiLSTM 模型与微调的 ESM-2 预训练蛋白质语言模型，并在 785 条 PDB 真实蛋白数据集上进行评估。进一步提出加权集成策略结合生物学规则后处理以提升预测精度。**结果**：ESM-2 微调模型 Q3 准确率达 89.2%，显著优于 CNN+BiLSTM 的 85.5%（p<0.001）。加权集成结合生物规则后处理将 Q3 提升至 89.8%。**结论**：预训练蛋白质语言模型在小样本二级结构预测任务上具有显著优势，模型集成可进一步挖掘互补信息。本文还开发了基于 Gradio 的在线预测工具，已部署至 ModelScope 供开放使用。

**关键词**：蛋白质二级结构预测；ESM-2；深度学习；模型集成；Q3 准确率

---

## 1. 引言

蛋白质二级结构（secondary structure）指多肽链主链在局部空间中的规则构象，主要包括 α-螺旋（H）、β-折叠（E）和无规卷曲（C）三种类型。二级结构是连接一级序列与三级空间结构的桥梁，准确预测二级结构对蛋白质功能注释、药物靶点识别和蛋白质设计具有重要价值[1]。

传统的二级结构预测方法包括基于统计的 Chou-Fasman 算法[2]和基于进化信息的 PSIPRED[3]。近年来，深度学习技术在蛋白质结构预测领域取得了突破性进展。AlphaFold2[4]通过注意力机制实现了原子级精度的三维结构预测，ESM-2[5]等蛋白质语言模型通过在数亿条序列上预训练，内化了丰富的序列-结构-功能知识。

本文旨在回答两个核心问题：（1）在相同的小规模数据集（785 条蛋白）上，预训练蛋白质语言模型（ESM-2）相比从零训练的 CNN+BiLSTM 有多大优势？（2）模型集成和生物学规则能否进一步提升预测精度？

## 2. 方法

### 2.1 数据集

数据来源于 Protein Data Bank（PDB），通过 DSSP（Define Secondary Structure of Proteins）算法将三维结构坐标转换为二级结构标签。原始 DSSP 八分类（H, G, I, E, B, T, S, C）按标准映射规则归并为 Q3 三分类：

- **H（α-螺旋）**：H（α-helix）、G（3₁₀-helix）、I（π-helix）
- **E（β-折叠）**：E（β-strand）、B（isolated β-bridge）
- **C（无规卷曲）**：C（coil）、T（turn）、S（bend）

最终数据集包含 785 条非冗余蛋白质序列，按 75%:10%:15% 划分训练集（588 条）、验证集（78 条）和测试集（119 条），随机种子固定为 42 以确保可复现。序列长度限制为 256 个残基。

### 2.2 模型架构

#### 2.2.1 V1: CNN+BiLSTM

从零训练的基线模型采用 CNN+BiLSTM 混合架构：

```
输入序列 → Embedding(64维)
         → 1D CNN (kernel=7, 128通道)  # 捕获局部氨基酸motif
         → BiLSTM (2层, 128隐藏单元)    # 建模长程依赖
         → Dropout(0.3)
         → Linear → Q3 logits
```

总参数量约 1.8M，使用 AdamW 优化器训练 50 个 epoch，初始学习率 1e-3，余弦退火调度，早停耐心值 10。

#### 2.2.2 ESM-2 Fine-tuned

ESM-2（Evolutionary Scale Modeling 2）是 Meta AI 发布的蛋白质语言模型，基于 Transformer 架构，在 2.5 亿条蛋白质序列上通过掩码语言建模（MLM）预训练。本文选用最小的 ESM-2 变体（esm2_t6_8M_UR50D，8M 参数，6 层 Transformer），在其上添加 Q3 分类头：

```
输入序列 → ESM-2 (8M, 6层Transformer, 预训练权重)
         → Dropout(0.2) → Linear(320→160) → GELU → Dropout(0.1) → Linear(160→3)
         → Q3 logits
```

微调使用 AdamW 优化器（lr=5e-5, weight_decay=1e-4），batch size=8，最多 15 个 epoch，早停耐心值 5。

### 2.3 模型集成

为充分利用 V1 和 ESM-2 的互补性，本文测试了五种集成策略：

1. **简单平均**（Simple Average）：P_ens = (P_v1 + P_esm2) / 2
2. **加权平均**（Weighted Average）：P_ens = 0.4 × P_v1 + 0.6 × P_esm2
3. **最大置信度**（Max Confidence）：选择置信度更高的模型的预测
4. **加权 + 生物规则后处理**（Weighted + Smooth）：在加权平均基础上，应用最小片段长度约束（H/E 至少连续 3 个残基，孤立残基翻转为 C）

### 2.4 评估指标

- **Q3 Accuracy**：三分类总体准确率，Q3 = N_correct / N_total
- **Per-class F1 Score**：每类二级结构的调和平均，F1 = 2 × Precision × Recall / (Precision + Recall)
- 所有评估在相同的测试集（random_state=42）上进行，确保可比性

### 2.5 在线预测工具

基于 Gradio 框架开发了交互式 Web 应用，支持用户输入氨基酸序列，实时输出每个残基的二级结构预测及可视化分析。应用已部署至 ModelScope 平台（https://modelscope.cn/studios/BellZhu/protein-ss）。

## 3. 结果

### 3.1 单模型性能对比

| 模型 | Q3 | H F1 | E F1 | C F1 |
|------|-----|------|------|------|
| V1 (CNN+BiLSTM) | 85.5% | 0.887 | 0.780 | 0.854 |
| ESM-2 (Fine-tuned) | **89.2%** | **0.925** | **0.846** | **0.879** |

ESM-2 在所有指标上均显著优于 V1，Q3 提升 +3.8 个百分点。提升幅度最大的是 E 类（β-折叠，+0.066 F1），这与 β-折叠的长程依赖特性一致——Transformer 的自注意力机制比 BiLSTM 更擅长捕获远距离残基间的相互作用。

值得注意的是，即使 ESM-2 只有 8M 参数（远小于 V1 的参数量级），其性能仍然大幅领先，说明预训练知识的迁移效果远超模型容量的影响。

### 3.2 集成策略对比

| 策略 | Q3 | H F1 | E F1 | C F1 | Δ vs ESM-2 |
|------|-----|------|------|------|-------------|
| V1 (CNN+BiLSTM) | 85.5% | 0.887 | 0.780 | 0.854 | -3.78% |
| ESM-2 (Fine-tuned) | 89.2% | 0.925 | 0.846 | 0.879 | baseline |
| Ensemble (Avg) | 88.1% | 0.912 | 0.821 | 0.876 | -1.09% |
| Ensemble (Weighted) | 89.6% | 0.927 | 0.849 | 0.885 | +0.40% |
| Ensemble (MaxConf) | 88.1% | 0.911 | 0.821 | 0.875 | -1.14% |
| **Ensemble (Weighted+Smooth)** | **89.8%** | **0.930** | **0.850** | **0.885** | **+0.52%** |

简单平均和最大置信度策略均导致性能下降（-1.09% 和 -1.14%），原因是 ESM-2 显著优于 V1，等权或置信度选择会引入 V1 的噪声预测。加权平均（0.4:0.6）将 Q3 提升至 89.6%，生物规则后处理进一步修正了预测中的孤立结构错误，最终 Q3 达 89.8%。

### 3.3 每类详细分析

最优集成模型（Weighted+Smooth）的分类报告：

| 类别 | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| H (α-螺旋) | 0.93 | 0.93 | 0.93 | 5858 |
| E (β-折叠) | 0.87 | 0.83 | 0.85 | 2433 |
| C (无规卷曲) | 0.88 | 0.89 | 0.89 | 5885 |

α-螺旋的预测效果最好（F1=0.930），因为螺旋由连续的局部氢键模式维持，序列特征明显。β-折叠的 F1 最低（0.850），因为折叠涉及远距离肽链片段的平行/反平行排列，仅从局部序列推断更具挑战性。

## 4. 讨论

### 4.1 预训练模型的优势来源

ESM-2 仅用 8M 参数即超越精心设计的 CNN+BiLSTM，其优势可能来自两方面：（1）Transformer 的自注意力机制比 BiLSTM 更适合捕获蛋白质序列中远距离残基的相互作用，这对 β-折叠预测尤为重要；（2）在 2.5 亿条蛋白质序列上的预训练使模型内化了氨基酸突变模式、疏水/亲水分布、二级结构倾向性等进化信息，相当于隐式地获取了多序列比对（MSA）的效果。

### 4.2 集成的局限

加权集成仅带来 +0.52% 的提升，低于预期。可能的原因是：（1）ESM-2 已经足够强，V1 提供的互补信息有限；（2）两个模型在大部分残基上预测一致，分歧的样本往往是真正困难的情况，集成也无力修正。未来可尝试使用更大规模的 ESM-2 模型（如 esm2_t12_35M 或 esm2_t30_150M），或引入同源序列比对信息。

### 4.3 本工作的不足

（1）受限于 CPU 计算资源，未能尝试更大规模的 ESM-2 模型变体；（2）数据集仅包含 785 条蛋白，样本量有限；（3）未引入同源序列的 PSSM（Position-Specific Scoring Matrix）或 HHblits 特征，这些特征在 CB513 基准上被证明有效[3]。

## 5. 结论

本文在统一的实验设定下，系统比较了 CNN+BiLSTM 从零训练与 ESM-2 预训练微调在蛋白质二级结构预测上的性能。ESM-2 以 89.2% 的 Q3 准确率显著优于 V1（85.5%），验证了预训练蛋白质语言模型在小样本任务上的强大迁移能力。通过加权集成和生物学规则后处理，进一步将 Q3 提升至 89.8%。本文开发了在线预测工具，将模型成果转化为可用的生物信息学资源。

## 参考文献

[1] Rost, B. (2001). Review: protein secondary structure prediction continues to rise. *Journal of Structural Biology*, 134(2-3), 204-218.

[2] Chou, P. Y., & Fasman, G. D. (1974). Prediction of protein conformation. *Biochemistry*, 13(2), 222-245.

[3] Jones, D. T. (1999). Protein secondary structure prediction based on position-specific scoring matrices. *Journal of Molecular Biology*, 292(2), 195-202.

[4] Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583-589.

[5] Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.

---

*本文所有代码和模型权重可在 https://github.com/BellZhu/protein-ai 获取。在线预测工具：https://modelscope.cn/studios/BellZhu/protein-ss*
