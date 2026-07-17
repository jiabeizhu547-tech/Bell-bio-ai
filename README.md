# 🧬 Protein AI — 蛋白质序列智能分析

> **从氨基酸序列预测蛋白质二级结构 & 酶功能分类**
>
> 朱家贝（Bell Zhu）| 生物信息学 × 深度学习 | 2026.07

[![ModelScope](https://img.shields.io/badge/ModelScope-在线演示-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMSAxNy45M2MtMy45NS0uNDktNy0zLjg1LTctNy45MyAwLS40NS4wNC0uODkuMTEtMS4zMmw1Ljg5IDUuODl2My4zNnptMS0zLjM2TDUuNjggMTEuMjRjLjg3LTIuOTcgMy40MS01LjE4IDYuNDItNS42MS4zLjMuNTkuNjIuODkuOTZsNS44OSA1Ljg5Yy0uMyAzLjMtMi45IDYtNi4yOSA2LjY5eiIvPjwvc3ZnPg==)](https://modelscope.cn/studios/BellZhu/protein-ss)
[![GitHub](https://img.shields.io/badge/GitHub-Bell--bio--ai-black?logo=github)](https://github.com/jiabeizhu547-tech/Bell-bio-ai)

---

## 📋 项目概览

| Phase | 模块 | 方法 | 核心指标 |
|-------|------|------|----------|
| **Phase 1** | 二级结构预测 — 从零建模 | CNN + BiLSTM (V1) | Q3 **85.5%** |
| **Phase 2** | 二级结构预测 — 预训练微调 | ESM-2 Fine-tuned | Q3 **89.2%** |
| **Phase 3** | Web 工具部署 | Gradio → ModelScope | 🟢 在线运行 |
| **Phase 4** | 模型集成 + 论文 | Weighted Ensemble + Bio Smooth | Q3 **89.8%** |
| **Phase 5** | 酶功能分类 | ESM-2 嵌入 + MLP | Acc **87.3%** |

---

## 🔬 模块一：蛋白质二级结构预测

### 任务定义

输入一条氨基酸序列，对每个残基预测其二级结构类别：

| 标签 | 结构类型 | 视觉表示 |
|------|----------|----------|
| **H** | α-螺旋 (Alpha Helix) | 🔵 螺旋状 |
| **E** | β-折叠 (Beta Sheet) | 🔴 箭头状 |
| **C** | 无规卷曲 (Coil) | 🟡 线状 |

### 模型架构

```
序列输入: "MKFLILFNILV..."
    │
    ├──→ V1: Embedding → 1D CNN(k=7) → BiLSTM(2层) → Linear → Q3
    │
    └──→ ESM-2: 6层 Transformer (8M, 预训练) → Classifier Head → Q3
              │
              └──→ Ensemble: 0.4×V1 + 0.6×ESM-2 → Bio Smooth → 最终预测
```

### Benchmark 结果

```
模型                          Q3       H F1     E F1     C F1
──────────────────────────────────────────────────────────────
V1 (CNN+BiLSTM)              85.5%    0.887    0.780    0.854
ESM-2 (Fine-tuned)           89.2%    0.925    0.846    0.879
Ensemble (Weighted+Smooth)   89.8%    0.930    0.850    0.885  ← 最佳
```

ESM-2 相对 V1 提升 **+3.8% Q3**。集成 + 生物规则平滑在此基础上再提升 **+0.52%**。

### 生物规则后处理

- 最小片段长度约束：H/E 至少连续 **3 个残基**
- 孤立残基修正：单个 H/E 被 C 包围时翻转为 C
- 低置信度短片段修正

---

## 🧪 模块二：EC 酶功能分类

### 任务定义

输入一条蛋白质序列，预测其酶委员会（EC）一级分类（7 类）：

| EC 编号 | 类别 | 数据量 | F1 Score |
|---------|------|--------|----------|
| EC 1 | 氧化还原酶 Oxidoreductase | 4,905 | **0.86** |
| EC 2 | 转移酶 Transferase | 4,908 | **0.83** |
| EC 3 | 水解酶 Hydrolase | 4,763 | **0.85** |
| EC 4 | 裂合酶 Lyase | 4,196 | **0.81** |
| EC 5 | 异构酶 Isomerase | 4,722 | **0.90** |
| EC 6 | 连接酶 Ligase | 4,858 | **0.88** |
| EC 7 | 转位酶 Translocase | 4,979 | **0.97** |

### 方法

```
蛋白质序列
    ↓ ESM-2 (冻结, 特征提取)
    ↓ Mean Pooling → 320-dim 嵌入
    ↓ MLP: 320→160→64→7
    ↓ Dropout(0.3) + BatchNorm + GELU
    ↓ EC 类别概率
```

- **数据来源**：UniProt REST API，SwissProt 已审阅条目
- **训练集**：33,331 条真实酶序列（7 类均衡采样）
- **测试准确率**：**87.3%**（随机基线 14.3%）
- **模型大小**：仅 266 KB（MLP 分类头）

---

## 🚀 在线演示 · Live Demo

> **🔗 永久免费 · Permanent Free URL**

### ModelScope 魔搭社区（阿里云）

👉 **https://modelscope.cn/studios/BellZhu/protein-ss**

[![ModelScope](https://img.shields.io/badge/🚀-在线试用_Live_Demo-6366f1?style=for-the-badge)](https://modelscope.cn/studios/BellZhu/protein-ss)

| 特性 Feature | 说明 Detail |
|-------------|-------------|
| 💰 费用 | **完全免费**（2vCPU / 16GB RAM / 50GB 存储） |
| ⚡ 速度 | 首次启动约 3-5 分钟（模型下载），之后秒开 |
| 🔓 注册 | 无需登录即可使用 |
| 🌍 域名 | 永久有效，不会过期 |
| 🎛️ 功能 | 二级结构预测 · EC 酶分类 · 突变效应分析 |

### 网站首页 · Landing Page

👉 **https://jiabeizhu547-tech.github.io/Bell-bio-ai/**

项目介绍、安装指南、性能基准一站查看。

---

## ⚡ 快速开始 · Quick Start

### 1️⃣ 克隆项目 · Clone

```bash
git clone https://github.com/jiabeizhu547-tech/Bell-bio-ai.git
cd Bell-bio-ai
```

### 2️⃣ 安装依赖 · Install Dependencies

```bash
# 创建虚拟环境（推荐）
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

# 安装依赖
pip install -r requirements.txt
```

### 3️⃣ 启动本地服务 · Launch

```bash
# 方式一：Gradio Web 应用（推荐）
python app.py
# → 浏览器自动打开 http://localhost:7860

# 方式二：API 后端 + 前端页面
python server.py
# → 打开 index.html 或访问 http://localhost:5000
```

### 4️⃣ 推理与训练 · Inference & Training

```bash
# === 二级结构预测 ===
python inference.py                     # 单条序列推理
python train.py                         # 训练 V1 (CNN+BiLSTM)
python train_esm2.py                    # 微调 ESM-2
python evaluate.py                      # 评估所有模型

# === EC 酶分类 ===
python protein_function/train_ec_classifier.py  # 训练（自动从 UniProt 下载数据）
```

---

## 🤖 安装为 Claude Code 技能 · Install as Claude Code Skill

将 Protein AI 注册为 Claude Code 技能，在对话中用 `/protein-ai` 直接调用蛋白质分析功能。

Register Protein AI as a Claude Code skill to invoke protein analysis with `/protein-ai` in conversations.

### 方法一：项目级技能（推荐）· Project-level Skill

在项目根目录执行以下命令创建技能文件：

Run these commands in your project root to create the skill file:

```bash
mkdir -p .claude/skills

cat > .claude/skills/protein-ai.md << 'SKILL_EOF'
---
name: protein-ai
description: >-
  Predict protein secondary structure (Q3 89.8%),
  EC enzyme classification (Acc 87.3%), and mutation
  effects (zero-shot LLR scoring) from amino acid
  sequences. Invoke when the user asks about protein
  analysis, structure prediction, enzyme function,
  or mutation pathogenicity.
allowed-tools: [Bash, Read, Write]
---

# Protein AI Skill

## 功能 · What it does

- **🧬 二级结构预测**: 输入氨基酸序列 → 逐残基 H/E/C 分类，Q3 89.8%
- **🧪 EC 酶分类**: 输入序列 → 7 类酶功能预测，Acc 87.3%
- **⚡ 突变效应**: 输入点突变 → ESM-2 零样本 LLR 致病性打分

## 使用方式 · How to use

### 本地推理 · Local Inference

    # 确保已安装依赖
    pip install -r requirements.txt

    # 二级结构预测
    python inference.py --sequence "MKVLILACLVALALACTVQ..."

    # EC 酶分类
    python -c "
    from protein_function.train_ec_classifier import predict
    print(predict('MSKVQVTGSVLKAAAVDAVAAAGYPVEITGDLKRLGFKGVFIEK'))
    "

### 在线 Demo · Online Demo

无需安装：https://modelscope.cn/studios/BellZhu/protein-ss

No installation: https://modelscope.cn/studios/BellZhu/protein-ss

## 触发词 · Triggers

用户说以下内容时自动调用：
- "预测这个蛋白质的二级结构"
- "这个酶属于哪一类"
- "分析这个突变是否致病"
- "predict secondary structure of this protein sequence"
- "classify this enzyme"
- "score this mutation"

## 模型信息 · Model Info

| 模型 | 架构 | Q3 / Acc | 大小 |
|------|------|----------|------|
| V1 | CNN + BiLSTM | 85.5% | 8.6 MB |
| ESM-2 | Transformer 6L | 89.2% | 29 MB |
| Ensemble | ESM-2 + V1 + Bio | 89.8% | 38 MB |
| EC 分类器 | ESM-2 embed + MLP | 87.3% | 266 KB |
SKILL_EOF
```

> 💡 也可以手动创建 `.claude/skills/protein-ai.md`，把上面 `---` 到 `SKILL_EOF` 之间的内容复制进去。

### 方法二：用户级技能（所有项目可用）· User-level Skill

将上述文件复制到 `~/.claude/skills/protein-ai.md`，所有项目均可使用 `/protein-ai`。

Copy the file to `~/.claude/skills/protein-ai.md` to make it available across all projects.

```bash
# Windows (PowerShell)
Copy-Item .claude/skills/protein-ai.md $env:USERPROFILE\.claude\skills\protein-ai.md

# macOS / Linux
cp .claude/skills/protein-ai.md ~/.claude/skills/protein-ai.md
```

### 使用 · Usage

安装后在 Claude Code 中输入：

After installation, in Claude Code:

```
/protein-ai 分析这段序列: MKVLILACLVALALACTVQAKTENPKKT
```

或直接说 / or just say:

```
用 Protein AI 预测这个突变是否致病: TP53 R175H
```

---

## 📁 项目结构 · Project Structure

```
protein-ai/
├── app.py                           # Gradio Web 应用
├── train.py                         # V1 训练 (CNN+BiLSTM)
├── train_esm2.py                    # ESM-2 微调训练
├── ensemble.py                      # 模型集成 + 生物规则平滑
├── evaluate.py                      # 统一评估脚本
├── inference.py                     # 推理封装
├── visualize_3d.py                  # 3D 结构可视化 (PyMOL)
├── fetch_pdb_data.py                # PDB 数据批量下载
├── fetch_batch.py                   # 批量数据获取
├── download_data.py                 # 合成数据生成
├── requirements.txt                 # Python 依赖
├── paper.md                         # 小论文
│
├── models/                          # 训练产物
│   ├── best_model.pt                # V1 权重 (8.6 MB)
│   ├── best_model_esm2.pt           # ESM-2 权重 (30 MB) → .gitignore
│   └── benchmark_*.png              # Benchmark 对比图
│
├── data/                            # 数据集
│   ├── real_sequences.fasta         # PDB 真实序列 (785 条)
│   ├── real_structures.fasta        # 对应 Q3 标签
│   ├── demo_sequences.fasta         # 合成 Demo 数据
│   └── enzyme/
│       ├── enzyme_sequences.fasta   # UniProt 酶序列 (33,331 条)
│       └── enzyme_labels.txt        # EC 类别标签
│
├── protein_function/                # 蛋白质功能预测
│   ├── train_ec_classifier.py       # EC 分类器训练脚本
│   └── models/
│       ├── ec_classifier.pt         # 模型权重 (266 KB)
│       └── ec_classification_results.png
│
├── outputs/                         # 可视化输出
│   └── 3d_vis/                      # 5 个蛋白的 3D 结构图
│
└── hf_space/                        # ModelScope 部署（独立仓库）
    ├── app.py                       # 部署版 Gradio 应用
    └── esm_model/                   # ESM-2 本地模型文件
```

---

## 🛠 技术栈

| 用途 | 工具/库 |
|------|---------|
| 深度学习框架 | PyTorch 2.x |
| 预训练模型 | HuggingFace Transformers (ESM-2) |
| 数据处理 | NumPy, Pandas, BioPython |
| 评估指标 | scikit-learn |
| 可视化 | Matplotlib, PyMOL |
| Web 演示 | Gradio |
| 数据源 | PDB, UniProt REST API |
| 部署平台 | ModelScope Studio |

---

## 📚 参考资料

- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123–1130.
- Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.
- Jones, D. T. (1999). Protein secondary structure prediction based on position-specific scoring matrices. *Journal of Molecular Biology*, 292(2), 195–202.

---

## 📧 联系方式

- **作者**：朱家贝（Bell Zhu）
- **GitHub**：[@jiabeizhu547-tech](https://github.com/jiabeizhu547-tech)
- **ModelScope**：[BellZhu/protein-ss](https://modelscope.cn/studios/BellZhu/protein-ss)
