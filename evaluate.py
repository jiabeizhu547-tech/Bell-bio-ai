"""
=============================================================================
[DNA] 统一评估脚本 — V1 vs ESM-2 真实 Benchmark
=============================================================================

在完全相同的训练/验证/测试集划分上，加载三个已保存的模型，计算真实的
Q3 准确率和每类 F1，替代 train_esm2.py 中写死的 V1 分数。

用法：
    python evaluate.py

输出：
    - 控制台：三模型对比表格 + 分类报告
    - models/benchmark_v1_vs_esm2_real.png：真实数据对比图

作者: Bell | 日期: 2026-07-12
=============================================================================
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 0. 配置
# ============================================================================

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MAX_SEQ_LEN = 256
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# 数据划分比例（与 train.py / train_esm2.py 完全一致）
VAL_SIZE = 0.10
TEST_SIZE = 0.15

# ============================================================================
# 1. 数据加载（独立实现，不依赖训练脚本的 import 副作用）
# ============================================================================

AA_TO_IDX = {
    "A": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,
    "G": 6,  "H": 7,  "I": 8,  "K": 9,  "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
}

SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
IDX_TO_SS = {v: k for k, v in SS_TO_IDX.items()}

# DSSP 8 类 → Q3 映射
DSSP8_TO_Q3 = {
    "H": "H", "G": "H", "I": "H",  # α-helix, 3₁₀-helix, π-helix
    "E": "E", "B": "E",             # β-sheet, isolated β-bridge
    "C": "C", "T": "C", "S": "C",   # coil, turn, bend
}


def load_fasta(path: Path) -> list[str]:
    """加载 FASTA 文件，返回序列/结构列表。"""
    items = []
    with open(path) as f:
        for block in f.read().strip().split("\n>"):
            lines = block.strip().split("\n")
            if len(lines) >= 2:
                item = "".join(lines[1:]).replace(">", "")
                if len(item) >= 20:
                    items.append(item)
    return items


# ============================================================================
# 2. V1 模型评估（CNN + BiLSTM）
# ============================================================================

def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """氨基酸序列 → 整数数组。"""
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, aa in enumerate(seq[:max_len]):
        encoded[i] = AA_TO_IDX.get(aa.upper(), 21)
    return encoded


def encode_structure(ss: str, max_len: int) -> np.ndarray:
    """二级结构字符串 → Q3 整数标签，padding 位置为 -1。"""
    encoded = np.full(max_len, -1, dtype=np.int64)
    for i, s in enumerate(ss[:max_len]):
        q3 = DSSP8_TO_Q3.get(s.upper())
        if q3:
            encoded[i] = SS_TO_IDX[q3]
    return encoded


class V1Dataset(Dataset):
    """与 train.py 中 ProteinSSDataset 完全一致。"""

    def __init__(self, sequences: list[str], structures: list[str], max_len: int):
        self.max_len = max_len
        self.data = []
        for seq, ss in zip(sequences, structures):
            x = encode_sequence(seq, max_len)
            y = encode_structure(ss, max_len)
            mask = (y != -1)
            self.data.append((x, y, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, mask = self.data[idx]
        return (
            torch.LongTensor(x),
            torch.LongTensor(y),
            torch.BoolTensor(mask),
        )


# ============================================================================
# 3. ESM-2 模型评估
# ============================================================================

class ESM2Dataset(Dataset):
    """与 train_esm2.py 中 ProteinSSDataset 完全一致。"""

    def __init__(self, sequences: list[str], structures: list[str],
                 max_len: int, tokenizer):
        self.max_len = max_len
        self.data = []

        for seq, ss in zip(sequences, structures):
            seq = seq[:max_len]
            ss = ss[:max_len]

            spaced_seq = " ".join(list(seq))
            tokens = tokenizer(
                spaced_seq,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )

            labels = np.full(max_len, -100, dtype=np.int64)
            for i, s in enumerate(ss):
                q3 = DSSP8_TO_Q3.get(s.upper())
                if q3:
                    labels[i] = SS_TO_IDX[q3]

            self.data.append({
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": torch.LongTensor(labels),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# 4. 评估函数
# ============================================================================

@torch.no_grad()
def evaluate_v1(model, dataloader, device):
    """评估 V1 模型（CNN+BiLSTM）。"""
    model.eval()
    all_preds, all_labels = [], []

    for batch_x, batch_y, batch_mask in tqdm(dataloader, desc="V1 评估", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)

        logits = model(batch_x)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds[batch_mask].cpu().numpy())
        all_labels.extend(batch_y[batch_mask].cpu().numpy())

    q3 = accuracy_score(all_labels, all_preds)
    return q3, all_preds, all_labels


@torch.no_grad()
def evaluate_esm2(model, dataloader, device):
    """评估 ESM-2 模型。"""
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="ESM-2 评估", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)

        valid_mask = (labels != -100)
        all_preds.extend(preds[valid_mask].cpu().numpy())
        all_labels.extend(labels[valid_mask].cpu().numpy())

    q3 = accuracy_score(all_labels, all_preds)
    return q3, all_preds, all_labels


# ============================================================================
# 5. 可视化
# ============================================================================

def plot_benchmark(results: dict, save_path: Path):
    """
    生成双模型对比图。
    results = {
        "V1 (CNN+BiLSTM)":     {"q3": ..., "f1": [H, E, C]},
        "V2 (CNN+Attention)":  {"q3": ..., "f1": [H, E, C]},
        "ESM-2 (Fine-tuned)":  {"q3": ..., "f1": [H, E, C]},
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    model_names = list(results.keys())
    colors = ["#8a9a70", "#d4922a", "#d4c8a8"]

    # 左图：Q3 准确率对比
    q3_scores = [results[m]["q3"] * 100 for m in model_names]
    bars = axes[0].bar(model_names, q3_scores, color=colors, edgecolor="white", linewidth=2)
    for bar, score in zip(bars, q3_scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{score:.1f}%", ha="center", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Q3 Accuracy (%)")
    axes[0].set_title("Q3 Accuracy Comparison (Same Test Set)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.3)

    # 右图：每类 F1 对比
    categories = ["α-Helix (H)", "β-Sheet (E)", "Coil (C)"]
    x = np.arange(len(categories))
    n_models = len(model_names)
    width = 0.8 / n_models

    for i, name in enumerate(model_names):
        offset = (i - (n_models - 1) / 2) * width
        axes[1].bar(x + offset, results[name]["f1"], width,
                    label=name, color=colors[i], edgecolor="white")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Per-Class F1 Score Comparison")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Benchmark 图已保存: {save_path}")


# ============================================================================
# 6. 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("[DNA] 统一评估 — V1 vs V2 vs ESM-2 真实 Benchmark")
    print(f"[Device] {DEVICE.upper()}")
    print("=" * 60)

    # ----- 6.1 加载数据 -----
    print("\n[Data] 加载数据...")
    sequences = load_fasta(DATA_DIR / "real_sequences.fasta")
    structures = load_fasta(DATA_DIR / "real_structures.fasta")
    print(f"  共 {len(sequences)} 条蛋白质序列")

    # 与训练脚本完全一致的数据划分
    seq_train, seq_temp, ss_train, ss_temp = train_test_split(
        sequences, structures,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=42,
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    seq_val, seq_test, ss_val, ss_test = train_test_split(
        seq_temp, ss_temp,
        test_size=1 - val_ratio,
        random_state=42,
    )
    print(f"  训练: {len(seq_train)}  验证: {len(seq_val)}  测试: {len(seq_test)}")

    # ----- 6.2 创建 DataLoader -----
    # V1 DataLoader
    v1_test_dataset = V1Dataset(seq_test, ss_test, MAX_SEQ_LEN)
    v1_test_loader = DataLoader(v1_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ESM-2 DataLoader（需要 tokenizer）
    from transformers import EsmTokenizer
    print("\n[Setup] 加载 ESM-2 tokenizer...")
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)

    esm2_test_dataset = ESM2Dataset(seq_test, ss_test, MAX_SEQ_LEN, tokenizer)
    esm2_test_loader = DataLoader(esm2_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----- 6.3 评估两个模型 -----
    results = {}

    # --- V1 ---
    print("\n" + "-" * 40)
    print("[Eval] V1: CNN + BiLSTM")
    from train import ProteinSSPredictor

    v1_config = {
        "embed_dim": 64, "cnn_channels": 128, "cnn_kernel": 7,
        "lstm_hidden": 128, "lstm_layers": 2, "dropout": 0.3, "num_classes": 3,
    }
    v1_model = ProteinSSPredictor(v1_config).to(DEVICE)
    v1_ckpt = torch.load(MODEL_DIR / "best_model.pt", weights_only=False, map_location=DEVICE)
    v1_model.load_state_dict(v1_ckpt["model_state_dict"])

    v1_q3, v1_preds, v1_labels = evaluate_v1(v1_model, v1_test_loader, DEVICE)
    v1_report = classification_report(v1_labels, v1_preds,
                                       target_names=["H", "E", "C"], output_dict=True)
    v1_f1 = [v1_report["H"]["f1-score"], v1_report["E"]["f1-score"], v1_report["C"]["f1-score"]]
    results["V1 (CNN+BiLSTM)"] = {"q3": v1_q3, "f1": v1_f1}
    print(f"  Q3: {v1_q3:.4f} ({v1_q3*100:.1f}%)")
    print(f"  F1:  H={v1_f1[0]:.3f}  E={v1_f1[1]:.3f}  C={v1_f1[2]:.3f}")

    # --- ESM-2 ---
    print("\n" + "-" * 40)
    print("[Eval] ESM-2: Fine-tuned")
    from train_esm2 import ESM2ForSecondaryStructure

    esm2_model = ESM2ForSecondaryStructure(
        ESM_MODEL_NAME, num_classes=3, dropout=0.2,
    ).to(DEVICE)
    esm2_ckpt = torch.load(MODEL_DIR / "best_model_esm2.pt", weights_only=False, map_location=DEVICE)
    esm2_model.load_state_dict(esm2_ckpt["model_state_dict"])

    esm2_q3, esm2_preds, esm2_labels = evaluate_esm2(esm2_model, esm2_test_loader, DEVICE)
    esm2_report = classification_report(esm2_labels, esm2_preds,
                                          target_names=["H", "E", "C"], output_dict=True)
    esm2_f1 = [esm2_report["H"]["f1-score"], esm2_report["E"]["f1-score"], esm2_report["C"]["f1-score"]]
    results["ESM-2 (Fine-tuned)"] = {"q3": esm2_q3, "f1": esm2_f1}
    print(f"  Q3: {esm2_q3:.4f} ({esm2_q3*100:.1f}%)")
    print(f"  F1:  H={esm2_f1[0]:.3f}  E={esm2_f1[1]:.3f}  C={esm2_f1[2]:.3f}")

    # ----- 6.4 汇总对比 -----
    print("\n" + "=" * 80)
    print("[Benchmark] V1 vs ESM-2 真实对比（相同测试集）")
    print("=" * 80)

    header = f"{'模型':<25} {'Q3':<10} {'H F1':<10} {'E F1':<10} {'C F1':<10}"
    print(header)
    print("-" * 65)

    best_q3 = max(r["q3"] for r in results.values())
    for name, r in results.items():
        marker = " ← 最优" if r["q3"] == best_q3 else ""
        print(f"{name:<25} {r['q3']*100:>5.1f}%   {r['f1'][0]:>.3f}     {r['f1'][1]:>.3f}     {r['f1'][2]:>.3f}{marker}")

    # ESM-2 vs V1 提升
    delta_q3 = (results["ESM-2 (Fine-tuned)"]["q3"] - results["V1 (CNN+BiLSTM)"]["q3"]) * 100
    print(f"\n  ESM-2 相对 V1 提升: +{delta_q3:.1f}% Q3")

    # ----- 6.5 详细分类报告 -----
    print("\n" + "-" * 40)
    print("[Detail] ESM-2 每类详细指标:")
    print(classification_report(esm2_labels, esm2_preds,
                                 target_names=["H (α-helix)", "E (β-sheet)", "C (coil)"]))

    # ----- 6.6 保存对比图 -----
    plot_benchmark(results, MODEL_DIR / "benchmark_v1_vs_esm2_real.png")

    # ----- 6.7 更新 train_esm2.py 中写死的 V1 值 -----
    print(f"\n[Action] 请将 train_esm2.py 第 430-431 行的写死值替换为：")
    print(f"  v1_q3 = {v1_q3:.4f}")
    print(f"  v1_f1 = [{v1_f1[0]:.4f}, {v1_f1[1]:.4f}, {v1_f1[2]:.4f}]")

    print(f"\n[Done] 评估完成！")
    return results


if __name__ == "__main__":
    results = main()
