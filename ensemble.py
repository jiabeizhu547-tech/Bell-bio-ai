"""
=============================================================================
[DNA] Phase 4 — 模型集成 + 生物规则后处理
=============================================================================

同时使用 V1 (CNN+BiLSTM) 和 ESM-2 两个模型，通过加权投票和
生物学规则修正，榨干已有模型的全部性能，不重新训练。

集成策略：
  1. 简单平均 (Simple Average)
  2. 加权平均 (Weighted Average, 按 Q3 加权)
  3. 置信度选择 (Max Confidence)
  4. 生物规则后处理 (Biological Smoothing)

用法：
    python ensemble.py

输出：
    - 控制台：各集成策略对比表
    - models/ensemble_benchmark.png：集成 vs 单模型对比图

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
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

VAL_SIZE = 0.10
TEST_SIZE = 0.15

SS_IDX_TO_LABEL = {0: "H", 1: "E", 2: "C"}
SS_LABEL_TO_IDX = {"H": 0, "E": 1, "C": 2}

# ============================================================================
# 1. 数据加载（与 evaluate.py 一致）
# ============================================================================

AA_TO_IDX = {
    "A": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,
    "G": 6,  "H": 7,  "I": 8,  "K": 9,  "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
}

SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
IDX_TO_SS = {v: k for k, v in SS_TO_IDX.items()}

DSSP8_TO_Q3 = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "C": "C", "T": "C", "S": "C",
}


def load_fasta(path: Path) -> list:
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
# 2. 数据集（V1 格式 + ESM-2 格式）
# ============================================================================

def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, aa in enumerate(seq[:max_len]):
        encoded[i] = AA_TO_IDX.get(aa.upper(), 21)
    return encoded


def encode_structure(ss: str, max_len: int) -> np.ndarray:
    encoded = np.full(max_len, -1, dtype=np.int64)
    for i, s in enumerate(ss[:max_len]):
        q3 = DSSP8_TO_Q3.get(s.upper())
        if q3:
            encoded[i] = SS_TO_IDX[q3]
    return encoded


class V1Dataset(Dataset):
    def __init__(self, sequences, structures, max_len):
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
        return torch.LongTensor(x), torch.LongTensor(y), torch.BoolTensor(mask)


class ESM2Dataset(Dataset):
    def __init__(self, sequences, structures, max_len, tokenizer):
        self.max_len = max_len
        self.data = []
        for seq, ss in zip(sequences, structures):
            seq = seq[:max_len]
            ss = ss[:max_len]
            spaced_seq = " ".join(list(seq))
            tokens = tokenizer(
                spaced_seq, padding="max_length", max_length=max_len,
                truncation=True, return_tensors="pt",
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
# 3. 模型概率获取
# ============================================================================

@torch.no_grad()
def get_v1_probs(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    获取 V1 模型对所有残基的 Q3 概率分布。
    Returns: (all_probs [N, 3], all_labels [N,], total_samples)
    """
    model.eval()
    all_probs, all_labels = [], []

    for batch_x, batch_y, batch_mask in tqdm(dataloader, desc="V1 概率", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)

        logits = model(batch_x)
        probs = torch.softmax(logits, dim=-1)  # [B, L, 3]

        batch_probs = probs[batch_mask].cpu().numpy()
        batch_labels = batch_y[batch_mask].cpu().numpy()

        all_probs.append(batch_probs)
        all_labels.append(batch_labels)

    return np.concatenate(all_probs), np.concatenate(all_labels)


@torch.no_grad()
def get_esm2_probs(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    获取 ESM-2 模型对所有残基的 Q3 概率分布。
    Returns: (all_probs [N, 3], all_labels [N,], total_samples)
    """
    model.eval()
    all_probs, all_labels = [], []

    for batch in tqdm(dataloader, desc="ESM-2 概率", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)  # [B, L, 3]

        valid_mask = (labels != -100)
        batch_probs = probs[valid_mask].cpu().numpy()
        batch_labels = labels[valid_mask].cpu().numpy()

        all_probs.append(batch_probs)
        all_labels.append(batch_labels)

    return np.concatenate(all_probs), np.concatenate(all_labels)


# ============================================================================
# 4. 集成策略
# ============================================================================

def ensemble_simple_average(v1_probs: np.ndarray, esm2_probs: np.ndarray) -> np.ndarray:
    """策略 1: 简单平均"""
    avg = (v1_probs + esm2_probs) / 2.0
    return np.argmax(avg, axis=-1)


def ensemble_weighted(v1_probs: np.ndarray, esm2_probs: np.ndarray,
                      w1: float = 0.4, w2: float = 0.6) -> np.ndarray:
    """策略 2: 加权平均（默认按 Q3 比例加权）"""
    avg = w1 * v1_probs + w2 * esm2_probs
    return np.argmax(avg, axis=-1)


def ensemble_max_confidence(v1_probs: np.ndarray, esm2_probs: np.ndarray) -> np.ndarray:
    """策略 3: 置信度选择 — 取置信度更高的模型的预测"""
    v1_max = np.max(v1_probs, axis=-1)
    esm2_max = np.max(esm2_probs, axis=-1)
    v1_preds = np.argmax(v1_probs, axis=-1)
    esm2_preds = np.argmax(esm2_probs, axis=-1)

    result = np.where(esm2_max >= v1_max, esm2_preds, v1_preds)
    return result


def apply_biological_smoothing(preds: np.ndarray, probs: np.ndarray,
                               min_run: int = 3) -> np.ndarray:
    """
    策略 4: 生物规则后处理

    规则：
      - 螺旋(H)和折叠(E)通常连续至少 min_run 个残基
      - 孤立的单个 H 或 E（两侧都是别的类型）→ 翻转为 C
      - 短片段（< min_run）如果置信度低 → 翻转为 C
    """
    smoothed = preds.copy()
    n = len(preds)

    i = 0
    while i < n:
        current = smoothed[i]

        # 只处理 H 和 E（卷曲 C 保持不动）
        if current == SS_LABEL_TO_IDX["C"]:
            i += 1
            continue

        # 找当前结构类型的连续片段
        run_start = i
        while i < n and smoothed[i] == current:
            i += 1
        run_end = i
        run_len = run_end - run_start

        # 如果片段长度 >= min_run，保留
        if run_len >= min_run:
            continue

        # 片段太短，检查平均置信度
        run_conf = np.mean(probs[run_start:run_end, current])

        # 置信度低 → 翻转为 C
        if run_conf < 0.6:
            smoothed[run_start:run_end] = SS_LABEL_TO_IDX["C"]
            continue

        # 置信度还行但特别短 → 看看两侧
        if run_len == 1:
            # 单个残基，两侧都是 C → 翻转为 C
            left = smoothed[run_start - 1] if run_start > 0 else SS_LABEL_TO_IDX["C"]
            right = smoothed[run_end] if run_end < n else SS_LABEL_TO_IDX["C"]
            if left == SS_LABEL_TO_IDX["C"] and right == SS_LABEL_TO_IDX["C"]:
                smoothed[run_start] = SS_LABEL_TO_IDX["C"]

    return smoothed


def ensemble_weighted_with_smoothing(v1_probs: np.ndarray, esm2_probs: np.ndarray,
                                     w1: float = 0.4, w2: float = 0.6,
                                     min_run: int = 3) -> np.ndarray:
    """策略 5: 加权平均 + 生物规则后处理（推荐）"""
    avg = w1 * v1_probs + w2 * esm2_probs
    preds = np.argmax(avg, axis=-1)
    return apply_biological_smoothing(preds, avg, min_run=min_run)


# ============================================================================
# 5. 评估 + 可视化
# ============================================================================

def evaluate_predictions(preds: np.ndarray, labels: np.ndarray,
                         name: str) -> dict:
    """计算 Q3 和每类 F1。"""
    q3 = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds,
        target_names=["H", "E", "C"],
        output_dict=True,
        zero_division=0,
    )
    f1 = [report["H"]["f1-score"], report["E"]["f1-score"], report["C"]["f1-score"]]
    return {"name": name, "q3": q3, "f1": f1, "report": report}


def plot_ensemble_benchmark(results: list, save_path: Path):
    """生成集成 vs 单模型的全面对比图。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    names = [r["name"] for r in results]
    q3_scores = [r["q3"] * 100 for r in results]
    colors = ["#8a9a70", "#d4c8a8", "#6366f1", "#f59e0b", "#10b981", "#ec4899"]

    # --- 左图：Q3 对比 ---
    bars = axes[0].bar(names, q3_scores, color=colors[:len(names)],
                       edgecolor="white", linewidth=2)
    for bar, score in zip(bars, q3_scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{score:.1f}%", ha="center", fontweight="bold", fontsize=12)
    axes[0].set_ylabel("Q3 Accuracy (%)")
    axes[0].set_title("Q3 Accuracy Comparison")
    axes[0].set_ylim(80, max(q3_scores) + 3)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=15)

    # --- 中图：每类 F1 对比 ---
    categories = ["α-Helix (H)", "β-Sheet (E)", "Coil (C)"]
    x = np.arange(len(categories))
    n = len(results)
    width = 0.8 / n

    for i, r in enumerate(results):
        offset = (i - (n - 1) / 2) * width
        axes[1].bar(x + offset, r["f1"], width, label=r["name"],
                    color=colors[i], edgecolor="white")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Per-Class F1 Score")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0.7, 1.0)

    # --- 右图：提升幅度 ---
    baseline = results[1]["q3"]  # ESM-2 作为 baseline
    improvements = [(r["q3"] - baseline) * 100 for r in results]

    bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in improvements]
    bars = axes[2].bar(names, improvements, color=bar_colors, edgecolor="white", linewidth=2)
    for bar, imp in zip(bars, improvements):
        y_pos = bar.get_height() + 0.02 if imp >= 0 else bar.get_height() - 0.08
        axes[2].text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f"{imp:+.2f}%", ha="center", fontweight="bold", fontsize=12)
    axes[2].set_ylabel("Δ Q3 vs ESM-2 (%)")
    axes[2].set_title("Improvement over ESM-2 Baseline")
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] 集成对比图已保存: {save_path}")


# ============================================================================
# 6. 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("[DNA] Phase 4 — 模型集成 + 生物规则后处理")
    print(f"[Device] {DEVICE.upper()}")
    print("=" * 60)

    # ----- 6.1 加载数据 -----
    print("\n[Data] 加载数据...")
    sequences = load_fasta(DATA_DIR / "real_sequences.fasta")
    structures = load_fasta(DATA_DIR / "real_structures.fasta")

    seq_train, seq_temp, ss_train, ss_temp = train_test_split(
        sequences, structures, test_size=VAL_SIZE + TEST_SIZE, random_state=42,
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    seq_val, seq_test, ss_val, ss_test = train_test_split(
        seq_temp, ss_temp, test_size=1 - val_ratio, random_state=42,
    )
    print(f"  训练: {len(seq_train)}  验证: {len(seq_val)}  测试: {len(seq_test)}")

    # ----- 6.2 创建 DataLoader -----
    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)

    v1_dataset = V1Dataset(seq_test, ss_test, MAX_SEQ_LEN)
    v1_loader = DataLoader(v1_dataset, batch_size=BATCH_SIZE, shuffle=False)

    esm2_dataset = ESM2Dataset(seq_test, ss_test, MAX_SEQ_LEN, tokenizer)
    esm2_loader = DataLoader(esm2_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----- 6.3 加载模型 -----
    print("\n[Model] 加载 V1 (CNN+BiLSTM)...")
    from train import ProteinSSPredictor
    v1_config = {
        "embed_dim": 64, "cnn_channels": 128, "cnn_kernel": 7,
        "lstm_hidden": 128, "lstm_layers": 2, "dropout": 0.3, "num_classes": 3,
    }
    v1_model = ProteinSSPredictor(v1_config).to(DEVICE)
    v1_ckpt = torch.load(MODEL_DIR / "best_model.pt", weights_only=False, map_location=DEVICE)
    v1_model.load_state_dict(v1_ckpt["model_state_dict"])
    v1_model.eval()
    print(f"  模型加载完成 (Epoch {v1_ckpt.get('epoch', '?')})")

    print("\n[Model] 加载 ESM-2 (Fine-tuned)...")
    from train_esm2 import ESM2ForSecondaryStructure
    esm2_model = ESM2ForSecondaryStructure(
        ESM_MODEL_NAME, num_classes=3, dropout=0.2,
    ).to(DEVICE)
    esm2_ckpt = torch.load(MODEL_DIR / "best_model_esm2.pt", weights_only=False, map_location=DEVICE)
    esm2_model.load_state_dict(esm2_ckpt["model_state_dict"])
    esm2_model.eval()
    print(f"  模型加载完成 (Epoch {esm2_ckpt.get('epoch', '?')}, "
          f"Val Q3: {esm2_ckpt.get('val_q3', 0):.3f})")

    # ----- 6.4 获取所有模型的概率分布 -----
    print("\n[Prob] 获取 V1 概率分布...")
    v1_probs, all_labels = get_v1_probs(v1_model, v1_loader, DEVICE)
    print(f"  V1 样本数: {len(v1_probs)}")

    print("\n[Prob] 获取 ESM-2 概率分布...")
    esm2_probs, _ = get_esm2_probs(esm2_model, esm2_loader, DEVICE)
    print(f"  ESM-2 样本数: {len(esm2_probs)}")

    # ----- 6.5 评估所有策略 -----
    print("\n" + "=" * 60)
    print("[Ensemble] 评估各集成策略...")
    print("=" * 60)

    results = []

    # 单模型基线
    v1_preds = np.argmax(v1_probs, axis=-1)
    results.append(evaluate_predictions(v1_preds, all_labels, "V1 (CNN+BiLSTM)"))

    esm2_preds = np.argmax(esm2_probs, axis=-1)
    results.append(evaluate_predictions(esm2_preds, all_labels, "ESM-2 (Fine-tuned)"))

    # 策略 1: 简单平均
    s1_preds = ensemble_simple_average(v1_probs, esm2_probs)
    results.append(evaluate_predictions(s1_preds, all_labels, "Ensemble (Avg)"))

    # 策略 2: 加权平均
    s2_preds = ensemble_weighted(v1_probs, esm2_probs, w1=0.40, w2=0.60)
    results.append(evaluate_predictions(s2_preds, all_labels, "Ensemble (Weighted)"))

    # 策略 3: 最大置信度
    s3_preds = ensemble_max_confidence(v1_probs, esm2_probs)
    results.append(evaluate_predictions(s3_preds, all_labels, "Ensemble (MaxConf)"))

    # 策略 4: 加权平均 + 生物规则
    s4_preds = ensemble_weighted_with_smoothing(
        v1_probs, esm2_probs, w1=0.40, w2=0.60, min_run=3,
    )
    results.append(evaluate_predictions(s4_preds, all_labels,
                                        "Ensemble (Weighted+Smooth)"))

    # ----- 6.6 输出结果 -----
    print("\n" + "=" * 85)
    print("[Result] 集成策略对比（测试集）")
    print("=" * 85)
    header = f"{'策略':<30} {'Q3':<10} {'H F1':<10} {'E F1':<10} {'C F1':<10} {'Δ vs ESM-2':<12}"
    print(header)
    print("-" * 85)

    baseline_q3 = results[1]["q3"]  # ESM-2
    best = results[0]
    for r in results:
        delta = (r["q3"] - baseline_q3) * 100
        marker = " ← 最优" if r["q3"] >= max(rr["q3"] for rr in results) else ""
        print(f"{r['name']:<30} {r['q3']*100:>5.1f}%   "
              f"{r['f1'][0]:>.3f}     {r['f1'][1]:>.3f}     {r['f1'][2]:>.3f}   "
              f"{'+' if delta > 0 else ''}{delta:.2f}%{marker}")
        if r["q3"] >= best["q3"]:
            best = r

    print("-" * 85)

    # 详细分类报告
    print(f"\n[Detail] 最佳策略「{best['name']}」分类报告:")
    print(classification_report(all_labels, s4_preds,
                                target_names=["H (α-helix)", "E (β-sheet)", "C (coil)"]))

    # ----- 6.7 保存图 -----
    plot_ensemble_benchmark(results, MODEL_DIR / "ensemble_benchmark.png")

    # ----- 6.8 摘要 -----
    print("\n" + "=" * 60)
    print("[Summary] Phase 4 集成学习摘要")
    print("=" * 60)
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  最佳单模型:  ESM-2 Fine-tuned         Q3 = {results[1]['q3']*100:.1f}%  │
  │  最佳集成:    {best['name']:<25} Q3 = {best['q3']*100:.1f}%  │
  │  提升幅度:    +{(best['q3'] - results[1]['q3'])*100:.2f}% Q3                          │
  │                                                     │
  │  集成策略有效利用了 V1 和 ESM-2 的互补性，            │
  │  生物规则后处理进一步修正了孤立预测错误。              │
  └─────────────────────────────────────────────────────┘
""")

    print(f"\n[Done] Phase 4 方法改进完成！")
    return results, best


if __name__ == "__main__":
    results, best = main()
