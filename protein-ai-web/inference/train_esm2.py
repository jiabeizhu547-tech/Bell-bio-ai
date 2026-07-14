"""
=============================================================================
[DNA] ESM-2 微调 — 蛋白质二级结构预测 (Q3)
=============================================================================

Phase 2: 使用 Meta 的 ESM-2 预训练蛋白质语言模型，在相同的 PDB 数据集上
微调二级结构预测任务，与 Phase 1 的 CNN+BiLSTM 模型做 Benchmark 对比。

ESM-2 模型在 2.5 亿条蛋白质序列上预训练，已经内化了大量蛋白质
序列-结构-功能的知识，只需少量标注数据即可达到高精度。

模型: facebook/esm2_t6_8M_UR50D (8M 参数，适合 CPU 训练)
任务: Token-level Q3 分类 (H=α螺旋, E=β折叠, C=卷曲)

作者: Bell | 日期: 2026-07
=============================================================================
"""

import os
# 国内用户通过镜像访问 HuggingFace
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 0. 全局配置
# ============================================================================

CONFIG = {
    "data_dir": Path(__file__).parent / "data",
    "model_dir": Path(__file__).parent / "models",
    "max_seq_len": 256,
    "test_size": 0.15,
    "val_size": 0.10,

    # ESM-2 模型选择
    # esm2_t6_8M:    8M 参数, 最快
    # esm2_t12_35M:  35M 参数, 平衡
    # esm2_t30_150M: 150M 参数, 需要 GPU
    "model_name": "facebook/esm2_t6_8M_UR50D",

    # 训练
    "batch_size": 8,         # ESM 模型较大，batch 要小
    "learning_rate": 5e-5,   # 预训练模型用更低的 LR
    "epochs": 15,            # ESM 收敛快，不需要太多 epoch
    "early_stop_patience": 5,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "dropout": 0.2,

    # 输出
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. 数据处理（复用 train_v2.py 的数据逻辑）
# ============================================================================

AA_TO_IDX = {
    "A": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,
    "G": 6,  "H": 7,  "I": 8,  "K": 9,  "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
}

SS_TO_IDX = {"H": 0, "E": 1, "C": 2}
IDX_TO_SS = {v: k for k, v in SS_TO_IDX.items()}


def load_data():
    """加载本地 FASTA 数据。"""
    seq_file = CONFIG["data_dir"] / "real_sequences.fasta"
    ss_file = CONFIG["data_dir"] / "real_structures.fasta"

    sequences, structures = [], []

    with open(seq_file) as f:
        for block in f.read().strip().split("\n>"):
            lines = block.strip().split("\n")
            if len(lines) >= 2:
                seq = "".join(lines[1:]).replace(">", "")
                if len(seq) >= 20:
                    sequences.append(seq)

    with open(ss_file) as f:
        for block in f.read().strip().split("\n>"):
            lines = block.strip().split("\n")
            if len(lines) >= 2:
                ss = "".join(lines[1:]).replace(">", "")
                if len(ss) >= 20:
                    structures.append(ss)

    print(f"[Data] 共 {len(sequences)} 条蛋白质序列")
    return sequences, structures


class ProteinSSDataset(Dataset):
    """蛋白质二级结构预测数据集，输出 ESM tokenizer 格式。"""

    def __init__(self, sequences: list[str], structures: list[str],
                 max_len: int, tokenizer):
        self.max_len = max_len
        self.data = []

        # 用 tokenizer 将氨基酸序列转为 ESM 格式
        for seq, ss in zip(sequences, structures):
            seq = seq[:max_len]
            ss = ss[:max_len]

            # ESM tokenizer 接受空格分隔的氨基酸
            spaced_seq = " ".join(list(seq))
            tokens = tokenizer(
                spaced_seq,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )

            # 结构标签（仅对非 padding 位置有效）
            labels = np.full(max_len, -100, dtype=np.int64)  # -100 = ignore in CrossEntropy
            for i, s in enumerate(ss):
                if s in SS_TO_IDX:
                    labels[i] = SS_TO_IDX[s]
                elif s == "G":     # 310-helix → H
                    labels[i] = SS_TO_IDX["H"]
                elif s == "I":     # pi-helix → H
                    labels[i] = SS_TO_IDX["H"]
                elif s == "B":     # isolated beta-bridge → E
                    labels[i] = SS_TO_IDX["E"]
                elif s in ("T", "S"):  # turn, bend → C
                    labels[i] = SS_TO_IDX["C"]

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
# 2. ESM-2 微调模型
# ============================================================================

class ESM2ForSecondaryStructure(nn.Module):
    """在 ESM-2 预训练模型上加 Q3 分类头。"""

    def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.2,
                 local_files_only: bool = False):
        super().__init__()
        from transformers import EsmModel

        try:
            self.esm = EsmModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            if local_files_only:
                raise
            self.esm = EsmModel.from_pretrained(model_name)
        hidden_size = self.esm.config.hidden_size

        # 简单的分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids:    [B, L]
        attention_mask: [B, L]
        Returns: logits [B, L, num_classes]
        """
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # outputs.last_hidden_state: [B, L, hidden_size]
        logits = self.classifier(outputs.last_hidden_state)
        return logits


# ============================================================================
# 3. 训练 & 评估
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="训练", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)

        # 只计算非 padding 位置的 loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.permute(0, 2, 1), labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

        total_loss += loss.item()

        # 收集预测（仅非 ignore 位置）
        preds = torch.argmax(logits, dim=-1)
        valid_mask = (labels != -100)
        all_preds.extend(preds[valid_mask].cpu().numpy())
        all_labels.extend(labels[valid_mask].cpu().numpy())

    q3 = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), q3


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="评估", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.permute(0, 2, 1), labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        valid_mask = (labels != -100)
        all_preds.extend(preds[valid_mask].cpu().numpy())
        all_labels.extend(labels[valid_mask].cpu().numpy())

    q3 = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), q3, all_preds, all_labels


def plot_benchmark(v1_results: dict, esm2_results: dict, save_path: Path):
    """生成 V1 vs ESM-2 对比图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Q3 准确率对比
    models = ["CNN+BiLSTM (V1)", "ESM-2 (Fine-tuned)"]
    q3_scores = [v1_results["q3"] * 100, esm2_results["q3"] * 100]
    colors = ["#8a9a70", "#d4c8a8"]

    bars = axes[0].bar(models, q3_scores, color=colors, edgecolor="white", linewidth=2)
    for bar, score in zip(bars, q3_scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{score:.1f}%", ha="center", fontweight="bold", fontsize=14)
    axes[0].set_ylabel("Q3 Accuracy (%)")
    axes[0].set_title("Overall Q3 Accuracy: V1 vs ESM-2")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.3)

    # 右图：每类 F1 对比
    categories = ["α-Helix (H)", "β-Sheet (E)", "Coil (C)"]
    x = np.arange(len(categories))
    width = 0.35

    axes[1].bar(x - width / 2, v1_results["f1_per_class"], width,
                label="CNN+BiLSTM (V1)", color="#8a9a70", edgecolor="white")
    axes[1].bar(x + width / 2, esm2_results["f1_per_class"], width,
                label="ESM-2 (Fine-tuned)", color="#d4c8a8", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Per-Class F1 Score Comparison")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Benchmark 对比图已保存: {save_path}")


# ============================================================================
# 4. 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("[DNA] ESM-2 微调 — 蛋白质二级结构预测")
    print(f"[Device] {CONFIG['device'].upper()}")
    print(f"[Model] {CONFIG['model_name']}")
    print("=" * 60)

    # 加载 ESM tokenizer
    from transformers import EsmTokenizer
    print("\n[Setup] 加载 ESM-2 tokenizer...")
    tokenizer = EsmTokenizer.from_pretrained(CONFIG["model_name"])

    # ----- 4.1 加载数据 -----
    print("[Data] 加载数据...")
    sequences, structures = load_data()

    # 与 V1 相同的训练/验证/测试集划分
    seq_train, seq_temp, ss_train, ss_temp = train_test_split(
        sequences, structures,
        test_size=CONFIG["val_size"] + CONFIG["test_size"],
        random_state=42,
    )
    val_ratio = CONFIG["val_size"] / (CONFIG["val_size"] + CONFIG["test_size"])
    seq_val, seq_test, ss_val, ss_test = train_test_split(
        seq_temp, ss_temp, test_size=1 - val_ratio, random_state=42,
    )

    print(f"  训练: {len(seq_train)}  验证: {len(seq_val)}  测试: {len(seq_test)}")

    train_dataset = ProteinSSDataset(seq_train, ss_train, CONFIG["max_seq_len"], tokenizer)
    val_dataset = ProteinSSDataset(seq_val, ss_val, CONFIG["max_seq_len"], tokenizer)
    test_dataset = ProteinSSDataset(seq_test, ss_test, CONFIG["max_seq_len"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # ----- 4.2 初始化 ESM-2 模型 -----
    print(f"\n[Model] 加载 ESM-2 模型...")
    model = ESM2ForSecondaryStructure(
        CONFIG["model_name"],
        num_classes=3,
        dropout=CONFIG["dropout"],
    ).to(CONFIG["device"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {total_params:,} (可训练: {trainable_params:,})")

    # 冻结 ESM 主干的前几层（可选优化）
    # for param in model.esm.embeddings.parameters():
    #     param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                   weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # ----- 4.3 训练 -----
    print(f"\n[Train] 开始微调 (最高 {CONFIG['epochs']} epochs)...")
    history = {"train_loss": [], "val_loss": [], "train_q3": [], "val_q3": []}
    best_val_q3 = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_q3 = train_epoch(model, train_loader, optimizer, CONFIG["device"])
        val_loss, val_q3, _, _ = evaluate(model, val_loader, CONFIG["device"])

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_q3"].append(train_q3)
        history["val_q3"].append(val_q3)

        marker = ""
        if val_q3 > best_val_q3:
            best_val_q3 = val_q3
            best_epoch = epoch
            patience_counter = 0
            marker = " [Best]"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_q3": val_q3,
                "config": CONFIG,
            }, CONFIG["model_dir"] / "best_model_esm2.pt")
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
              f"Loss: {train_loss:.4f} > {val_loss:.4f} | "
              f"Q3: {train_q3:.3f} > {val_q3:.3f}{marker}")

        if patience_counter >= CONFIG["early_stop_patience"]:
            print(f"\n[Early Stop] 验证 Q3 {CONFIG['early_stop_patience']} 轮未提升")
            break

    # ----- 4.4 测试评估 -----
    print(f"\n[Eval] 在测试集上评估最佳 ESM-2 模型...")
    checkpoint = torch.load(CONFIG["model_dir"] / "best_model_esm2.pt",
                            weights_only=False, map_location=CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_q3, all_preds, all_labels = evaluate(model, test_loader, CONFIG["device"])

    print(f"\n{'='*50}")
    print(f"  ESM-2 测试 Q3: {test_q3:.4f} ({test_q3*100:.1f}%)")
    print(f"  最佳验证 Q3: {best_val_q3:.4f} (Epoch {best_epoch})")
    print(f"{'='*50}")

    print("\n[Report] ESM-2 每类指标:")
    target_names = ["H (alpha-helix)", "E (beta-sheet)", "C (coil)"]
    report = classification_report(all_labels, all_preds, target_names=target_names,
                                    output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # ----- 4.5 加载 V1 结果做 Benchmark -----
    print("\n" + "=" * 60)
    print("[Benchmark] V1 vs ESM-2 对比分析")
    print("=" * 60)

    # V1 的真实测试结果（由 evaluate.py 在同一测试集上实测得到）
    # 运行 evaluate.py 可重新生成这些数值
    v1_q3 = 0.8545
    v1_f1 = [0.8866, 0.7800, 0.8539]  # H, E, C

    esm2_q3 = test_q3
    esm2_f1 = [report[n]["f1-score"] for n in target_names]

    print(f"\n{'指标':<25} {'V1 (CNN+BiLSTM)':<20} {'ESM-2 (Fine-tuned)':<20} {'提升':<10}")
    print("-" * 75)
    print(f"{'Q3 Accuracy':<25} {v1_q3*100:<20.1f}% {esm2_q3*100:<20.1f}% {'+' if esm2_q3 > v1_q3 else ''}{abs(esm2_q3 - v1_q3)*100:.1f}%")
    for i, name in enumerate(target_names):
        diff = esm2_f1[i] - v1_f1[i]
        print(f"{name + ' F1':<25} {v1_f1[i]:<20.3f} {esm2_f1[i]:<20.3f} {'+' if diff > 0 else ''}{diff:.3f}")

    # 保存对比图
    plot_benchmark(
        {"q3": v1_q3, "f1_per_class": v1_f1},
        {"q3": esm2_q3, "f1_per_class": esm2_f1},
        CONFIG["model_dir"] / "benchmark_v1_vs_esm2.png",
    )

    print(f"\n[Done] ESM-2 模型: {CONFIG['model_dir'] / 'best_model_esm2.pt'}")
    print(f"[Done] Benchmark 图: {CONFIG['model_dir'] / 'benchmark_v1_vs_esm2.png'}")
    return model, history


if __name__ == "__main__":
    model, history = main()
