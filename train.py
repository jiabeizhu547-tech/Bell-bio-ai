"""
=============================================================================
[DNA] 蛋白质二级结构预测 — PyTorch 完整训练脚本
=============================================================================

本脚本实现了从零开始的蛋白质二级结构（Q3）预测模型。

任务说明：
  给定一条氨基酸序列，预测每个残基的二级结构类型：
    H = α-螺旋 (Alpha-helix)     — 占蛋白质内约 35%
    E = β-折叠 (Beta-sheet)      — 占蛋白质内约 20%
    C = 无规卷曲 (Coil/Loop)     — 占蛋白质内约 45%

模型架构：
  CNN + BiLSTM 混合模型（适合蛋白质序列数据）

评估指标：
  Q3 accuracy = 正确预测的残基数 / 总残基数

作者：朱家贝 | 日期：2026-07
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 0. 全局配置
# ============================================================================

CONFIG = {
    # 数据
    "data_dir": Path(__file__).parent / "data",
    "max_seq_len": 256,         # 最大序列长度（截断/补齐）
    "test_size": 0.15,          # 测试集比例
    "val_size": 0.10,           # 验证集比例

    # 模型
    "embed_dim": 64,            # 氨基酸 embedding 维度
    "cnn_channels": 128,        # CNN 通道数
    "cnn_kernel": 7,            # CNN 卷积核大小
    "lstm_hidden": 128,         # LSTM 隐藏层大小
    "lstm_layers": 2,           # LSTM 层数
    "dropout": 0.3,             # Dropout 比例
    "num_classes": 3,           # Q3 分类：H, E, C

    # 训练
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 50,
    "early_stop_patience": 10,  # 早停等待轮数
    "weight_decay": 1e-4,

    # 输出
    "model_dir": Path(__file__).parent / "models",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. 数据处理
# ============================================================================

# 20 种标准氨基酸 → 整数编码（加上 0=padding, 21=unknown）
AA_TO_IDX = {
    "A": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,
    "G": 6,  "H": 7,  "I": 8,  "K": 9,  "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
}
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}

# Q3 结构标签 → 整数编码
SS_TO_IDX = {"H": 0, "E": 1, "C": 2}   # α-螺旋, β-折叠, 无规卷曲
IDX_TO_SS = {v: k for k, v in SS_TO_IDX.items()}


def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """将氨基酸序列编码为整数数组，截断或补齐到 max_len。"""
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, aa in enumerate(seq[:max_len]):
        encoded[i] = AA_TO_IDX.get(aa.upper(), 21)  # 21 = unknown
    return encoded


def encode_structure(ss: str, max_len: int) -> np.ndarray:
    """将二级结构字符串编码为整数标签。"""
    encoded = np.full(max_len, -1, dtype=np.int64)  # -1 = 忽略（padding）
    for i, s in enumerate(ss[:max_len]):
        s_upper = s.upper()
        if s_upper in SS_TO_IDX:
            encoded[i] = SS_TO_IDX[s_upper]
        # DSSP 8 类 → 映射到 3 类（Q3）
        elif s_upper == "G":   # 3₁₀-螺旋 → α-螺旋
            encoded[i] = SS_TO_IDX["H"]
        elif s_upper == "I":   # π-螺旋 → α-螺旋
            encoded[i] = SS_TO_IDX["H"]
        elif s_upper == "B":   # 孤立 β-桥 → β-折叠
            encoded[i] = SS_TO_IDX["E"]
        elif s_upper == "T":   # 转角 → 卷曲
            encoded[i] = SS_TO_IDX["C"]
        elif s_upper == "S":   # 弯曲 → 卷曲
            encoded[i] = SS_TO_IDX["C"]
    return encoded


class ProteinSSDataset(Dataset):
    """
    蛋白质二级结构数据集。

    输入格式（FASTA 风格）：
      >seq_id
      ACDEFGHIKLM...
    对应的二级结构格式：
      >seq_id
      HHHHCCCEEE...

    或从 CSV 文件加载（列: sequence, structure）。
    """

    def __init__(self, sequences: list[str], structures: list[str], max_len: int):
        self.max_len = max_len
        self.data = []

        for seq, ss in zip(sequences, structures):
            x = encode_sequence(seq, max_len)
            y = encode_structure(ss, max_len)
            # 掩码：标记非 padding 位置
            mask = (y != -1)
            self.data.append((x, y, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, mask = self.data[idx]
        return (
            torch.LongTensor(x),       # [max_len]
            torch.LongTensor(y),       # [max_len]
            torch.BoolTensor(mask),    # [max_len]
        )


def generate_synthetic_data(n_samples: int = 1000) -> tuple:
    """
    生成合成蛋白质数据用于测试训练流程。

    模拟真实分布：
    - 序列长度服从 gamma 分布（大部分 50-300）
    - 二级结构有生物学上的规律（α-螺旋倾向 Ala/Leu, β-折叠倾向 Val/Ile）
    - 不是完全随机，所以模型可以学到一些信号
    """
    amino_acids = list(AA_TO_IDX.keys())
    ss_classes = list(SS_TO_IDX.keys())

    # 每种氨基酸倾向的二级结构（简化版 Chou-Fasman 参数）
    aa_helix_prob = {  # P(α-螺旋 | 氨基酸)
        "A": 0.42, "L": 0.41, "M": 0.30, "E": 0.35, "K": 0.32,
        "Q": 0.25, "R": 0.25, "H": 0.24, "F": 0.22, "W": 0.21,
        "I": 0.20, "V": 0.20, "D": 0.19, "Y": 0.18, "C": 0.16,
        "N": 0.12, "S": 0.12, "T": 0.12, "G": 0.08, "P": 0.10,
    }
    aa_sheet_prob = {  # P(β-折叠 | 氨基酸)
        "V": 0.45, "I": 0.40, "Y": 0.38, "F": 0.36, "W": 0.35,
        "T": 0.30, "L": 0.28, "C": 0.28, "M": 0.25, "Q": 0.22,
        "R": 0.22, "N": 0.20, "H": 0.20, "A": 0.18, "S": 0.18,
        "G": 0.14, "K": 0.15, "D": 0.12, "E": 0.12, "P": 0.08,
    }

    sequences, structures = [], []

    for i in range(n_samples):
        length = int(np.random.gamma(shape=5, scale=30))
        length = max(30, min(length, 500))

        seq_chars = []
        ss_chars = []

        # 生成片段（3-8 个残基一组），模仿真实二级结构片段
        pos = 0
        while pos < length:
            fragment_len = random.randint(3, 8)
            # 随机选择片段类型
            ss_type = random.choices(
                ["H", "E", "C"],
                weights=[0.35, 0.20, 0.45]
            )[0]

            for _ in range(min(fragment_len, length - pos)):
                if ss_type == "H":
                    aa = random.choices(
                        amino_acids,
                        weights=[aa_helix_prob.get(a, 0.2) for a in amino_acids]
                    )[0]
                elif ss_type == "E":
                    aa = random.choices(
                        amino_acids,
                        weights=[aa_sheet_prob.get(a, 0.2) for a in amino_acids]
                    )[0]
                else:
                    aa = random.choice(amino_acids)

                seq_chars.append(aa)
                ss_chars.append(ss_type)
                pos += 1

        sequences.append("".join(seq_chars))
        structures.append("".join(ss_chars))

    return sequences, structures


def load_or_generate_data():
    """
    加载真实数据或生成合成数据。

    尝试从 data_dir 读取；如果没有，生成合成数据用于测试流程。
    """
    seq_file = CONFIG["data_dir"] / "real_sequences.fasta"
    ss_file = CONFIG["data_dir"] / "real_structures.fasta"

    if seq_file.exists() and ss_file.exists():
        print(f"[Data] 加载本地数据：{seq_file}")
        sequences = []
        structures = []
        with open(seq_file) as f:
            content = f.read().strip().split("\n>")
            for block in content:
                lines = block.strip().split("\n")
                if len(lines) >= 2:
                    sequences.append("".join(lines[1:]).replace(">", ""))

        with open(ss_file) as f:
            content = f.read().strip().split("\n>")
            for block in content:
                lines = block.strip().split("\n")
                if len(lines) >= 2:
                    structures.append("".join(lines[1:]).replace(">", ""))
    else:
        print("🔧 未找到本地数据，生成合成数据用于测试流程...")
        sequences, structures = generate_synthetic_data(n_samples=800)

    print(f"  [Eval] 共 {len(sequences)} 条蛋白质序列")
    return sequences, structures


# ============================================================================
# 2. 模型定义
# ============================================================================

class ProteinSSPredictor(nn.Module):
    """
    蛋白质二级结构预测模型。

    架构：
      Amino Acid Embedding
        ↓
      1D CNN（提取局部 motif，如 3-7 残基的螺旋/折叠倾向）
        ↓
      BiLSTM（捕获长程依赖，如 β-折叠配对残基可能相距很远）
        ↓
      Linear → Q3 logits（每个残基独立预测）

    设计理由：
      - CNN 捕获局部序列模式（短程的二级结构倾向）
      - BiLSTM 捕获长程相互作用（β-折叠的两条链可能相距几十个残基）
      - 这种混合架构在蛋白质序列任务中表现优异
    """

    def __init__(self, config: dict):
        super().__init__()

        # 氨基酸 embedding（22 种：20 AA + padding + unknown）
        self.embedding = nn.Embedding(
            num_embeddings=22,
            embedding_dim=config["embed_dim"],
            padding_idx=0,
        )

        # 1D 卷积层
        self.conv1 = nn.Conv1d(
            in_channels=config["embed_dim"],
            out_channels=config["cnn_channels"],
            kernel_size=config["cnn_kernel"],
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(config["cnn_channels"])

        # BiLSTM 层
        self.lstm = nn.LSTM(
            input_size=config["cnn_channels"],
            hidden_size=config["lstm_hidden"],
            num_layers=config["lstm_layers"],
            bidirectional=True,
            batch_first=True,
            dropout=config["dropout"] if config["lstm_layers"] > 1 else 0,
        )

        # 输出层：BiLSTM hidden*2 → Q3 logits
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(config["lstm_hidden"] * 2, config["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] — 氨基酸索引
        Returns:
            logits: [batch_size, seq_len, num_classes]
        """
        # Embedding: [B, L] → [B, L, embed_dim]
        x = self.embedding(x)

        # CNN: [B, L, E] → [B, E, L] → [B, C, L] → [B, C, L] → [B, L, C]
        x = x.permute(0, 2, 1)          # [B, embed_dim, L]
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)          # [B, L, C]

        # BiLSTM: [B, L, C] → [B, L, hidden*2]
        x, _ = self.lstm(x)

        # 分类: [B, L, hidden*2] → [B, L, num_classes]
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


# ============================================================================
# 3. 训练 & 评估
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_x, batch_y, batch_mask in tqdm(dataloader, desc="训练", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)

        # 前向传播
        logits = model(batch_x)  # [B, L, num_classes]

        # 计算损失（只对非 padding 位置）
        logits_masked = logits[batch_mask]      # [N_valid, 3]
        labels_masked = batch_y[batch_mask]     # [N_valid]
        loss = criterion(logits_masked, labels_masked)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 记录预测（用于计算 Q3）
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds[batch_mask].cpu().numpy())
        all_labels.extend(labels_masked.cpu().numpy())

    q3 = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, q3


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """在验证集或测试集上评估模型。"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_x, batch_y, batch_mask in tqdm(dataloader, desc="评估", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_mask = batch_mask.to(device)

        logits = model(batch_x)

        logits_masked = logits[batch_mask]
        labels_masked = batch_y[batch_mask]
        loss = criterion(logits_masked, labels_masked)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds[batch_mask].cpu().numpy())
        all_labels.extend(labels_masked.cpu().numpy())

    q3 = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    return avg_loss, q3, all_preds, all_labels


def plot_training_curve(history: dict, save_path: Path):
    """绘制训练曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    axes[0].plot(history["train_loss"], label="训练损失", color="#178a8c", linewidth=2)
    axes[0].plot(history["val_loss"], label="验证损失", color="#d4922a", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("训练 & 验证损失")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Q3 曲线
    axes[1].plot(history["train_q3"], label="训练 Q3", color="#178a8c", linewidth=2)
    axes[1].plot(history["val_q3"], label="验证 Q3", color="#d4922a", linewidth=2)
    axes[1].axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Baseline (70%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Q3 Accuracy")
    axes[1].set_title("训练 & 验证 Q3 准确率")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] 训练曲线已保存：{save_path}")


# ============================================================================
# 4. 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("[DNA] 蛋白质二级结构预测 — PyTorch 训练脚本")
    print(f"[Device] 设备：{CONFIG['device'].upper()}")
    print("=" * 60)

    # ----- 4.1 加载数据 -----
    print("\n[Data] 准备数据...")
    sequences, structures = load_or_generate_data()

    # 划分训练/验证/测试集
    seq_train, seq_temp, ss_train, ss_temp = train_test_split(
        sequences, structures, test_size=CONFIG["val_size"] + CONFIG["test_size"],
        random_state=42
    )
    val_ratio = CONFIG["val_size"] / (CONFIG["val_size"] + CONFIG["test_size"])
    seq_val, seq_test, ss_val, ss_test = train_test_split(
        seq_temp, ss_temp, test_size=1 - val_ratio, random_state=42
    )

    print(f"  训练集：{len(seq_train)} 条")
    print(f"  验证集：{len(seq_val)} 条")
    print(f"  测试集：{len(seq_test)} 条")

    # 创建 Dataset 和 DataLoader
    train_dataset = ProteinSSDataset(seq_train, ss_train, CONFIG["max_seq_len"])
    val_dataset = ProteinSSDataset(seq_val, ss_val, CONFIG["max_seq_len"])
    test_dataset = ProteinSSDataset(seq_test, ss_test, CONFIG["max_seq_len"])

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0,
    )

    # ----- 4.2 初始化模型 -----
    print(f"\n[Model] 初始化模型...")
    model = ProteinSSPredictor(CONFIG).to(CONFIG["device"])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量：{total_params:,}（可训练：{trainable_params:,}）")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.CrossEntropyLoss()

    # ----- 4.3 训练循环 -----
    print(f"\n[Train] 开始训练（{CONFIG['epochs']} epochs）...")
    history = {"train_loss": [], "val_loss": [], "train_q3": [], "val_q3": []}
    best_val_q3 = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        # 训练
        train_loss, train_q3 = train_epoch(
            model, train_loader, optimizer, criterion, CONFIG["device"]
        )

        # 验证
        val_loss, val_q3, _, _ = evaluate(
            model, val_loader, criterion, CONFIG["device"]
        )

        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # 记录
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_q3"].append(train_q3)
        history["val_q3"].append(val_q3)

        # 打印
        marker = ""
        if val_q3 > best_val_q3:
            best_val_q3 = val_q3
            best_epoch = epoch
            patience_counter = 0
            marker = " [Best] 最佳模型已保存"

            # 保存最佳模型
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_q3": val_q3,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
            }, CONFIG["model_dir"] / "best_model.pt")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
            f"Loss: {train_loss:.4f} → {val_loss:.4f} | "
            f"Q3: {train_q3:.3f} → {val_q3:.3f} | "
            f"LR: {current_lr:.2e}{marker}"
        )

        # 早停
        if patience_counter >= CONFIG["early_stop_patience"]:
            print(f"\n[Stop] 早停触发：验证 Q3 在 {CONFIG['early_stop_patience']} 轮内未提升")
            break

    # ----- 4.4 测试评估 -----
    print(f"\n[Eval] 在测试集上评估最佳模型...")
    checkpoint = torch.load(CONFIG["model_dir"] / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_q3, all_preds, all_labels = evaluate(
        model, test_loader, criterion, CONFIG["device"]
    )
    print(f"\n{'='*40}")
    print(f"  [Result] 测试集 Q3 准确率：{test_q3:.4f} ({test_q3*100:.1f}%)")
    print(f"  [Best] 最佳验证 Q3：{best_val_q3:.4f}（Epoch {best_epoch}）")
    print(f"{'='*40}")

    # 详细分类报告
    print("\n[Report] 每类指标：")
    target_names = ["H (α-螺旋)", "E (β-折叠)", "C (卷曲)"]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # ----- 4.5 可视化 -----
    plot_training_curve(history, CONFIG["model_dir"] / "training_curve.png")

    # ----- 4.6 演示预测 -----
    print("\n[Demo] 演示预测：")
    demo_seq = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 残基的示例序列
    demo_x = encode_sequence(demo_seq, CONFIG["max_seq_len"])
    demo_x = torch.LongTensor(demo_x).unsqueeze(0).to(CONFIG["device"])  # [1, L]

    model.eval()
    with torch.no_grad():
        logits = model(demo_x)
        preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

    # 只显示前 60 个残基
    demo_len = min(60, len(demo_seq))
    print(f"  序列:  {demo_seq[:demo_len]}")
    print(f"  预测:  {''.join(IDX_TO_SS.get(p, '?') for p in preds[:demo_len])}")
    print(f"  图例:  H=α-螺旋  E=β-折叠  C=无规卷曲")

    print(f"\n[Done] 训练完成！模型已保存至 {CONFIG['model_dir'] / 'best_model.pt'}")
    print(f"[Plot] 训练曲线已保存至 {CONFIG['model_dir'] / 'training_curve.png'}")

    return model, history


if __name__ == "__main__":
    model, history = main()
