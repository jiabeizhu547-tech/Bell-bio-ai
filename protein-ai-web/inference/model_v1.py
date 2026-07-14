"""
=======================================================================
V1 模型定义 + 生物规则后处理 — 供集成推理使用
=======================================================================

包含：
  1. ProteinSSPredictor — CNN+BiLSTM 混合架构
  2. AA_TO_IDX — 氨基酸编码表
  3. encode_sequence — 序列 → 整数编码
  4. apply_biological_smoothing — 生物学规则后处理

作者: 朱家贝 | 日期: 2026-07-12
=======================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 氨基酸编码（与 train.py 完全一致）
# ============================================================================

AA_TO_IDX = {
    "A": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,
    "G": 6,  "H": 7,  "I": 8,  "K": 9,  "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
}

SS_LABEL_TO_IDX = {"H": 0, "E": 1, "C": 2}

MAX_SEQ_LEN = 256


def encode_sequence(seq: str) -> np.ndarray:
    """将氨基酸序列编码为整数数组，补齐到 MAX_SEQ_LEN。"""
    encoded = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    for i, aa in enumerate(seq[:MAX_SEQ_LEN]):
        encoded[i] = AA_TO_IDX.get(aa.upper(), 21)  # 21 = unknown
    return encoded


# ============================================================================
# V1 模型: CNN + BiLSTM
# ============================================================================

class ProteinSSPredictor(nn.Module):
    """
    蛋白质二级结构预测模型（CNN + BiLSTM）。

    架构：
      Embedding → 1D CNN → BiLSTM → Dropout → Linear → Q3 logits
    """

    def __init__(self, config: dict):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=22,
            embedding_dim=config["embed_dim"],
            padding_idx=0,
        )

        self.conv1 = nn.Conv1d(
            in_channels=config["embed_dim"],
            out_channels=config["cnn_channels"],
            kernel_size=config["cnn_kernel"],
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(config["cnn_channels"])

        self.lstm = nn.LSTM(
            input_size=config["cnn_channels"],
            hidden_size=config["lstm_hidden"],
            num_layers=config["lstm_layers"],
            bidirectional=True,
            batch_first=True,
            dropout=config["dropout"] if config["lstm_layers"] > 1 else 0,
        )

        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(config["lstm_hidden"] * 2, config["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] — 氨基酸索引
        Returns:
            logits: [batch_size, seq_len, num_classes]
        """
        x = self.embedding(x)                     # [B, L, E]
        x = x.permute(0, 2, 1)                    # [B, E, L]
        x = F.relu(self.bn1(self.conv1(x)))       # [B, C, L]
        x = x.permute(0, 2, 1)                    # [B, L, C]
        x, _ = self.lstm(x)                       # [B, L, H*2]
        x = self.dropout(x)
        logits = self.classifier(x)               # [B, L, 3]
        return logits


# ============================================================================
# 生物学规则后处理
# ============================================================================

def apply_biological_smoothing(preds: np.ndarray, probs: np.ndarray,
                                min_run: int = 3) -> np.ndarray:
    """
    生物学规则后处理 — 修正不符合蛋白质结构常识的预测。

    规则：
      - 螺旋(H)和折叠(E)通常连续至少 min_run 个残基
      - 短片段（< min_run）且置信度 < 0.6 → 翻转为 C
      - 孤立残基（单个 H/E 两侧都是 C）→ 翻转为 C
    """
    smoothed = preds.copy()
    n = len(preds)

    i = 0
    while i < n:
        current = smoothed[i]

        # 卷曲 C 保持不动
        if current == SS_LABEL_TO_IDX["C"]:
            i += 1
            continue

        # 找当前结构类型的连续片段
        run_start = i
        while i < n and smoothed[i] == current:
            i += 1
        run_end = i
        run_len = run_end - run_start

        # 片段长度达标 → 保留
        if run_len >= min_run:
            continue

        # 片段太短，检查平均置信度
        run_conf = np.mean(probs[run_start:run_end, current])

        # 置信度低 → 翻转为 C
        if run_conf < 0.6:
            smoothed[run_start:run_end] = SS_LABEL_TO_IDX["C"]
            continue

        # 单个残基，两侧都是 C → 翻转为 C
        if run_len == 1:
            left = smoothed[run_start - 1] if run_start > 0 else SS_LABEL_TO_IDX["C"]
            right = smoothed[run_end] if run_end < n else SS_LABEL_TO_IDX["C"]
            if left == SS_LABEL_TO_IDX["C"] and right == SS_LABEL_TO_IDX["C"]:
                smoothed[run_start] = SS_LABEL_TO_IDX["C"]

    return smoothed
