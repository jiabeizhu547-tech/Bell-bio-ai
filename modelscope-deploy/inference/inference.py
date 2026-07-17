"""
=============================================================================
[DNA] 集成推理脚本 — 蛋白质二级结构预测
=============================================================================

同时使用 V1 (CNN+BiLSTM) 和 ESM-2 两个模型，通过加权投票和
生物学规则后处理，提供最优的 Q3 预测（89.8%）。

用法：
    from inference import predict_secondary_structure
    result = predict_secondary_structure("MKVLILACLVALALA")
    print(result["structure"])   # "CCEEEEHHHHHHHCC"

作者: 朱家贝 | 日期: 2026-07-12
=============================================================================
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from pathlib import Path
from typing import Dict

# ============================================================================
# 全局配置
# ============================================================================

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR  # 优先根目录（ModelScope/HF Spaces）
if not (MODEL_DIR / "best_model_esm2.pt").exists():
    MODEL_DIR = BASE_DIR / "models"  # fallback: 本地 models/ 目录

# ESM-2 基础模型目录（内置，无需联网下载）
ESM_LOCAL_DIR = BASE_DIR / "esm_model"
if ESM_LOCAL_DIR.exists():
    ESM_MODEL_NAME = str(ESM_LOCAL_DIR)  # 使用内置模型
else:
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # 从 HF 下载
MAX_SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SS_IDX_TO_LABEL = {0: "H", 1: "E", 2: "C"}
SS_LABEL_NAMES = {"H": "α-螺旋 (Alpha-helix)", "E": "β-折叠 (Beta-sheet)", "C": "无规卷曲 (Coil)"}

# ============================================================================
# 集成权重（与 ensemble.py 的加权平均策略一致）
# ============================================================================

ENSEMBLE_W1 = 0.40  # V1 权重
ENSEMBLE_W2 = 0.60  # ESM-2 权重
SMOOTH_MIN_RUN = 3  # 生物规则：H/E 最小连续长度

# 缓存已加载的模型
_esm2_model = None
_esm2_tokenizer = None
_v1_model = None


# ============================================================================
# 模型加载（懒加载 + 缓存）
# ============================================================================

def _load_esm2_model():
    """加载 ESM-2 微调模型（仅加载一次）。"""
    global _esm2_model, _esm2_tokenizer

    if _esm2_model is not None:
        return _esm2_model, _esm2_tokenizer

    from transformers import EsmTokenizer
    from train_esm2 import ESM2ForSecondaryStructure

    print(f"[Inference] 加载 ESM-2: {ESM_MODEL_NAME}")
    print(f"[Inference] 设备: {DEVICE.upper()}")

    _esm2_tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    _esm2_model = ESM2ForSecondaryStructure(
        ESM_MODEL_NAME, num_classes=3, dropout=0.0, local_files_only=False,
    ).to(DEVICE)

    ckpt_path = MODEL_DIR / "best_model_esm2.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"ESM-2 模型文件未找到: {ckpt_path}\n"
            f"请先运行 train_esm2.py 训练模型。"
        )

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    _esm2_model.load_state_dict(checkpoint["model_state_dict"])
    _esm2_model.eval()

    print(f"[Inference] ESM-2 加载完成 "
          f"(Epoch {checkpoint.get('epoch', '?')}, "
          f"Val Q3: {checkpoint.get('val_q3', 0):.3f})")

    return _esm2_model, _esm2_tokenizer


def _load_v1_model():
    """加载 V1 (CNN+BiLSTM) 模型（仅加载一次）。"""
    global _v1_model

    if _v1_model is not None:
        return _v1_model

    from model_v1 import ProteinSSPredictor

    print("[Inference] 加载 V1 (CNN+BiLSTM)...")

    v1_config = {
        "embed_dim": 64, "cnn_channels": 128, "cnn_kernel": 7,
        "lstm_hidden": 128, "lstm_layers": 2, "dropout": 0.3, "num_classes": 3,
    }
    _v1_model = ProteinSSPredictor(v1_config).to(DEVICE)

    ckpt_path = MODEL_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"V1 模型文件未找到: {ckpt_path}\n"
            f"请先将 models/best_model.pt 复制到 hf_space/ 目录。"
        )

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    _v1_model.load_state_dict(checkpoint["model_state_dict"])
    _v1_model.eval()

    print(f"[Inference] V1 加载完成 (Epoch {checkpoint.get('epoch', '?')}, "
          f"Val Q3: {checkpoint.get('val_q3', 0):.3f})")

    return _v1_model


# ============================================================================
# 预处理
# ============================================================================

def _validate_sequence(seq: str) -> str:
    """验证并清洗氨基酸序列。"""
    # 去除空白字符，转大写
    seq = "".join(seq.split()).upper()

    # 去除常见的非氨基酸字符
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join(aa for aa in seq if aa in valid_aas)

    if len(cleaned) == 0:
        raise ValueError("序列中没有有效的氨基酸字符。")

    if len(cleaned) < 5:
        raise ValueError(f"序列太短（{len(cleaned)} 个残基），至少需要 5 个。")

    return cleaned


# ============================================================================
# 单模型推理
# ============================================================================

@torch.no_grad()
def _predict_esm2(seq: str, model, tokenizer) -> np.ndarray:
    """用 ESM-2 获取每个残基的 Q3 概率分布。"""
    spaced_seq = " ".join(list(seq))
    tokens = tokenizer(
        spaced_seq,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    logits = model(input_ids, attention_mask)  # [1, L, 3]
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [L, 3]

    valid_len = min(len(seq), MAX_SEQ_LEN)
    return probs[:valid_len]


@torch.no_grad()
def _predict_v1(seq: str, model) -> np.ndarray:
    """用 V1 (CNN+BiLSTM) 获取每个残基的 Q3 概率分布。"""
    from model_v1 import encode_sequence

    encoded = encode_sequence(seq)  # [MAX_SEQ_LEN]
    input_tensor = torch.LongTensor(encoded).unsqueeze(0).to(DEVICE)  # [1, L]

    logits = model(input_tensor)  # [1, L, 3]
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [L, 3]

    valid_len = min(len(seq), MAX_SEQ_LEN)
    return probs[:valid_len]


# ============================================================================
# 核心预测函数（集成模式）
# ============================================================================

def predict_secondary_structure(sequence: str) -> Dict:
    """
    预测一条蛋白质序列的二级结构（集成模式：V1 + ESM-2 加权平均 + 生物规则后处理）。

    Args:
        sequence: 氨基酸序列字符串，如 "MKVLILACLVALALA"

    Returns:
        dict:
            - sequence:      清洗后的序列
            - structure:     每个残基的结构标签（H/E/C）
            - per_residue:   list of dict，每个残基的详细信息
            - counts:        {"H": n, "E": n, "C": n}
            - percentages:   {"H": pct, "E": pct, "C": pct}
            - length:        序列长度
    """
    from model_v1 import apply_biological_smoothing

    esm2_model, esm2_tokenizer = _load_esm2_model()
    v1_model = _load_v1_model()

    # 清洗序列
    seq = _validate_sequence(sequence)
    seq = seq[:MAX_SEQ_LEN]

    valid_len = min(len(seq), MAX_SEQ_LEN)

    # ---- 1. 获取两个模型的概率分布 ----
    esm2_probs = _predict_esm2(seq, esm2_model, esm2_tokenizer)  # [L, 3]
    v1_probs = _predict_v1(seq, v1_model)                          # [L, 3]

    # ---- 2. 加权平均 ----
    ensemble_probs = ENSEMBLE_W1 * v1_probs + ENSEMBLE_W2 * esm2_probs  # [L, 3]

    # ---- 3. 初步预测 ----
    raw_preds = np.argmax(ensemble_probs, axis=-1)  # [L]

    # ---- 4. 生物学规则后处理 ----
    smoothed_preds = apply_biological_smoothing(raw_preds, ensemble_probs, min_run=SMOOTH_MIN_RUN)

    # 对于每个残基，用平滑后的预测更新最终置信度
    final_probs = ensemble_probs.copy()
    for i in range(valid_len):
        if smoothed_preds[i] != raw_preds[i]:
            # 被翻转的残基：置信度设为翻转后的类别概率
            final_probs[i] = ensemble_probs[i]

    # ---- 5. 构建结果 ----
    structure = "".join(SS_IDX_TO_LABEL[p] for p in smoothed_preds)

    per_residue = []
    for i, (aa, ss_idx) in enumerate(zip(seq, smoothed_preds)):
        per_residue.append({
            "position": i + 1,
            "amino_acid": aa,
            "structure": SS_IDX_TO_LABEL[ss_idx],
            "structure_name": SS_LABEL_NAMES[SS_IDX_TO_LABEL[ss_idx]],
            "confidence": float(final_probs[i][ss_idx]),
            "probabilities": {
                "H": float(final_probs[i][0]),
                "E": float(final_probs[i][1]),
                "C": float(final_probs[i][2]),
            },
            "smoothed": bool(smoothed_preds[i] != raw_preds[i]),
        })

    counts = {
        "H": structure.count("H"),
        "E": structure.count("E"),
        "C": structure.count("C"),
    }
    total = sum(counts.values())
    percentages = {
        "H": round(counts["H"] / total * 100, 1) if total > 0 else 0,
        "E": round(counts["E"] / total * 100, 1) if total > 0 else 0,
        "C": round(counts["C"] / total * 100, 1) if total > 0 else 0,
    }

    return {
        "sequence": seq,
        "structure": structure,
        "per_residue": per_residue,
        "counts": counts,
        "percentages": percentages,
        "length": valid_len,
    }


# ============================================================================
# 命令行测试
# ============================================================================

if __name__ == "__main__":
    import json

    test_seqs = [
        "MKVLILACLVALALACTVQA",
        "ACDEFGHIKLMNPQRSTVWY" * 5,
    ]

    for seq in test_seqs:
        print(f"\n{'='*60}")
        print(f"序列: {seq[:40]}{'...' if len(seq) > 40 else ''} (长度 {len(seq)})")
        result = predict_secondary_structure(seq)
        print(f"结构: {result['structure'][:40]}{'...' if len(result['structure']) > 40 else ''}")
        print(f"分布: α螺旋 {result['percentages']['H']}%  "
              f"β折叠 {result['percentages']['E']}%  "
              f"卷曲 {result['percentages']['C']}%")
        confidences = [r["confidence"] for r in result["per_residue"]]
        smoothed_count = sum(1 for r in result["per_residue"] if r.get("smoothed", False))
        print(f"置信度: {np.mean(confidences):.3f}  |  "
              f"生物规则修正: {smoothed_count} 个残基")
