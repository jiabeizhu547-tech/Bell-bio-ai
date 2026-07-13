"""
=============================================================================
[DNA] ESM-2 推理脚本 — 蛋白质二级结构预测
=============================================================================

加载训练好的 ESM-2 模型，提供单个蛋白质序列的二级结构预测接口。
供 Phase 3 Gradio Web 工具调用。

用法：
    from inference import predict_secondary_structure
    result = predict_secondary_structure("MKVLILACLVALALA")
    print(result["structure"])   # "CCEEEEHHHHHHHCC"

作者: Bell | 日期: 2026-07-12
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

MODEL_DIR = Path(__file__).parent / "models"
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_SEQ_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SS_IDX_TO_LABEL = {0: "H", 1: "E", 2: "C"}
SS_LABEL_NAMES = {"H": "α-螺旋 (Alpha-helix)", "E": "β-折叠 (Beta-sheet)", "C": "无规卷曲 (Coil)"}

# 缓存已加载的模型和 tokenizer，避免重复加载
_model = None
_tokenizer = None


# ============================================================================
# 模型加载（懒加载 + 缓存）
# ============================================================================

def _load_model():
    """加载 ESM-2 微调模型（仅加载一次）。"""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    from transformers import EsmTokenizer
    from train_esm2 import ESM2ForSecondaryStructure

    print(f"[Inference] 加载模型: {ESM_MODEL_NAME}")
    print(f"[Inference] 设备: {DEVICE.upper()}")

    # 使用本地缓存，避免网络超时
    _tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME, local_files_only=True)
    _model = ESM2ForSecondaryStructure(
        ESM_MODEL_NAME, num_classes=3, dropout=0.0, local_files_only=True,
    ).to(DEVICE)

    ckpt_path = MODEL_DIR / "best_model_esm2.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"模型文件未找到: {ckpt_path}\n"
            f"请先运行 train_esm2.py 训练模型。"
        )

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    print(f"[Inference] 模型加载完成 "
          f"(Epoch {checkpoint.get('epoch', '?')}, "
          f"Val Q3: {checkpoint.get('val_q3', 0):.3f})")

    return _model, _tokenizer


# ============================================================================
# 预处理
# ============================================================================

def _validate_sequence(seq: str) -> str:
    """验证并清洗氨基酸序列。"""
    # 去除空白字符，转大写
    seq = "".join(seq.split()).upper()

    # 去除常见的非氨基酸字符（如数字、特殊符号）
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = "".join(aa for aa in seq if aa in valid_aas)

    if len(cleaned) == 0:
        raise ValueError("序列中没有有效的氨基酸字符。")

    if len(cleaned) < 5:
        raise ValueError(f"序列太短（{len(cleaned)} 个残基），至少需要 5 个。")

    return cleaned


# ============================================================================
# 核心预测函数
# ============================================================================

def predict_secondary_structure(sequence: str) -> Dict:
    """
    预测一条蛋白质序列的二级结构。

    Args:
        sequence: 氨基酸序列字符串，如 "MKVLILACLVALALA"
                  支持空格、换行，会自动清洗

    Returns:
        dict:
            - sequence:      原始序列
            - structure:     每个残基的结构标签（H/E/C），如 "CCEEEEHHHHHCC"
            - per_residue:   list of dict，每个残基的详细信息
            - counts:        {"H": n, "E": n, "C": n}
            - percentages:   {"H": pct, "E": pct, "C": pct}
            - length:        序列长度
    """
    model, tokenizer = _load_model()

    # 清洗序列
    seq = _validate_sequence(sequence)
    seq = seq[:MAX_SEQ_LEN]  # 截断

    # Tokenize
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

    # 推理
    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # [1, L, 3]
        preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()  # [L]
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()  # [L, 3]

    # 只取有效位置（非 padding）
    valid_len = min(len(seq), MAX_SEQ_LEN)
    preds = preds[:valid_len]
    probs = probs[:valid_len]

    # 构建结果
    structure = "".join(SS_IDX_TO_LABEL[p] for p in preds)

    per_residue = []
    for i, (aa, ss_idx) in enumerate(zip(seq, preds)):
        per_residue.append({
            "position": i + 1,
            "amino_acid": aa,
            "structure": SS_IDX_TO_LABEL[ss_idx],
            "structure_name": SS_LABEL_NAMES[SS_IDX_TO_LABEL[ss_idx]],
            "confidence": float(probs[i][ss_idx]),
            "probabilities": {
                "H": float(probs[i][0]),
                "E": float(probs[i][1]),
                "C": float(probs[i][2]),
            },
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

    # 测试用例
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
        print(f"置信度: {np.mean([r['confidence'] for r in result['per_residue']]):.3f}")
