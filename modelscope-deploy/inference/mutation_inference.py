"""
突变效应预测推理模块
基于 ESM-2 零样本突变打分（log-likelihood ratio）

方法：
  1. 将目标位置替换为 <mask>，用 ESM-2 计算所有氨基酸在该位置的 log-probability
  2. 突变效应分数 LLR = log P(mutant | context) - log P(wildtype | context)
  3. 同时计算 WT/Mutant 嵌入余弦距离作为"结构扰动"指标

参考文献：
  - Meier et al., "Language models enable zero-shot prediction of
    the effects of mutations on protein function", NeurIPS 2021
  - Lin et al., "ESM-2: Evolutionary-scale prediction of atomic-level
    protein structure with a language model", Science 2023
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# === 配置 ===
MAX_SEQ_LEN = 256
DEVICE = "cpu"

AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")

# 阈值（可根据 benchmark 调整）
PATHOGENIC_THRESHOLD = -0.5   # LLR < -0.5 → 可能致病
DISRUPTION_THRESHOLD = 0.08   # 余弦距离 > 0.08 → 结构显著变化

# 三字母 → 单字母
AA_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def _load_esm():
    """加载 ESM-2（复用 ec_inference 的缓存，避免重复加载）。"""
    from ec_inference import _load_esm as _ec_load_esm
    return _ec_load_esm()


# ============================================================================
# 突变解析
# ============================================================================

def parse_mutation(mutation_str: str):
    """
    解析突变字符串，支持格式：
      A67T        标准 HGVS
      Ala67Thr    三字母
      p.Ala67Thr  带 p. 前缀
      A 67 T      带空格

    Returns:
        (pos_0based: int, wildtype_aa: str, mutant_aa: str)
    """
    import re

    s = mutation_str.strip().replace(" ", "")

    for prefix in ["p.", "NP_", "XP_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]

    # A67T
    m = re.match(r'^([A-Z])(\d+)([A-Z\*])$', s)
    if m:
        wt, pos, mt = m.groups()
        if mt == "*":
            mt = "X"  # 终止密码子视为未知
        return int(pos) - 1, wt, mt

    # Ala67Thr
    m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', s)
    if m:
        wt3, pos, mt3 = m.groups()
        wt = AA_3TO1.get(wt3.upper(), "?")
        mt = AA_3TO1.get(mt3.upper(), "?")
        if wt != "?" and mt != "?":
            return int(pos) - 1, wt, mt

    raise ValueError(f"无法解析突变格式: {mutation_str!r}")


# ============================================================================
# 核心预测
# ============================================================================

def predict_mutation(sequence: str, mutation_str: str):
    """
    预测单个点突变的效应（零样本）。

    Args:
        sequence:       野生型氨基酸序列（单字母）
        mutation_str:   突变描述，如 "R175H"

    Returns:
        dict:
            - variant:           突变标识，如 "R175H"
            - position:          1-indexed 位置
            - wildtype_aa:       原始氨基酸
            - mutant_aa:         突变氨基酸
            - score:             突变效应分数 (LLR)，越负越有害
            - prediction:        中文预测结论
            - confidence:        置信度 0-1
            - structure_disruption: 嵌入余弦距离 0-1
            - wt_logp:           野生型 log-probability
            - mt_logp:           突变型 log-probability
    """
    tokenizer, esm_model = _load_esm()

    # ---- 预处理序列 ----
    seq = "".join(sequence.split()).upper()
    seq = "".join(aa for aa in seq if aa in "ACDEFGHIKLMNPQRSTVWY")
    seq = seq[:MAX_SEQ_LEN]

    if len(seq) < 10:
        raise ValueError(f"序列太短（{len(seq)} 个残基），至少需要 10 个。")

    # ---- 解析突变 ----
    pos_0based, wt_aa, mt_aa = parse_mutation(mutation_str)

    if pos_0based < 0 or pos_0based >= len(seq):
        raise ValueError(
            f"突变位置 {pos_0based + 1} 超出序列范围 1-{len(seq)}"
        )
    if seq[pos_0based] != wt_aa:
        raise ValueError(
            f"序列位置 {pos_0based + 1} 的氨基酸是 {seq[pos_0based]}，"
            f"与输入的 {wt_aa} 不匹配"
        )
    if wt_aa == mt_aa:
        raise ValueError("野生型和突变型是同一个氨基酸，没有突变。")
    if wt_aa not in AA_TOKENS or (mt_aa not in AA_TOKENS and mt_aa != "X"):
        raise ValueError(f"不支持的氨基酸: {wt_aa}→{mt_aa}")

    # ---- 1. 零样本 LLR 打分 ----
    seq_list = list(seq)
    seq_list[pos_0based] = "<mask>"
    masked_seq = " ".join(seq_list)

    tokens = tokenizer(
        masked_seq,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = esm_model(
            input_ids=tokens["input_ids"].to(DEVICE),
            attention_mask=tokens["attention_mask"].to(DEVICE),
        )
        logits = outputs.logits  # [1, L, vocab_size]

    # 定位 mask 的 logits
    mask_positions = (tokens["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )
    if len(mask_positions[1]) == 0:
        raise RuntimeError(
            "ESM tokenizer 未识别 <mask> token，请检查模型配置。"
        )

    mask_col = mask_positions[1][0].item()
    mask_logits = logits[0, mask_col, :]

    wt_token_id = tokenizer.convert_tokens_to_ids(wt_aa)
    mt_token_id = tokenizer.convert_tokens_to_ids(
        mt_aa if mt_aa != "X" else "<mask>"
    )

    logp = F.log_softmax(mask_logits, dim=-1)
    wt_logp = logp[wt_token_id].item()
    mt_logp = logp[mt_token_id].item()
    llr = mt_logp - wt_logp

    # ---- 2. 结构扰动：WT vs Mutant 嵌入余弦距离 ----
    wt_spaced = " ".join(list(seq))
    mt_list = list(seq)
    mt_list[pos_0based] = mt_aa if mt_aa != "X" else "X"
    mt_spaced = " ".join(mt_list)

    wt_tokens = tokenizer(
        wt_spaced, padding="max_length", max_length=MAX_SEQ_LEN,
        truncation=True, return_tensors="pt",
    )
    mt_tokens = tokenizer(
        mt_spaced, padding="max_length", max_length=MAX_SEQ_LEN,
        truncation=True, return_tensors="pt",
    )

    with torch.no_grad():
        wt_out = esm_model(
            input_ids=wt_tokens["input_ids"].to(DEVICE),
            attention_mask=wt_tokens["attention_mask"].to(DEVICE),
            output_hidden_states=True,
        )
        mt_out = esm_model(
            input_ids=mt_tokens["input_ids"].to(DEVICE),
            attention_mask=mt_tokens["attention_mask"].to(DEVICE),
            output_hidden_states=True,
        )

    # Mean pooling
    wt_mask = wt_tokens["attention_mask"].unsqueeze(-1).float()
    mt_mask = mt_tokens["attention_mask"].unsqueeze(-1).float()

    wt_emb = (wt_out.hidden_states[-1] * wt_mask).sum(dim=1) / wt_mask.sum(dim=1)
    mt_emb = (mt_out.hidden_states[-1] * mt_mask).sum(dim=1) / mt_mask.sum(dim=1)

    cos_sim = F.cosine_similarity(wt_emb, mt_emb).item()
    disruption = float(1.0 - cos_sim)

    # ---- 3. 综合判断 ----
    if llr < PATHOGENIC_THRESHOLD and disruption > DISRUPTION_THRESHOLD:
        prediction = "🔴 可能致病 (Pathogenic)"
        confidence = min(1.0, abs(llr) / 3.0 + disruption * 3.0)
    elif llr < PATHOGENIC_THRESHOLD or disruption > DISRUPTION_THRESHOLD:
        prediction = "🟠 可能影响功能 (Possibly Damaging)"
        confidence = min(1.0, abs(llr) / 4.0 + disruption * 2.0)
    elif llr > 0.5 and disruption < DISRUPTION_THRESHOLD / 2:
        prediction = "🟢 可能良性 (Likely Benign)"
        confidence = min(1.0, (1.0 - abs(llr) / 3.0) * (1.0 - disruption * 5.0))
    else:
        prediction = "🟡 不确定 (Uncertain)"
        confidence = 0.5

    return {
        "variant": f"{wt_aa}{pos_0based + 1}{mt_aa}",
        "position": pos_0based + 1,
        "wildtype_aa": wt_aa,
        "mutant_aa": mt_aa,
        "score": round(llr, 4),
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "structure_disruption": round(disruption, 4),
        "wt_logp": round(wt_logp, 4),
        "mt_logp": round(mt_logp, 4),
    }


# ============================================================================
# 批量 & 饱和突变扫描
# ============================================================================

def predict_mutations_batch(sequence: str, mutation_strs: list):
    """批量预测多个突变，返回按有害程度排序的结果。"""
    results = []
    errors = []

    for mut_str in mutation_strs:
        mut_str = mut_str.strip()
        if not mut_str:
            continue
        try:
            results.append(predict_mutation(sequence, mut_str))
        except Exception as e:
            errors.append({"mutation": mut_str, "error": str(e)})

    results.sort(key=lambda x: x["score"])
    return {"results": results, "errors": errors}


def score_all_variants(sequence: str, position: int):
    """
    饱和突变扫描：对指定位置的所有 19 种可能突变打分。

    Args:
        sequence:  野生型序列
        position:  1-indexed 位置

    Returns:
        list[dict]，按 score 排序
    """
    pos_0based = position - 1
    seq = "".join(sequence.split()).upper()
    seq = "".join(aa for aa in seq if aa in "ACDEFGHIKLMNPQRSTVWY")

    if pos_0based < 0 or pos_0based >= len(seq):
        raise ValueError(f"位置 {position} 超出序列范围 1-{len(seq)}")

    wt_aa = seq[pos_0based]
    results = []

    for mt_aa in AA_TOKENS:
        if mt_aa == wt_aa:
            continue
        try:
            results.append(
                predict_mutation(sequence, f"{wt_aa}{position}{mt_aa}")
            )
        except Exception:
            continue

    results.sort(key=lambda x: x["score"])
    return results
