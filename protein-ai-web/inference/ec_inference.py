"""
EC 酶分类推理模块
基于 ESM-2 嵌入 + MLP 分类器，7 个 EC 一级类别
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# === 配置 ===
EC_CLASSES = {
    "1": "氧化还原酶 Oxidoreductase",
    "2": "转移酶 Transferase",
    "3": "水解酶 Hydrolase",
    "4": "裂合酶 Lyase",
    "5": "异构酶 Isomerase",
    "6": "连接酶 Ligase",
    "7": "转位酶 Translocase",
}

EC_DESCRIPTIONS = {
    "1": "催化氧化还原反应，涉及电子转移",
    "2": "催化功能基团从供体转移到受体",
    "3": "催化化学键的水解断裂",
    "4": "催化非水解、非氧化的化学键断裂",
    "5": "催化分子内异构化反应",
    "6": "催化两个分子通过共价键连接",
    "7": "催化离子或分子跨膜转运",
}

MAX_SEQ_LEN = 256
DEVICE = "cpu"

_model_cache = {}
_tokenizer_cache = None
_esm_model_cache = None


class ECClassifier(nn.Module):
    """轻量级 MLP 分类器，在 ESM-2 嵌入之上。"""

    def __init__(self, input_dim=320, hidden_dim=160, num_classes=7, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _load_esm():
    """加载 ESM-2 模型和 tokenizer（全局缓存）。"""
    global _tokenizer_cache, _esm_model_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache, _esm_model_cache

    from transformers import EsmTokenizer, EsmForMaskedLM

    # 优先用本地模型，没有再从 HF Hub 下载
    local_dir = Path(__file__).parent / "esm_model"
    if local_dir.exists():
        model_name = str(local_dir)
        print(f"[EC] 使用本地 ESM-2: {model_name}")
    else:
        model_name = "facebook/esm2_t6_8M_UR50D"
        print(f"[EC] 从 HF Hub 加载: {model_name}")

    _tokenizer_cache = EsmTokenizer.from_pretrained(model_name)
    _esm_model_cache = EsmForMaskedLM.from_pretrained(model_name).to(DEVICE)
    _esm_model_cache.eval()
    return _tokenizer_cache, _esm_model_cache


def _load_classifier():
    """加载 EC 分类器 MLP 权重。"""
    if "classifier" in _model_cache:
        return _model_cache["classifier"]

    model = ECClassifier(input_dim=320, hidden_dim=160, num_classes=7, dropout=0.3)
    ckpt_path = Path(__file__).parent / "ec_classifier.pt"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # checkpoint 可能包含 model_state_dict（训练完整 checkpoint）
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise FileNotFoundError(f"EC 模型未找到: {ckpt_path}")

    _model_cache["classifier"] = model
    return model


def _extract_embedding(sequence: str):
    """用 ESM-2 提取序列嵌入（mean pooling）。"""
    tokenizer, esm_model = _load_esm()
    seq = "".join(sequence.split()).upper()
    seq = "".join(aa for aa in seq if aa in "ACDEFGHIKLMNPQRSTVWY")
    seq = seq[:MAX_SEQ_LEN]

    if len(seq) < 10:
        raise ValueError(f"序列太短（{len(seq)} 个残基），至少需要 10 个。")

    tokens = tokenizer(
        " ".join(list(seq)),
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = esm_model(
            input_ids=tokens["input_ids"].to(DEVICE),
            attention_mask=tokens["attention_mask"].to(DEVICE),
            output_hidden_states=True,
        )
        # Mean pooling over sequence length (excluding special tokens)
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        embedding = (outputs.hidden_states[-1] * mask).sum(dim=1) / mask.sum(dim=1)
        embedding = embedding.squeeze().cpu()

    return embedding


def predict_ec(sequence: str):
    """
    预测蛋白质的 EC 酶类别。

    Returns:
        dict with keys:
            - predicted_class:  EC 类别编号 (str, "1"~"7")
            - predicted_name:   中文+英文名称
            - description:      功能描述
            - confidence:       预测置信度
            - probabilities:    所有 7 类的概率
            - all_results:      排序后的所有结果
    """
    embedding = _extract_embedding(sequence)
    classifier = _load_classifier()

    with torch.no_grad():
        logits = classifier(embedding.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = str(pred_idx + 1)  # EC classes are 1-indexed
    confidence = float(probs[pred_idx])

    all_results = []
    for i in range(7):
        ec_num = str(i + 1)
        all_results.append({
            "ec_class": ec_num,
            "name": EC_CLASSES[ec_num],
            "description": EC_DESCRIPTIONS[ec_num],
            "probability": float(probs[i]),
        })
    all_results.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "predicted_class": pred_class,
        "predicted_name": EC_CLASSES[pred_class],
        "description": EC_DESCRIPTIONS[pred_class],
        "confidence": confidence,
        "probabilities": {str(i + 1): float(probs[i]) for i in range(7)},
        "all_results": all_results,
    }
