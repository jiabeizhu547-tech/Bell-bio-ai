"""
=============================================================================
[DNA] Phase 5 — 蛋白质功能预测：酶分类（EC Number Prediction）
=============================================================================

基于 ESM-2 嵌入的酶委员会（EC）编号预测。
从蛋白质序列判断它属于哪类酶（7 大类 + 子类）。

方法：
  1. 从 SwissProt 下载带 EC 标注的酶序列
  2. 用 ESM-2 提取序列嵌入（冻结预训练权重）
  3. 训练轻量分类器（MLP + 注意力池化）
  4. 评估：Accuracy, Per-class F1

EC 一级分类（7 类）：
  EC 1: 氧化还原酶 (Oxidoreductases)
  EC 2: 转移酶   (Transferases)
  EC 3: 水解酶   (Hydrolases)
  EC 4: 裂合酶   (Lyases)
  EC 5: 异构酶   (Isomerases)
  EC 6: 连接酶   (Ligases)
  EC 7: 转位酶   (Translocases)

用法：
    python protein_function/train_ec_classifier.py

作者: 朱家贝 | 日期: 2026-07-12
=============================================================================
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 0. 配置
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "enzyme"
MODEL_DIR = BASE_DIR / "models"
FUNC_MODEL_DIR = BASE_DIR / "protein_function" / "models"
FUNC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 256
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 优先使用本地 ESM 模型（无需联网下载）
_LOCAL_ESM = BASE_DIR / "hf_space" / "esm_model"
if _LOCAL_ESM.exists():
    ESM_MODEL_NAME = str(_LOCAL_ESM)
    print(f"[Config] 使用本地 ESM-2: {ESM_MODEL_NAME}")
else:
    ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# EC 一级分类
EC_CLASSES = {
    "1": "氧化还原酶 Oxidoreductase",
    "2": "转移酶 Transferase",
    "3": "水解酶 Hydrolase",
    "4": "裂合酶 Lyase",
    "5": "异构酶 Isomerase",
    "6": "连接酶 Ligase",
    "7": "转位酶 Translocase",
}

# ============================================================================
# 1. 数据获取：从 SwissProt 下载酶序列
# ============================================================================

def fetch_enzyme_data():
    """
    从 UniProt/SwissProt 获取带 EC 编号的酶序列。

    优先使用本地缓存，否则通过 BioPython 在线查询。
    为保证可复现，同时生成 demo 数据用于快速测试。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seq_file = DATA_DIR / "enzyme_sequences.fasta"
    label_file = DATA_DIR / "enzyme_labels.txt"

    if seq_file.exists() and label_file.exists():
        print(f"[Data] 使用缓存: {seq_file}")
        return load_local_data(seq_file, label_file)

    # 尝试在线获取
    try:
        return fetch_from_uniprot(seq_file, label_file)
    except Exception as e:
        print(f"[Data] 在线获取失败 ({e})，生成 demo 数据用于测试流程")
        return generate_demo_data(seq_file, label_file)


def fetch_from_uniprot(seq_file, label_file):
    """
    通过 UniProt REST API 分页检索酶序列（TSV 格式，包含完整 EC 标注）。
    """
    print("[Data] 正在通过 UniProt REST API 检索酶序列...")

    import requests
    import time, csv, io

    sequences = []
    labels = []
    seen_ids = set()

    # 对每个 EC 大类分别查询（提高覆盖度且一次可拉满 500 条）
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    fields = "accession,ec,sequence"

    for ec_class in EC_CLASSES:
        class_seqs = 0
        params = {
            "query": f"ec:{ec_class}.* AND reviewed:true AND length:[30 TO 500]",
            "format": "tsv",
            "fields": fields,
            "size": 500,
        }
        page = 0
        max_pages = 10  # 每类最多 5000 条

        while page < max_pages:
            page += 1
            try:
                resp = requests.get(base_url, params=params, timeout=60)
                resp.raise_for_status()

                reader = csv.DictReader(io.StringIO(resp.text), delimiter="\t")
                batch_count = 0
                for row in reader:
                    accession = row.get("Entry", "")
                    if accession in seen_ids:
                        continue
                    seen_ids.add(accession)

                    ec_full = row.get("EC number", "")
                    if not ec_full:
                        continue
                    # 取第一个 EC 编号的一级分类
                    first_ec = ec_full.split(";")[0].strip()
                    if not first_ec or not first_ec[0].isdigit():
                        continue
                    ec_level1 = first_ec.split(".")[0]
                    if ec_level1 != ec_class:
                        continue

                    seq = row.get("Sequence", "").strip()
                    # 过滤非标准氨基酸
                    if any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in seq.upper()):
                        continue

                    sequences.append(seq.upper())
                    labels.append(ec_level1)
                    batch_count += 1
                    class_seqs += 1

                # 分页
                link_header = resp.headers.get("Link", "")
                if 'rel="next"' not in link_header:
                    break
                import re
                next_match = re.search(
                    r'<[^>]*cursor=([^&>]+)[^>]*>;\s*rel="next"', link_header
                )
                if next_match:
                    params["cursor"] = next_match.group(1)
                else:
                    break
                time.sleep(0.3)

            except requests.exceptions.RequestException as e:
                print(f"    EC {ec_class} 第 {page} 页失败: {e}")
                break

        print(f"  EC {ec_class} ({EC_CLASSES[ec_class]}): {class_seqs} 条")

    if len(sequences) == 0:
        raise RuntimeError("UniProt API 未返回任何酶序列")

    # 保存
    save_local_data(sequences, labels, seq_file, label_file)

    print(f"\n[Data] UniProt 酶序列总计: {len(sequences)} 条")
    for ec in sorted(set(labels), key=int):
        count = labels.count(ec)
        print(f"  EC {ec} ({EC_CLASSES[ec]}): {count} 条")

    return sequences, labels


def generate_demo_data(seq_file, label_file):
    """
    生成 demo 酶分类数据（有真实生物意义的简化版）。
    每种 EC 类别使用特定的氨基酸偏好模式生成序列。
    """
    print("[Data] 生成 demo 酶分类数据...")
    np.random.seed(42)

    # 每类酶的氨基酸偏好（基于真实酶家族特征简化）
    ec_aa_profiles = {
        "1": "ACDEFGHIKLMNPQRSTVWY",  # 氧化还原酶：富 Cys, His (金属结合)
        "2": "ACDEFGHIKLMNPQRSTVWY",  # 转移酶：富 Lys, Asp (底物结合)
        "3": "ACDEFGHIKLMNPQRSTVWY",  # 水解酶：富 Ser, His, Asp (催化三联体)
        "4": "ACDEFGHIKLMNPQRSTVWY",  # 裂合酶：富 Cys, Lys
        "5": "ACDEFGHIKLMNPQRSTVWY",  # 异构酶：富 His, Glu
        "6": "ACDEFGHIKLMNPQRSTVWY",  # 连接酶：富 Lys, Arg (ATP 结合)
        "7": "ACDEFGHIKLMNPQRSTVWY",  # 转位酶：富 Arg, Lys (膜结合)
    }

    # 每类酶的 motif（简化版真实 motif）
    ec_motifs = {
        "1": ["CXXC", "CXXH", "GXGXXG", "CXXXXC"],
        "2": ["GXGXXG", "KXXXXK", "DXD", "HXH"],
        "3": ["GXSXG", "HXXD", "SXXK", "DXXG"],
        "4": ["CXXC", "KXXK", "YXXG", "RXXD"],
        "5": ["HXXE", "KXXD", "CXXC", "GXXG"],
        "6": ["KXXG", "RXXD", "GXXGXG", "SXXK"],
        "7": ["RXXR", "KXXK", "LXXL", "AXXA"],
    }

    sequences = []
    labels = []

    for ec_level, name in EC_CLASSES.items():
        n_samples = 120
        profile = ec_aa_profiles[ec_level]
        motifs = ec_motifs[ec_level]

        for _ in range(n_samples):
            # 生成 80-250 长度的序列
            length = np.random.randint(80, 250)
            seq = []

            # 随机生成骨架序列
            for j in range(length):
                seq.append(profile[np.random.randint(len(profile))])

            # 在随机位置插入 motif
            for motif in motifs:
                motif_seq = motif.replace("X", profile[np.random.randint(len(profile))])
                pos = np.random.randint(0, max(1, length - len(motif_seq)))
                for k, aa in enumerate(motif_seq):
                    if pos + k < length:
                        seq[pos + k] = aa

            sequences.append("".join(seq))
            labels.append(ec_level)

    save_local_data(sequences, labels, seq_file, label_file)

    print(f"[Data] Demo 酶序列: {len(sequences)} 条")
    for ec in sorted(set(labels)):
        print(f"  EC {ec}: {labels.count(ec)} 条")

    return sequences, labels


def save_local_data(sequences, labels, seq_file, label_file):
    with open(seq_file, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">enzyme_{i}\n{seq}\n")
    with open(label_file, "w") as f:
        for label in labels:
            f.write(f"{label}\n")


def load_local_data(seq_file, label_file):
    sequences = []
    with open(seq_file) as f:
        for line in f:
            if not line.startswith(">"):
                sequences.append(line.strip())

    with open(label_file) as f:
        labels = [line.strip() for line in f]

    print(f"[Data] 加载酶序列: {len(sequences)} 条")
    return sequences, labels


# ============================================================================
# 2. ESM-2 嵌入提取
# ============================================================================

class EnzymeDataset(Dataset):
    """酶序列数据集，用于 ESM-2 嵌入提取。"""

    def __init__(self, sequences, labels, tokenizer, max_len):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

        # Tokenize 所有序列
        self.input_ids = []
        self.attention_masks = []

        for seq in tqdm(sequences, desc="Tokenizing"):
            spaced = " ".join(list(seq[:max_len]))
            tokens = tokenizer(
                spaced,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids.append(tokens["input_ids"].squeeze(0))
            self.attention_masks.append(tokens["attention_mask"].squeeze(0))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
        }


@torch.no_grad()
def extract_esm2_embeddings(dataloader, esm_model, device):
    """
    用 ESM-2 提取每条序列的嵌入向量（平均池化所有残基的 hidden states）。
    Returns: embeddings [N, 320], labels [N]
    """
    esm_model.eval()
    all_embeddings = []
    all_labels = []

    with tqdm(dataloader, desc="提取 ESM-2 嵌入") as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # 取最后一层 hidden states [B, L, 320]
            last_hidden = outputs.hidden_states[-1]

            # 平均池化（只对非 padding 位置）
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            masked_hidden = last_hidden * mask_expanded
            summed = masked_hidden.sum(dim=1)  # [B, 320]
            lengths = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
            embeddings = summed / lengths  # [B, 320]

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(batch["label"])

    return np.concatenate(all_embeddings), all_labels


# ============================================================================
# 3. EC 分类器
# ============================================================================

class ECClassifier(nn.Module):
    """
    轻量级 EC 分类器，在 ESM-2 嵌入之上训练。

    架构：
      ESM-2 嵌入 (320-dim)
        ↓ Dropout
        ↓ Linear(320 → 160) + GELU + BatchNorm
        ↓ Dropout
        ↓ Linear(160 → 64) + GELU
        ↓ Linear(64 → num_classes)
    """

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


class EmbeddingDataset(Dataset):
    """预提取的嵌入数据集，用于分类器训练。"""

    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        # labels 已经是整数编码（0, 1, 2, ...）
        self.labels = torch.LongTensor(list(labels))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ============================================================================
# 4. 训练
# ============================================================================

def train_classifier(model, train_loader, val_loader, epochs=50, lr=1e-3):
    """训练 EC 分类器。"""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None
    patience = 10
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)

        train_acc = correct / total
        history["train_loss"].append(total_loss / total)
        history["train_acc"].append(train_acc)

        # Val
        val_acc = evaluate_classifier(model, val_loader)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"Train Loss: {history['train_loss'][-1]:.4f}  "
                  f"Train Acc: {train_acc:.3f}  "
                  f"Val Acc: {val_acc:.3f}  "
                  f"{'*' if val_acc == best_val_acc else ''}")

        if no_improve >= patience:
            print(f"  早停 @ Epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc, history


@torch.no_grad()
def evaluate_classifier(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total


# ============================================================================
# 5. 可视化
# ============================================================================

def plot_ec_results(history, all_labels, all_preds, label_names, save_path):
    """生成 EC 分类结果图。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ---- 左图: 训练曲线 ----
    ax = axes[0]
    ax.plot(history["train_acc"], label="Train Acc", color="#6366f1", linewidth=2)
    ax.plot(history["val_acc"], label="Val Acc", color="#f59e0b", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 中图: 混淆矩阵 ----
    ax = axes[1]
    cm = confusion_matrix(all_labels, all_preds)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)

    # ---- 右图: Per-class F1 ----
    ax = axes[2]
    report = classification_report(all_labels, all_preds, output_dict=True,
                                    zero_division=0)
    classes = [l for l in label_names if l in report]
    f1_scores = [report[c]["f1-score"] for c in classes]
    colors = plt.cm.tab10(range(len(classes)))

    bars = ax.bar(classes, f1_scores, color=colors, edgecolor="white")
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f1:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(pad=2)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] 结果图已保存: {save_path}")


# ============================================================================
# 6. 主流程
# ============================================================================

def main():
    print("=" * 60)
    print("[DNA] Phase 5 — 蛋白质功能预测：EC 酶分类")
    print(f"[Device] {DEVICE.upper()}")
    print("=" * 60)

    # ----- 6.1 获取数据 -----
    print("\n[Step 1] 获取酶序列数据...")
    sequences, labels_str = fetch_enzyme_data()

    label_to_idx = {ec: i for i, ec in enumerate(sorted(set(labels_str)))}
    idx_to_label = {i: ec for ec, i in label_to_idx.items()}
    num_classes = len(label_to_idx)
    label_names = [EC_CLASSES.get(idx_to_label[i], f"EC {idx_to_label[i]}")
                   for i in range(num_classes)]

    print(f"\n  类别数: {num_classes}")
    for i, name in enumerate(label_names):
        count = labels_str.count(idx_to_label[i])
        print(f"    {i}: {name} — {count} 条")

    # 编码标签
    labels = np.array([label_to_idx[l] for l in labels_str])

    # 划分数据集
    seq_train, seq_temp, lab_train, lab_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels,
    )
    seq_val, seq_test, lab_val, lab_test = train_test_split(
        seq_temp, lab_temp, test_size=0.5, random_state=42, stratify=lab_temp,
    )

    print(f"\n  训练: {len(seq_train)}  验证: {len(seq_val)}  测试: {len(seq_test)}")

    # ----- 6.2 加载 ESM-2 并提取嵌入 -----
    print("\n[Step 2] 加载 ESM-2 并提取嵌入...")

    from transformers import EsmTokenizer, AutoModel

    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_base = AutoModel.from_pretrained(ESM_MODEL_NAME).to(DEVICE)
    esm_base.eval()

    # 创建 DataLoader
    train_dataset = EnzymeDataset(seq_train, lab_train, tokenizer, MAX_SEQ_LEN)
    val_dataset = EnzymeDataset(seq_val, lab_val, tokenizer, MAX_SEQ_LEN)
    test_dataset = EnzymeDataset(seq_test, lab_test, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 提取嵌入
    print("\n  提取训练集嵌入...")
    train_emb, _ = extract_esm2_embeddings(train_loader, esm_base, DEVICE)
    print(f"  训练嵌入: {train_emb.shape}")

    print("  提取验证集嵌入...")
    val_emb, _ = extract_esm2_embeddings(val_loader, esm_base, DEVICE)
    print(f"  验证嵌入: {val_emb.shape}")

    print("  提取测试集嵌入...")
    test_emb, _ = extract_esm2_embeddings(test_loader, esm_base, DEVICE)
    print(f"  测试嵌入: {test_emb.shape}")

    # ----- 6.3 训练分类器 -----
    print(f"\n[Step 3] 训练 EC 分类器 ({num_classes} 类)...")

    train_emb_dataset = EmbeddingDataset(train_emb, lab_train)
    val_emb_dataset = EmbeddingDataset(val_emb, lab_val)
    test_emb_dataset = EmbeddingDataset(test_emb, lab_test)

    emb_train_loader = DataLoader(train_emb_dataset, batch_size=64, shuffle=True)
    emb_val_loader = DataLoader(val_emb_dataset, batch_size=64, shuffle=False)
    emb_test_loader = DataLoader(test_emb_dataset, batch_size=64, shuffle=False)

    classifier = ECClassifier(
        input_dim=train_emb.shape[1],
        num_classes=num_classes,
        dropout=0.3,
    )
    classifier, best_val_acc, history = train_classifier(
        classifier, emb_train_loader, emb_val_loader,
        epochs=80, lr=1e-3,
    )

    # ----- 6.4 测试评估 -----
    print(f"\n[Step 4] 测试集评估...")
    test_acc = evaluate_classifier(classifier, emb_test_loader)

    # 获取预测
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in emb_test_loader:
            x = x.to(DEVICE)
            logits = classifier(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print(f"\n{'='*60}")
    print(f"[Result] EC 酶分类结果（测试集）")
    print(f"{'='*60}")
    print(f"  测试准确率: {test_acc:.3f} ({test_acc*100:.1f}%)")
    print(f"\n  分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=label_names, zero_division=0))

    # ----- 6.5 可视化 -----
    plot_ec_results(
        history, all_labels, all_preds, label_names,
        FUNC_MODEL_DIR / "ec_classification_results.png",
    )

    # ----- 6.6 保存模型 -----
    torch.save({
        "model_state_dict": classifier.state_dict(),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "test_accuracy": test_acc,
    }, FUNC_MODEL_DIR / "ec_classifier.pt")
    print(f"[Model] 分类器已保存: {FUNC_MODEL_DIR / 'ec_classifier.pt'}")

    # ----- 6.7 汇总 -----
    print(f"\n{'='*60}")
    print("[Summary] Phase 5 — 蛋白质功能预测")
    print("=" * 60)
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  EC 酶分类（{num_classes} 类）                          │
  │  ESM-2 嵌入 (frozen) + MLP 分类头                     │
  │  测试准确率: {test_acc*100:.1f}%                              │
  │                                                     │
  │  输入: 氨基酸序列                                     │
  │  输出: EC 编号（酶功能分类）                           │
  └─────────────────────────────────────────────────────┘
""")

    return classifier, test_acc


if __name__ == "__main__":
    main()
