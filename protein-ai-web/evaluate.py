"""
Protein AI — Evaluation Pipeline
评估脚本：计算 Q3、每类 F1、SOV 等指标

用法:
  python evaluate.py                          # 评估 CB513
  python evaluate.py --quick                  # 快速测试（内置示例）
  python evaluate.py --compare                # 对比不同配置
"""
import sys, os, re, json, argparse
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------- 评估指标 ----------

SS_CLASSES = ["H", "E", "C"]
SS_NAMES = {"H": "alpha-helix", "E": "beta-sheet", "C": "coil"}


def calc_q3(preds: list, labels: list) -> float:
    """Q3 Accuracy: 正确预测的残基比例"""
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    return correct / len(preds) if preds else 0


def calc_per_class_f1(preds: list, labels: list) -> dict:
    """计算每类 (H/E/C) 的 Precision, Recall, F1"""
    metrics = {}
    for cls in "HEC":
        tp = sum(1 for p, l in zip(preds, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(preds, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(preds, labels) if p != cls and l == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[cls] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    return metrics


def calc_sov(preds: list, labels: list) -> float:
    """Segment Overlap (SOV) measure"""
    segments_pred = _get_segments(preds)
    segments_true = _get_segments(labels)
    sov_sum = 0
    total_len = 0
    for cls in "HEC":
        true_segs = [s for s in segments_true if s[2] == cls]
        pred_segs = [s for s in segments_pred if s[2] == cls]
        for ts in true_segs:
            t_start, t_end, _ = ts
            max_ov = 0
            for ps in pred_segs:
                p_start, p_end, _ = ps
                ov_start = max(t_start, p_start)
                ov_end = min(t_end, p_end)
                ov = max(0, ov_end - ov_start + 1)
                if ov > 0:
                    n = t_end - t_start + 1
                    n_penalty = n
                    max_ov = max(max_ov, (ov + 1) * ov / n_penalty)
            sov_sum += max_ov
            total_len += t_end - t_start + 1
    return sov_sum / total_len if total_len > 0 else 0


def _get_segments(ss_list: list) -> list:
    """Extract consecutive segments from a list of SS labels"""
    segments = []
    if not ss_list:
        return segments
    start = 0
    for i in range(1, len(ss_list)):
        if ss_list[i] != ss_list[start]:
            segments.append((start, i - 1, ss_list[start]))
            start = i
    segments.append((start, len(ss_list) - 1, ss_list[start]))
    return segments


def evaluate(pred_seqs: list, true_seqs: list, names: list = None) -> dict:
    """
    完整评估: 返回 Q3, per-class F1, SOV, 混淆矩阵
    pred_seqs: list of predicted SS strings (e.g. ["CHHHHHEEECC...", ...])
    true_seqs: list of true SS strings
    names: optional list of sequence names
    """
    all_preds = list("".join(pred_seqs))
    all_labels = list("".join(true_seqs))
    
    q3 = calc_q3(all_preds, all_labels) * 100
    f1 = calc_per_class_f1(all_preds, all_labels)
    sov = calc_sov(all_preds, all_labels) * 100
    
    # 混淆矩阵
    cm = {a: {b: 0 for b in "HEC"} for a in "HEC"}
    for p, l in zip(all_preds, all_labels):
        if p in "HEC" and l in "HEC":
            cm[l][p] += 1
    
    result = {
        "q3": round(q3, 2),
        "sov": round(sov, 2),
        "f1_per_class": {cls: round(f1[cls]["f1"] * 100, 2) for cls in "HEC"},
        "precision": {cls: round(f1[cls]["precision"] * 100, 2) for cls in "HEC"},
        "recall": {cls: round(f1[cls]["recall"] * 100, 2) for cls in "HEC"},
        "confusion_matrix": cm,
        "total_residues": len(all_preds),
        "n_sequences": len(pred_seqs),
    }
    return result


# ---------- 数据加载 ----------

def load_cb513(data_dir: str = "inference/data") -> tuple:
    """
    加载 CB513 测试集
    返回: (names, seqs, structures)
    """
    seq_file = os.path.join(data_dir, "cb513_sequences.fasta")
    ss_file = os.path.join(data_dir, "cb513_structures.fasta")
    
    if not os.path.exists(seq_file) or not os.path.exists(ss_file):
        raise FileNotFoundError(
            f"CB513 数据未找到: {seq_file} 或 {ss_file}\n"
            f"请手动下载 CB513 并保存到 {data_dir}/ 目录\n"
            f"下载地址见: RESEARCH_PLAN.md"
        )
    
    names, seqs = _read_fasta(seq_file)
    _, structures = _read_fasta(ss_file)
    return names, seqs, structures


def _read_fasta(path: str) -> tuple:
    """读取 FASTA 文件，返回 (names, sequences)"""
    names, seqs = [], []
    with open(path, "r", encoding="utf-8") as f:
        name, seq = "", []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    seqs.append("".join(seq))
                    names.append(name)
                name = line[1:].split()[0]
                seq = []
            elif line:
                seq.append(line)
        if seq:
            seqs.append("".join(seq))
            names.append(name)
    return names, seqs


def save_cb513(names, seqs, structures, data_dir="inference/data"):
    """保存 CB513 为 FASTA 格式"""
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "cb513_sequences.fasta"), "w", encoding="utf-8") as f:
        for n, s in zip(names, seqs):
            f.write(f">{n}\n{s}\n")
    with open(os.path.join(data_dir, "cb513_structures.fasta"), "w", encoding="utf-8") as f:
        for n, s in zip(names, structures):
            f.write(f">{n}\n{s}\n")
    print(f"CB513 saved: {len(names)} sequences")


# ---------- 运行评估 ----------

def run_evaluation(model_config: str = "ensemble", data_dir: str = "inference/data"):
    """
    运行完整评估
    
    model_config: 
        "ensemble"  - 完整集成模型 (V1+ESM-2+平滑)
        "v1_only"   - 仅 V1 (CNN+BiLSTM)
        "esm2_only" - 仅 ESM-2
        "no_smooth" - 集成模型但无生物学平滑
    
    返回: dict with metrics
    """
    from inference import predict_secondary_structure
    from model_v1 import apply_biological_smoothing
    import inference as inf_mod
    
    # 备份原始配置
    orig_w1 = inf_mod.ENSEMBLE_W1
    orig_w2 = inf_mod.ENSEMBLE_W2
    orig_smooth = inf_mod.SMOOTH_MIN_RUN
    
    if model_config == "v1_only":
        inf_mod.ENSEMBLE_W1 = 1.0
        inf_mod.ENSEMBLE_W2 = 0.0
    elif model_config == "esm2_only":
        inf_mod.ENSEMBLE_W1 = 0.0
        inf_mod.ENSEMBLE_W2 = 1.0
    elif model_config == "no_smooth":
        inf_mod.SMOOTH_MIN_RUN = 999  # 禁止平滑
    
    names, seqs, true_ss = load_cb513(data_dir)
    
    pred_seqs = []
    for i, (name, seq) in enumerate(zip(names, seqs)):
        try:
            result = predict_secondary_structure(seq[:256])
            pred_seqs.append(result["structure"])
        except Exception as e:
            print(f"  [{i+1}/{len(seqs)}] {name}: error - {str(e)[:60]}")
            pred_seqs.append("C" * min(len(seq), 256))
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(seqs)}] ...")
    
    # 恢复原始配置
    inf_mod.ENSEMBLE_W1 = orig_w1
    inf_mod.ENSEMBLE_W2 = orig_w2
    inf_mod.SMOOTH_MIN_RUN = orig_smooth
    
    return evaluate(pred_seqs, [s[:len(p)] for s, p in zip(true_ss, pred_seqs)], names)


def print_report(results: dict, title: str = "Evaluation Report"):
    """打印评估报告"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Sequences:  {results['n_sequences']}")
    print(f"  Residues:   {results['total_residues']}")
    print(f"  Q3:         {results['q3']:.2f}%")
    print(f"  SOV:        {results['sov']:.2f}%")
    print(f"  Per-class F1:")
    for cls in "HEC":
        print(f"    {cls} ({SS_NAMES[cls]}):  "
              f"F1={results['f1_per_class'][cls]:.2f}%  "
              f"Prec={results['precision'][cls]:.2f}%  "
              f"Rec={results['recall'][cls]:.2f}%")
    print(f"  Confusion Matrix (true\\pred):")
    print(f"         H     E     C")
    cm = results['confusion_matrix']
    for true_cls in "HEC":
        row = f"    {true_cls}:  "
        for pred_cls in "HEC":
            row += f"{cm[true_cls][pred_cls]:5d} "
        print(row)
    print(f"{'='*50}\n")


# ---------- 快速测试 ----------

def quick_test():
    """用内置示例快速测试评估流水线"""
    print("Running quick test with example data...")
    pred = ["CHHHHHHHHHHHHHHHCCCC", "CCHHHHHHHHHHHHHHCCCCC"]
    true = ["CHHHHHHHHHHHHHHHCCCC", "CCCHHHHHHHHHHHHHHCCCC"]
    result = evaluate(pred, true, ["seq1", "seq2"])
    print_report(result, "Quick Test (example data)")
    return result


# ---------- Ablation 对比 ----------

def run_ablation(data_dir="inference/data"):
    """跑所有消融实验配置"""
    configs = [
        ("ensemble", "完整集成 (V1+ESM-2+平滑)"),
        ("v1_only", "仅 V1 (CNN+BiLSTM)"),
        ("esm2_only", "仅 ESM-2"),
        ("no_smooth", "集成无平滑"),
    ]
    results = {}
    for cfg, name in configs:
        print(f"\n--- {name} ---")
        try:
            r = run_evaluation(cfg, data_dir)
            print_report(r, name)
            results[cfg] = r
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            break
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein AI Evaluation")
    parser.add_argument("--quick", action="store_true", help="快速测试")
    parser.add_argument("--ablation", action="store_true", help="跑消融实验")
    parser.add_argument("--data", default="inference/data", help="数据目录")
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.ablation:
        run_ablation(args.data)
    else:
        try:
            r = run_evaluation("ensemble", args.data)
            print_report(r, "CB513 Evaluation")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("\n先跑 quick test 看看效果:")
            quick_test()
