"""
=============================================================================
[DNA] 蛋白质二级结构 3D 可视化
=============================================================================

功能：
  1. 下载蛋白质 PDB 结构文件
  2. 提取 Cα 原子的 3D 坐标（蛋白质骨架）
  3. 使用训练好的模型预测每个残基的二级结构（Q3）
  4. 生成 3D 对比图：
     - 实验标注的二级结构（HELIX/SHEET 记录）
     - 模型预测的二级结构
     - 差异高亮（预测错误的位置用红色标注）
  5. 同时生成 PyMOL 脚本（.pml），方便在 PyMOL 中交互式查看

输出目录：outputs/3d_vis/

作者：Bell | 日期：2026-07
=============================================================================
"""

import os
import sys
import math
import urllib.request
import time
import random
from pathlib import Path
from io import StringIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F

# 添加父目录到 path，以便导入 train.py 和 train_v2.py
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = SCRIPT_DIR / "outputs" / "3d_vis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# 可视化用的蛋白质（选一些经典的、结构多样的）
VIS_PROTEINS = [
    ("1UBQ", "Ubiquitin (76aa, alpha+beta)"),
    ("1L2Y", "Trp-cage (20aa, alpha-helix)"),
    ("1SHG", "SH3 Domain (57aa, beta-sheet)"),
    ("1CRN", "Crambin (46aa, small alpha+beta)"),
    ("4ICB", "Calcium-binding (75aa, EF-hand)"),
]

# Q3 颜色方案（与常用分子可视化软件一致）
SS_COLORS = {
    "H": "#FF6B6B",   # alpha-helix: 红色
    "E": "#4ECDC4",   # beta-sheet: 青色
    "C": "#95A5A6",   # coil: 灰色
}

SS_COLORS_RGB = {
    "H": (1.0, 0.42, 0.42),
    "E": (0.31, 0.80, 0.77),
    "C": (0.58, 0.65, 0.66),
}

ERROR_COLOR = "#E74C3C"  # 预测错误：亮红色


def download_pdb(pdb_id: str) -> str | None:
    """下载 PDB 文件。"""
    url = PDB_URL.format(pdb_id=pdb_id.upper())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Bell-Bio-AI/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [SKIP] {pdb_id} 下载失败: {e}")
        return None


def extract_backbone_coords(pdb_content: str) -> tuple[np.ndarray, str, str] | None:
    """
    从 PDB 文件提取 Cα 原子的 3D 坐标、序列和二级结构。

    Returns:
        (coords, sequence, ss_labels) 或 None
        coords: [N, 3] — Cα 原子的 (x, y, z) 坐标
        sequence: 氨基酸序列（单字母）
        ss_labels: Q3 二级结构标签（H/E/C）
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("pdb", StringIO(pdb_content))
    except Exception:
        return None

    model = structure[0]

    # 收集第一条蛋白链的残基和 Cα 坐标
    residues = []
    coords = []
    seq_chars = []

    for chain in model.get_chains():
        for res in chain:
            # 跳过异质残基和水分子
            if res.get_id()[0] != " ":
                continue

            aa = protein_letters_3to1.get(res.get_resname().upper(), None)
            if aa is None:
                continue

            # 获取 Cα 原子坐标
            ca_atom = None
            for atom in res:
                if atom.get_name() == "CA":
                    ca_atom = atom
                    break

            if ca_atom is None:
                continue

            coord = ca_atom.get_coord()
            residues.append(res)
            coords.append(coord)
            seq_chars.append(aa)

        break  # 只取第一条链

    if len(residues) < 20:
        return None

    coords = np.array(coords)
    sequence = "".join(seq_chars)

    # 从 HELIX/SHEET 记录提取实验二级结构
    ss_labels = ["C"] * len(residues)

    res_map = {}
    for i, res in enumerate(residues):
        rid = res.get_id()
        res_map[(rid[1], rid[2].strip())] = i

    # HELIX → H
    for line in pdb_content.split("\n"):
        if not line.startswith("HELIX"):
            continue
        try:
            init_num = int(line[21:25])
            init_icode = line[25].strip()
            end_num = int(line[33:37])
            end_icode = line[37].strip()
        except (ValueError, IndexError):
            continue
        start_key = (init_num, init_icode)
        end_key = (end_num, end_icode)
        if start_key in res_map and end_key in res_map:
            for i in range(res_map[start_key], res_map[end_key] + 1):
                ss_labels[i] = "H"

    # SHEET → E
    for line in pdb_content.split("\n"):
        if not line.startswith("SHEET "):
            continue
        try:
            init_num = int(line[22:26])
            init_icode = line[26].strip()
            end_num = int(line[33:37])
            end_icode = line[37].strip()
        except (ValueError, IndexError):
            continue
        start_key = (init_num, init_icode)
        end_key = (end_num, end_icode)
        if start_key in res_map and end_key in res_map:
            for i in range(res_map[start_key], res_map[end_key] + 1):
                ss_labels[i] = "E"

    ss_string = "".join(ss_labels)
    return coords, sequence, ss_string


def load_trained_model(model_path: str):
    """加载训练好的 PyTorch 模型。"""
    # 导入模型定义
    try:
        from train_v2 import ProteinSSPredictorV2, CONFIG as CONFIG_V2
        config = CONFIG_V2
        model = ProteinSSPredictorV2(config)
    except (ImportError, AttributeError):
        # 回退到 V1 模型
        from train import ProteinSSPredictor, CONFIG as CONFIG_V1
        config = CONFIG_V1
        model = ProteinSSPredictor(config)

    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")

    # 处理可能的 key 前缀差异
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 移除可能的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, config


def predict_ss(model, config: dict, sequence: str) -> str:
    """用模型预测一条序列的二级结构。"""
    from train import encode_sequence, IDX_TO_SS

    max_len = config.get("max_seq_len", 512)

    x = encode_sequence(sequence, max_len)
    x = torch.LongTensor(x).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

    ss_pred = "".join(IDX_TO_SS.get(p, "C") for p in preds[:len(sequence)])
    return ss_pred


def smooth_backbone(coords: np.ndarray, window: int = 1) -> np.ndarray:
    """平滑骨架坐标（移动平均），让 3D 线条更美观。"""
    smoothed = coords.copy().astype(float)
    for i in range(len(coords)):
        start = max(0, i - window)
        end = min(len(coords), i + window + 1)
        smoothed[i] = coords[start:end].mean(axis=0)
    return smoothed


def create_3d_visualization(
    coords: np.ndarray,
    sequence: str,
    ss_actual: str,
    ss_pred: str,
    pdb_id: str,
    description: str,
    q3_score: float,
):
    """
    创建 3D 蛋白质骨架可视化。

    生成三种视图：
      1. 实验标注的二级结构（左侧）
      2. 模型预测的二级结构（中间）
      3. 差异对比（右侧，错误预测高亮）
    """
    # 平滑骨架
    coords_smooth = smooth_backbone(coords, window=1)

    # 计算每个残基的颜色
    colors_actual = [SS_COLORS[s] for s in ss_actual]
    colors_pred = [SS_COLORS[s] for s in ss_pred]

    # 差异检测
    errors = [i for i in range(len(ss_actual)) if ss_actual[i] != ss_pred[i]]
    colors_diff = []
    for i in range(len(ss_actual)):
        if i in errors:
            colors_diff.append(ERROR_COLOR)
        else:
            colors_diff.append(SS_COLORS[ss_actual[i]])

    # 创建 3 行 1 列的图
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        f"{pdb_id}: {description}\n"
        f"Q3 Accuracy = {q3_score:.1%} ({len(errors)}/{len(ss_actual)} residues wrong)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ---- 子图 1: 实验标注 ----
    ax1 = fig.add_subplot(3, 1, 1, projection="3d")
    _plot_backbone_3d(ax1, coords_smooth, colors_actual, "Actual (HELIX/SHEET records)")

    # ---- 子图 2: 模型预测 ----
    ax2 = fig.add_subplot(3, 1, 2, projection="3d")
    _plot_backbone_3d(ax2, coords_smooth, colors_pred, "Predicted (Model V2)")

    # ---- 子图 3: 差异高亮 ----
    ax3 = fig.add_subplot(3, 1, 3, projection="3d")
    _plot_backbone_3d(ax3, coords_smooth, colors_diff, "Errors highlighted (Red = wrong prediction)")

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color=SS_COLORS["H"], linewidth=4, label="alpha-Helix (H)"),
        plt.Line2D([0], [0], color=SS_COLORS["E"], linewidth=4, label="beta-Sheet (E)"),
        plt.Line2D([0], [0], color=SS_COLORS["C"], linewidth=4, label="Coil/Loop (C)"),
        plt.Line2D([0], [0], color=ERROR_COLOR, linewidth=4, label="Prediction Error"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=5, fontsize=11, frameon=False,
    )

    # 保存
    save_path = OUTPUT_DIR / f"{pdb_id}_3d_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] 3D 图已保存: {save_path}")

    return save_path


def _plot_backbone_3d(ax, coords: np.ndarray, colors: list, title: str):
    """在 3D 坐标轴上绘制蛋白质骨架。"""
    # 每段单独绘制以实现颜色渐变
    for i in range(len(coords) - 1):
        ax.plot3D(
            [coords[i, 0], coords[i + 1, 0]],
            [coords[i, 1], coords[i + 1, 1]],
            [coords[i, 2], coords[i + 1, 2]],
            color=colors[i] if i < len(colors) else "#999999",
            linewidth=5,
            alpha=0.9,
            solid_capstyle="round",
        )

    # 标注 N 端和 C 端
    ax.scatter(*coords[0], color="#2ECC71", s=120, label="N-terminus", edgecolors="white", linewidth=1)
    ax.scatter(*coords[-1], color="#E74C3C", s=120, label="C-terminus", edgecolors="white", linewidth=1)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # 美化
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")


def generate_pymol_script(
    pdb_id: str,
    sequence: str,
    ss_actual: str,
    ss_pred: str,
    errors: list,
):
    """
    生成 PyMOL 脚本（.pml），可导入 PyMOL 进行交互式查看。

    脚本功能：
      - 从 PDB 下载结构
      - 按实验 SS 和预测 SS 分别上色
      - 高亮预测错误的位置
    """
    pml_content = f"""\
# ===========================================================================
# PyMOL 可视化脚本: {pdb_id}
# ===========================================================================
# 用法: 在 PyMOL 中运行 run {pdb_id}_view.pml
# ===========================================================================

# 加载结构
fetch {pdb_id}, async=0
hide everything
show cartoon

# 复制一份用于预测标注
create {pdb_id}_pred, {pdb_id}

# --- 实验标注 (左侧或上方) ---
# 根据 HELIX/SHEET 记录自动上色
color red, {pdb_id}

# --- 预测标注 (需要手动设置 per-residue B-factors) ---
# 运行以下命令逐残基上色:
# (这里用 B-factor 列来存储预测标签：H=1, E=2, C=3)
alter {pdb_id}_pred, b=0
"""

    # 为每个残基添加 alter 命令
    for i, (aa, ss_a, ss_p) in enumerate(zip(sequence, ss_actual, ss_pred)):
        resi = i + 1
        pml_content += f"# Residue {resi} ({aa}): actual={ss_a}, pred={ss_p}\n"

    pml_content += f"""
# 配色方案:
#   alpha-helix = red
#   beta-sheet  = cyan
#   coil/loop   = gray
#   预测错误    = yellow sticks

# 显示预测错误的残基
select errors_{pdb_id}, {pdb_id} and resi {"+".join(str(e + 1) for e in errors)}
show sticks, errors_{pdb_id}
color yellow, errors_{pdb_id}

# 调整视角
zoom {pdb_id}
bg_color white
set ray_trace_mode, 1
set cartoon_fancy_helices, 1
set cartoon_highlight_style, 2
set depth_cue, 0

# 标注
label {pdb_id} and n. CA and resi 1, "N"
label {pdb_id} and n. CA and resi {len(sequence)}, "C"

print "============================================"
print " {pdb_id}: Q3 errors = {len(errors)} / {len(sequence)}"
print "============================================"
"""
    save_path = OUTPUT_DIR / f"{pdb_id}_view.pml"
    with open(save_path, "w") as f:
        f.write(pml_content)
    print(f"  [OK] PyMOL 脚本已保存: {save_path}")

    return save_path


def create_ss_distribution_chart(ss_actual: str, ss_pred: str, pdb_id: str):
    """创建二级结构分布的条形对比图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    categories = ["H (Helix)", "E (Sheet)", "C (Coil)"]
    actual_counts = [ss_actual.count("H"), ss_actual.count("E"), ss_actual.count("C")]
    pred_counts = [ss_pred.count("H"), ss_pred.count("E"), ss_pred.count("C")]
    colors = [SS_COLORS["H"], SS_COLORS["E"], SS_COLORS["C"]]

    # 柱状图
    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width / 2, actual_counts, width, label="Actual", color=colors, edgecolor="white", alpha=0.9)
    axes[0].bar(x + width / 2, pred_counts, width, label="Predicted", color=colors, edgecolor="black", alpha=0.5, hatch="//")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("Residue Count")
    axes[0].set_title(f"SS Distribution: {pdb_id}")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # 饼图（实际）
    axes[1].pie(
        actual_counts,
        labels=categories,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title(f"Actual SS Composition ({len(ss_actual)} aa)")

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{pdb_id}_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] 分布图已保存: {save_path}")
    return save_path


def create_strip_chart(sequence: str, ss_actual: str, ss_pred: str, pdb_id: str):
    """
    创建蛋白质二级结构的条带图（strip chart），
    直观展示每个残基的实际 vs 预测二级结构。
    """
    n = len(sequence)
    fig, axes = plt.subplots(3, 1, figsize=(max(12, n * 0.15), 8))

    ss_map = {"H": 0, "E": 1, "C": 2}
    actual_vals = np.array([ss_map.get(s, 2) for s in ss_actual])
    pred_vals = np.array([ss_map.get(s, 2) for s in ss_pred])
    errors = (actual_vals != pred_vals)

    x = np.arange(n)

    # 子图 1: 氨基酸序列
    axes[0].set_xlim(-1, n)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"{pdb_id} — Amino Acid Sequence", fontweight="bold")
    for i, aa in enumerate(sequence):
        axes[0].text(i, 0.5, aa, ha="center", va="center",
                     fontsize=8, fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.1", facecolor="#F0F0F0", alpha=0.8))
    axes[0].set_yticks([])
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["left"].set_visible(False)

    # 子图 2: 实际二级结构
    axes[1].set_xlim(-1, n)
    axes[1].set_ylim(-0.5, 2.5)
    axes[1].set_title("Actual Secondary Structure (HELIX/SHEET)", fontweight="bold")
    for i in range(n):
        color = SS_COLORS[ss_actual[i]]
        axes[1].bar(i, actual_vals[i], width=0.9, color=color, alpha=0.85)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["H (Helix)", "E (Sheet)", "C (Coil)"])
    axes[1].grid(axis="y", alpha=0.2)

    # 子图 3: 预测二级结构 + 错误高亮
    axes[2].set_xlim(-1, n)
    axes[2].set_ylim(-0.5, 2.5)
    axes[2].set_title("Predicted Secondary Structure (Red outline = error)", fontweight="bold")
    for i in range(n):
        color = SS_COLORS[ss_pred[i]]
        edgecolor = "red" if errors[i] else color
        linewidth = 2 if errors[i] else 0.5
        axes[2].bar(i, pred_vals[i], width=0.9, color=color, alpha=0.85,
                    edgecolor=edgecolor, linewidth=linewidth)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["H (Helix)", "E (Sheet)", "C (Coil)"])
    axes[2].set_xlabel("Residue Index")
    axes[2].grid(axis="y", alpha=0.2)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{pdb_id}_stripchart.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [OK] 条带图已保存: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("[DNA] 蛋白质二级结构 3D 可视化")
    print("=" * 60)

    # ----- 1. 加载模型 -----
    model_dir = SCRIPT_DIR / "models"
    v2_path = model_dir / "best_model_v2.pt"
    v1_path = model_dir / "best_model.pt"

    model_path = None
    if v2_path.exists():
        model_path = v2_path
        print(f"\n[Model] 加载 V2 模型: {model_path}")
    elif v1_path.exists():
        model_path = v1_path
        print(f"\n[Model] 加载 V1 模型: {model_path}")
    else:
        print("\n[ERROR] 未找到训练好的模型文件！请先运行 train.py 或 train_v2.py")
        return

    model, config = load_trained_model(str(model_path))
    print(f"  [OK] 模型已加载")

    # ----- 2. 逐个蛋白质可视化 -----
    print(f"\n[Vis] 开始可视化 {len(VIS_PROTEINS)} 个蛋白质...")
    print("-" * 60)

    for pdb_id, description in VIS_PROTEINS:
        print(f"\n[Protein] {pdb_id}: {description}")

        # 下载 PDB
        pdb_content = download_pdb(pdb_id)
        if pdb_content is None:
            print(f"  [SKIP] 下载失败，跳过 {pdb_id}")
            continue

        # 提取坐标和实验二级结构
        result = extract_backbone_coords(pdb_content)
        if result is None:
            print(f"  [SKIP] 骨架提取失败")
            continue

        coords, sequence, ss_actual = result
        print(f"  Residues: {len(sequence)}, "
              f"H={ss_actual.count('H')}, E={ss_actual.count('E')}, C={ss_actual.count('C')}")

        # 模型预测
        ss_pred = predict_ss(model, config, sequence)

        # 计算 Q3
        correct = sum(1 for a, p in zip(ss_actual, ss_pred) if a == p)
        q3 = correct / len(ss_actual)
        errors = [i for i in range(len(ss_actual)) if ss_actual[i] != ss_pred[i]]
        print(f"  Q3 Accuracy: {q3:.1%} ({correct}/{len(ss_actual)}, {len(errors)} errors)")

        # 生成可视化
        create_3d_visualization(coords, sequence, ss_actual, ss_pred, pdb_id, description, q3)
        create_ss_distribution_chart(ss_actual, ss_pred, pdb_id)
        create_strip_chart(sequence, ss_actual, ss_pred, pdb_id)
        generate_pymol_script(pdb_id, sequence, ss_actual, ss_pred, errors)

        # 温和间隔
        time.sleep(0.3 + random.random() * 0.3)

    # ----- 3. 汇总报告 -----
    print("\n" + "=" * 60)
    print("[Done] 所有可视化已生成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"\n  生成的文件：")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.1f} KB)")
    print("\n  提示：")
    print("    - *_3d_comparison.png: 3D 骨架对比（实验 vs 预测 vs 错误）")
    print("    - *_distribution.png: 二级结构分布柱状图 + 饼图")
    print("    - *_stripchart.png: 逐残基的条带对比图")
    print("    - *_view.pml: 可用于 PyMOL 交互式查看的脚本")
    print("=" * 60)


if __name__ == "__main__":
    main()
