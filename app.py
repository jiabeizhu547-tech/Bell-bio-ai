"""
=============================================================================
[DNA] Phase 3 — 蛋白质二级结构预测 Web 工具
=============================================================================

基于 ESM-2 微调模型的 Gradio Web 应用。
输入氨基酸序列 → 秒出每个残基的二级结构预测（H=α螺旋, E=β折叠, C=卷曲）

作者: Bell | 日期: 2026-07-12
=============================================================================
"""

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import numpy as np
import io, tempfile
from inference import predict_secondary_structure

# ============================================================================
# 配色方案 — 亮色简约
# ============================================================================

# 结构类型颜色（醒目）
SS_COLORS = {
    "H": "#DC2626",  # 红色 — α-螺旋
    "E": "#2563EB",  # 蓝色 — β-折叠
    "C": "#D97706",  # 琥珀色 — 卷曲
}

SS_BG_COLORS = {
    "H": "#FEE2E2",  # 浅红底
    "E": "#DBEAFE",  # 浅蓝底
    "C": "#FEF3C7",  # 浅黄底
}

SS_NAMES = {
    "H": "α-螺旋",
    "E": "β-折叠",
    "C": "无规卷曲",
}

SS_ENAMES = {
    "H": "Alpha-helix",
    "E": "Beta-sheet",
    "C": "Coil",
}


# ============================================================================
# 核心预测函数
# ============================================================================

def predict(sequence: str):
    if not sequence or not sequence.strip():
        return None, "", "", "", ""

    try:
        result = predict_secondary_structure(sequence)
    except ValueError as e:
        return None, f"**输入错误**：{e}", "", "", ""
    except FileNotFoundError as e:
        return None, f"**模型未找到**：{e}", "", "", ""
    except Exception as e:
        import traceback
        return None, f"**预测失败**：{e}\n```\n{traceback.format_exc()}\n```", "", "", ""

    seq = result["sequence"]
    ss = result["structure"]
    pct = result["percentages"]
    counts = result["counts"]
    confidences = [r["confidence"] for r in result["per_residue"]]
    avg_conf = np.mean(confidences)

    # ---- 1. 彩色序列展示 ----
    colored_html = _build_colored_html(seq, ss)

    # ---- 2. 统计卡片 ----
    stats_html = _build_stats_cards(counts, pct, result["length"], avg_conf)

    # ---- 3. 纯文本结构 ----
    structure_text = f"```\n序列:  {seq}\n结构:  {ss}\n图例:  H=α螺旋  E=β折叠  C=卷曲\n```"

    # ---- 4. 可视化图 ----
    fig = _build_visualization(seq, ss, confidences, counts, pct)
    tmp_path = tempfile.mktemp(suffix=".png")
    fig.savefig(tmp_path, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    return colored_html, stats_html, structure_text, "", tmp_path


# ============================================================================
# HTML 构建
# ============================================================================

def _build_colored_html(seq: str, ss: str) -> str:
    """构建彩色序列展示 HTML — 每个残基一个色块，按行排列。"""
    chars_per_row = 60
    parts = [
        '<div style="font-family: \'JetBrains Mono\', \'Cascadia Code\', \'Consolas\', monospace; '
        'font-size: 15px; line-height: 2.2; background: #ffffff; '
        'border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 24px; '
        'box-shadow: 0 1px 3px rgba(0,0,0,0.06);">'
    ]

    for row_start in range(0, len(seq), chars_per_row):
        chunk_seq = seq[row_start:row_start + chars_per_row]
        chunk_ss = ss[row_start:row_start + chars_per_row]

        # 位置标尺
        if row_start == 0:
            parts.append(
                '<div style="display: flex; flex-wrap: wrap; margin-bottom: 2px; '
                'color: #9ca3af; font-size: 9px; font-weight: 600;">'
            )
            for j in range(len(chunk_seq)):
                pos = j + 1
                if pos == 1 or pos % 10 == 0:
                    parts.append(
                        f'<span style="min-width: 19px; text-align: left;">{pos}</span>'
                    )
                else:
                    parts.append('<span style="min-width: 19px;"></span>')
            parts.append('</div>')

        # 氨基酸行
        parts.append('<div style="display: flex; flex-wrap: wrap;">')
        for j, (aa, s) in enumerate(zip(chunk_seq, chunk_ss)):
            bg = SS_BG_COLORS.get(s, "#f3f4f6")
            fg = SS_COLORS.get(s, "#6b7280")
            pos = row_start + j + 1
            parts.append(
                f'<span style="background: {bg}; color: {fg}; font-weight: 700; '
                f'min-width: 19px; height: 26px; line-height: 26px; text-align: center; '
                f'border-radius: 3px; margin: 1px; display: inline-block; '
                f'font-size: 14px;" '
                f'title="位置 {pos}: {aa} → {SS_NAMES.get(s, s)}">{aa}</span>'
            )
        parts.append('</div>')

        # 结构标签行
        parts.append('<div style="display: flex; flex-wrap: wrap; margin-bottom: 6px;">')
        for j, s in enumerate(chunk_ss):
            fg = SS_COLORS.get(s, "#6b7280")
            parts.append(
                f'<span style="color: {fg}; font-weight: 800; '
                f'min-width: 19px; text-align: center; font-size: 11px; '
                f'margin: 0 1px; display: inline-block;">{s}</span>'
            )
        parts.append('</div>')

    parts.append('</div>')
    return "".join(parts)


def _build_stats_cards(counts, pct, length, avg_conf):
    """构建统计信息卡片（HTML 格式）。"""
    cards = []
    configs = [
        ("H", "α-螺旋 Alpha-helix", "蛋白质内部最稳定的规则结构，由主链氢键维持"),
        ("E", "β-折叠 Beta-sheet", "多条肽链片段平行或反平行排列形成的片层"),
        ("C", "无规卷曲 Coil", "连接螺旋和折叠的柔性环区，功能上常为活性位点"),
    ]

    for ss_type, full_name, desc in configs:
        color = SS_COLORS[ss_type]
        bg = SS_BG_COLORS[ss_type]
        cards.append(
            f'<div style="flex: 1; min-width: 180px; background: #ffffff; '
            f'border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px 20px; '
            f'box-shadow: 0 1px 3px rgba(0,0,0,0.05);">'
            f'<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">'
            f'<span style="background: {color}; color: #fff; font-weight: 800; font-size: 24px; '
            f'width: 44px; height: 44px; border-radius: 10px; display: flex; '
            f'align-items: center; justify-content: center;">{ss_type}</span>'
            f'<div>'
            f'<div style="font-weight: 700; font-size: 16px; color: #111827;">{full_name}</div>'
            f'<div style="font-size: 13px; color: #6b7280;">{desc}</div>'
            f'</div></div>'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; '
            f'border-top: 1px solid #f3f4f6; padding-top: 10px; margin-top: 4px;">'
            f'<span style="font-size: 13px; color: #6b7280;">残基数</span>'
            f'<span style="font-weight: 800; font-size: 22px; color: {color};">{counts[ss_type]}</span>'
            f'<span style="font-size: 13px; color: #6b7280;">占比</span>'
            f'<span style="font-weight: 800; font-size: 22px; color: {color};">{pct[ss_type]}%</span>'
            f'</div></div>'
        )

    # 汇总行
    summary = (
        f'<div style="display: flex; gap: 10px; margin-top: 12px; '
        f'background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; '
        f'padding: 12px 20px; align-items: center; justify-content: center; gap: 32px;">'
        f'<div><span style="color: #6b7280; font-size: 13px;">序列长度 </span>'
        f'<span style="font-weight: 800; font-size: 20px; color: #111827;">{length}</span>'
        f'<span style="color: #9ca3af; font-size: 12px;"> aa</span></div>'
        f'<div><span style="color: #6b7280; font-size: 13px;">平均置信度 </span>'
        f'<span style="font-weight: 800; font-size: 20px; color: #111827;">{avg_conf:.3f}</span></div>'
        f'</div>'
    )

    return (
        '<div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 8px;">'
        + "".join(cards)
        + '</div>'
        + summary
    )


# ============================================================================
# 可视化图表 — 亮色简约风格
# ============================================================================

def _build_visualization(seq, ss, confidences, counts, pct):
    """生成亮色主题的预测可视化。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "text.color": "#374151",
        "axes.edgecolor": "#d1d5db",
        "axes.labelcolor": "#4b5563",
        "xtick.color": "#6b7280",
        "ytick.color": "#6b7280",
        "grid.alpha": 0.4,
        "axes.facecolor": "#fafafa",
    })

    fig = plt.figure(figsize=(15, 7), facecolor="white")

    # ---- 子图 1: 残基条带图（最多显示 250 个） ----
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.set_facecolor("#fafafa")

    display_len = min(len(seq), 250)
    x_positions = range(display_len)
    colors = [SS_COLORS[s] for s in ss[:display_len]]

    ax1.bar(x_positions, [1] * display_len, color=colors, width=1.0, linewidth=0)
    ax1.set_xlim(-1, display_len)
    ax1.set_ylim(0, 1.15)
    ax1.set_yticks([])
    ax1.set_xlabel(f"Residue position  (showing first {display_len} of {len(seq)})",
                   fontweight="bold")
    ax1.set_title("Per-Residue Secondary Structure Prediction", fontweight="bold",
                  fontsize=14, pad=12)

    # 图例
    legend_elements = [
        patches.Patch(facecolor=SS_COLORS["H"], label=f"H  α-Helix ({pct['H']}%)"),
        patches.Patch(facecolor=SS_COLORS["E"], label=f"E  β-Sheet ({pct['E']}%)"),
        patches.Patch(facecolor=SS_COLORS["C"], label=f"C  Coil ({pct['C']}%)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", framealpha=0.9,
               facecolor="white", edgecolor="#d1d5db", fontsize=10)

    # ---- 子图 2: 饼图 ----
    ax2 = fig.add_subplot(2, 2, 3)
    labels = [f"H  {pct['H']}%", f"E  {pct['E']}%", f"C  {pct['C']}%"]
    sizes = [counts["H"], counts["E"], counts["C"]]
    pie_colors = [SS_COLORS["H"], SS_COLORS["E"], SS_COLORS["C"]]
    explode = (0.03, 0.03, 0.03)

    wedges, texts = ax2.pie(
        sizes, labels=labels, colors=pie_colors, explode=explode,
        startangle=90, textprops={"fontsize": 12, "fontweight": "bold",
                                   "color": "#111827"}
    )
    for w in wedges:
        w.set_edgecolor("white")
        w.set_linewidth(1.5)
    ax2.set_title("Structure Distribution", fontweight="bold", fontsize=14, pad=12)

    # ---- 子图 3: 置信度分布 ----
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_facecolor("#fafafa")
    ax3.hist(confidences, bins=25, color="#6366f1", edgecolor="white",
             alpha=0.85, linewidth=1)
    ax3.axvline(x=np.mean(confidences), color="#DC2626", linestyle="--",
                linewidth=2.5, label=f"Mean: {np.mean(confidences):.3f}")
    ax3.set_xlabel("Confidence", fontweight="bold")
    ax3.set_ylabel("Count", fontweight="bold")
    ax3.set_title("Prediction Confidence Distribution", fontweight="bold",
                  fontsize=14, pad=12)
    ax3.legend(facecolor="white", edgecolor="#d1d5db", fontsize=10,
               loc="upper left")
    ax3.set_xlim(0.25, 1.05)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout(pad=2)
    return fig


# ============================================================================
# 示例
# ============================================================================

EXAMPLES = [
    ["MKVLILACLVALALACTVQAKTENPKKT"],
    ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"],
    ["MVKVGINGFGRIGRLVTRAAFNSGKVDIVAIND"],
    ["GAMGINTRLSQQQPY"],
]

# ============================================================================
# CSS
# ============================================================================

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 1.5rem !important;
}
body, .gradio-container {
    background: #f8f9fa !important;
    color: #111827 !important;
}
h1 { font-size: 2rem !important; font-weight: 800 !important; color: #111827 !important; }
h3 { font-weight: 600 !important; color: #374151 !important; }
label { font-weight: 600 !important; color: #374151 !important; font-size: 14px !important; }
textarea {
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 14px !important;
    background: #ffffff !important;
    color: #111827 !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 12px !important;
    line-height: 1.8 !important;
}
textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
button.primary {
    background: #6366f1 !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
}
button.primary:hover {
    background: #4f46e5 !important;
}
footer { display: none !important; }
"""

# ============================================================================
# Gradio 界面
# ============================================================================

with gr.Blocks(title="蛋白质二级结构预测") as demo:

    # ---- 头部 ----
    gr.Markdown(
        """
        # 🧬 Protein Secondary Structure Prediction
        ### ESM-2 深度学习模型 &nbsp;·&nbsp; Q3 Accuracy 89.2% &nbsp;·&nbsp; 785 PDB Proteins

        输入氨基酸序列，秒出每个残基的二级结构类型。
        """
    )

    # ---- 输入区 ----
    with gr.Row():
        with gr.Column(scale=3):
            input_seq = gr.Textbox(
                label="氨基酸序列",
                placeholder="例如: MKVLILACLVALALACTVQAKTENPKKT  （使用 20 种标准氨基酸单字母缩写）",
                lines=4,
                max_lines=10,
            )
            with gr.Row():
                submit_btn = gr.Button("开始预测", variant="primary", scale=3)
                clear_btn = gr.Button("清空", scale=1)

        with gr.Column(scale=2):
            gr.Markdown(
                """
                #### 使用说明
                - 粘贴**氨基酸序列**，单字母大写
                - 支持 **20 种标准氨基酸**（ACDEFGHIKLMNPQRSTVWY）
                - 最多 **256 个残基**，自动过滤非法字符
                - 空格和换行会被自动清除
                """
            )

    gr.Examples(examples=EXAMPLES, inputs=input_seq, label="示例序列")

    gr.Markdown("---")

    # ---- 结果区 ----
    gr.Markdown("### 预测结果")

    html_output = gr.HTML()

    stats_output = gr.HTML()

    structure_output = gr.Markdown()

    plot_output = gr.Image(label="可视化分析", type="filepath")

    status_output = gr.Markdown(visible=False)

    # ---- 事件 ----
    submit_btn.click(
        fn=predict,
        inputs=[input_seq],
        outputs=[html_output, stats_output, structure_output, status_output, plot_output],
    )

    input_seq.submit(
        fn=predict,
        inputs=[input_seq],
        outputs=[html_output, stats_output, structure_output, status_output, plot_output],
    )

    clear_btn.click(
        fn=lambda: (None, "", "", "", ""),
        inputs=[],
        outputs=[html_output, stats_output, structure_output, status_output, plot_output],
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 1)[0] or ".")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        css=CSS,
    )
