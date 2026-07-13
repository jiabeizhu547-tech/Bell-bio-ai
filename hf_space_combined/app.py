"""
🧬 Protein AI — 蛋白质序列智能分析
双功能：二级结构预测 + EC 酶分类
Gradio Web 应用 — 部署至 HuggingFace Spaces
"""

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io, tempfile

# ============================================================================
# 配色
# ============================================================================
SS_COLORS = {"H": "#DC2626", "E": "#2563EB", "C": "#D97706"}
SS_BG_COLORS = {"H": "#FEE2E2", "E": "#DBEAFE", "C": "#FEF3C7"}
SS_NAMES = {"H": "α-螺旋", "E": "β-折叠", "C": "无规卷曲"}

EC_COLORS = ["#EF4444", "#F59E0B", "#10B981", "#3B82F6", "#8B5CF6", "#EC4899", "#06B6D4"]
EC_ICONS = ["🔴", "🟠", "🟢", "🔵", "🟣", "🩷", "🩵"]


# ============================================================================
# Tab 1: 二级结构预测
# ============================================================================
def predict_ss(sequence: str):
    if not sequence or not sequence.strip():
        return None, "", "", None

    try:
        from inference import predict_secondary_structure
        result = predict_secondary_structure(sequence)
    except Exception as e:
        import traceback
        return None, f"**❌ {e}**\n```\n{traceback.format_exc()}\n```", "", None

    seq = result["sequence"]
    ss = result["structure"]
    pct = result["percentages"]
    counts = result["counts"]
    confidences = [r["confidence"] for r in result["per_residue"]]
    avg_conf = np.mean(confidences)

    colored_html = _build_colored_html(seq, ss)
    stats_html = _build_stats_cards(counts, pct, result["length"], avg_conf)
    structure_text = f"```\n序列:  {seq}\n结构:  {ss}\n图例:  H=α螺旋  E=β折叠  C=卷曲\n```"

    fig = _build_ss_plot(seq, ss, confidences, counts, pct)
    tmp_path = tempfile.mktemp(suffix=".png")
    fig.savefig(tmp_path, format="png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    return colored_html, stats_html, structure_text, tmp_path


def _build_colored_html(seq, ss):
    chars_per_row = 60
    parts = [
        '<div style="font-family: \'JetBrains Mono\', \'Consolas\', monospace; font-size: 14px; '
        'line-height: 2.1; background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; '
        'padding: 18px 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">'
    ]
    for row_start in range(0, len(seq), chars_per_row):
        chunk_seq = seq[row_start:row_start + chars_per_row]
        chunk_ss = ss[row_start:row_start + chars_per_row]
        # position ruler
        if row_start == 0:
            parts.append('<div style="display: flex; flex-wrap: wrap; color: #9ca3af; font-size: 9px; font-weight: 600; margin-bottom: 2px;">')
            for j in range(len(chunk_seq)):
                pos = j + 1
                parts.append(f'<span style="min-width: 19px; text-align: left;">{pos if pos == 1 or pos % 10 == 0 else ""}</span>')
            parts.append('</div>')
        # amino acids
        parts.append('<div style="display: flex; flex-wrap: wrap;">')
        for j, (aa, s) in enumerate(zip(chunk_seq, chunk_ss)):
            bg, fg = SS_BG_COLORS.get(s, "#f3f4f6"), SS_COLORS.get(s, "#6b7280")
            parts.append(
                f'<span style="background:{bg};color:{fg};font-weight:700;min-width:19px;height:24px;'
                f'line-height:24px;text-align:center;border-radius:3px;margin:1px;display:inline-block;'
                f'font-size:13px;" title="Pos {row_start + j + 1}: {aa}→{SS_NAMES.get(s, s)}">{aa}</span>'
            )
        parts.append('</div>')
        # structure labels
        parts.append('<div style="display: flex; flex-wrap: wrap; margin-bottom: 6px;">')
        for s in chunk_ss:
            parts.append(f'<span style="color:{SS_COLORS.get(s, "#6b7280")};font-weight:800;min-width:19px;text-align:center;font-size:10px;margin:0 1px;display:inline-block;">{s}</span>')
        parts.append('</div>')
    parts.append('</div>')
    return "".join(parts)


def _build_stats_cards(counts, pct, length, avg_conf):
    configs = [
        ("H", "α-螺旋 Alpha-helix", "局部氢键维持的稳定螺旋结构"),
        ("E", "β-折叠 Beta-sheet", "平行/反平行排列的片层结构"),
        ("C", "无规卷曲 Coil", "连接螺旋和折叠的柔性环区"),
    ]
    cards = []
    for ss_type, full_name, desc in configs:
        color, bg = SS_COLORS[ss_type], SS_BG_COLORS[ss_type]
        cards.append(
            f'<div style="flex:1;min-width:160px;background:#fff;border:1px solid #e5e7eb;'
            f'border-radius:10px;padding:14px 18px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="background:{color};color:#fff;font-weight:800;font-size:22px;'
            f'width:40px;height:40px;border-radius:8px;display:flex;align-items:center;justify-content:center;">{ss_type}</span>'
            f'<div><div style="font-weight:700;font-size:15px;color:#111827;">{full_name}</div>'
            f'<div style="font-size:12px;color:#6b7280;">{desc}</div></div></div>'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'border-top:1px solid #f3f4f6;padding-top:8px;">'
            f'<span style="font-size:12px;color:#6b7280;">残基</span>'
            f'<span style="font-weight:800;font-size:20px;color:{color};">{counts[ss_type]}</span>'
            f'<span style="font-size:12px;color:#6b7280;">占比</span>'
            f'<span style="font-weight:800;font-size:20px;color:{color};">{pct[ss_type]}%</span>'
            f'</div></div>'
        )
    summary = (
        f'<div style="display:flex;gap:28px;margin-top:10px;background:#f9fafb;'
        f'border:1px solid #e5e7eb;border-radius:10px;padding:10px 20px;justify-content:center;">'
        f'<div><span style="color:#6b7280;font-size:12px;">序列长度 </span>'
        f'<span style="font-weight:800;font-size:18px;color:#111827;">{length}</span>'
        f'<span style="color:#9ca3af;font-size:11px;"> aa</span></div>'
        f'<div><span style="color:#6b7280;font-size:12px;">平均置信度 </span>'
        f'<span style="font-weight:800;font-size:18px;color:#111827;">{avg_conf:.3f}</span></div>'
        f'</div>'
    )
    return '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;">' + "".join(cards) + '</div>' + summary


def _build_ss_plot(seq, ss, confidences, counts, pct):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": "#374151",
                         "axes.edgecolor": "#d1d5db", "axes.facecolor": "#fafafa"})
    fig = plt.figure(figsize=(14, 6), facecolor="white")

    # bar chart
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.set_facecolor("#fafafa")
    display_len = min(len(seq), 250)
    colors = [SS_COLORS[s] for s in ss[:display_len]]
    ax1.bar(range(display_len), [1] * display_len, color=colors, width=1.0, linewidth=0)
    ax1.set_xlim(-1, display_len); ax1.set_ylim(0, 1.15); ax1.set_yticks([])
    ax1.set_xlabel(f"Residue position (first {display_len} of {len(seq)})", fontweight="bold")
    ax1.set_title("Per-Residue Secondary Structure", fontweight="bold", fontsize=13, pad=10)
    legend_elements = [
        patches.Patch(facecolor=SS_COLORS["H"], label=f"H  α-Helix ({pct['H']}%)"),
        patches.Patch(facecolor=SS_COLORS["E"], label=f"E  β-Sheet ({pct['E']}%)"),
        patches.Patch(facecolor=SS_COLORS["C"], label=f"C  Coil ({pct['C']}%)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", framealpha=0.9, facecolor="white", fontsize=9)

    # pie chart
    ax2 = fig.add_subplot(2, 2, 3)
    labels = [f"H  {pct['H']}%", f"E  {pct['E']}%", f"C  {pct['C']}%"]
    sizes = [counts["H"], counts["E"], counts["C"]]
    pie_colors = [SS_COLORS["H"], SS_COLORS["E"], SS_COLORS["C"]]
    wedges, _ = ax2.pie(sizes, labels=labels, colors=pie_colors, explode=(0.03, 0.03, 0.03),
                        startangle=90, textprops={"fontsize": 11, "fontweight": "bold", "color": "#111827"})
    for w in wedges: w.set_edgecolor("white"); w.set_linewidth(1.5)
    ax2.set_title("Structure Distribution", fontweight="bold", fontsize=13, pad=10)

    # confidence histogram
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_facecolor("#fafafa")
    ax3.hist(confidences, bins=25, color="#6366f1", edgecolor="white", alpha=0.85)
    ax3.axvline(x=np.mean(confidences), color="#DC2626", linestyle="--", linewidth=2.5,
                label=f"Mean: {np.mean(confidences):.3f}")
    ax3.set_xlabel("Confidence", fontweight="bold"); ax3.set_ylabel("Count", fontweight="bold")
    ax3.set_title("Confidence Distribution", fontweight="bold", fontsize=13, pad=10)
    ax3.legend(facecolor="white", fontsize=9); ax3.set_xlim(0.25, 1.05); ax3.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout(pad=2)
    return fig


# ============================================================================
# Tab 2: EC 酶分类
# ============================================================================
def predict_ec_fn(sequence: str):
    if not sequence or not sequence.strip():
        return None, "", ""

    try:
        from ec_inference import predict_ec
        result = predict_ec(sequence)
    except Exception as e:
        import traceback
        return None, f"**❌ {e}**\n```\n{traceback.format_exc()}\n```", ""

    # Build result HTML
    html = _build_ec_html(result)
    # Chart
    fig_path = _build_ec_plot(result)
    return html, "", fig_path


def _build_ec_html(result):
    prob = result["probabilities"]
    top = result["all_results"]

    bars = []
    for i, item in enumerate(top):
        pct_val = item["probability"] * 100
        bar_width = int(pct_val)
        is_top = (i == 0)
        bar_html = (
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;'
            f'{"background:#eef2ff;border-radius:8px;padding:6px 10px;" if is_top else ""}">'
            f'<span style="font-weight:700;font-size:14px;min-width:60px;color:#374151;">{EC_ICONS[i]} EC {item["ec_class"]}</span>'
            f'<span style="font-size:13px;min-width:160px;color:#4b5563;">{item["name"]}</span>'
            f'<div style="flex:1;background:#e5e7eb;border-radius:6px;height:22px;overflow:hidden;">'
            f'<div style="background:{EC_COLORS[int(item["ec_class"]) - 1]};width:{bar_width}%;height:100%;'
            f'border-radius:6px;display:flex;align-items:center;justify-content:flex-end;padding-right:6px;">'
            f'<span style="color:#fff;font-size:11px;font-weight:700;">{pct_val:.1f}%</span>'
            f'</div></div></div>'
        )
        bars.append(bar_html)

    return (
        f'<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:20px 24px;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.05);">'
        f'<div style="text-align:center;margin-bottom:16px;">'
        f'<span style="font-size:48px;">{EC_ICONS[0]}</span><br>'
        f'<span style="font-weight:800;font-size:22px;color:#111827;">EC {result["predicted_class"]}: '
        f'{result["predicted_name"]}</span><br>'
        f'<span style="font-size:14px;color:#6b7280;">{result["description"]}</span><br>'
        f'<span style="font-weight:700;font-size:15px;color:#6366f1;">置信度: {result["confidence"]:.1%}</span>'
        f'</div>'
        + "".join(bars) +
        '</div>'
    )


def _build_ec_plot(result):
    """EC 预测水平条形图。"""
    names = [r["name"].split(" ")[0][:6] for r in result["all_results"]]
    probs = [r["probability"] * 100 for r in result["all_results"]]
    classes = [r["ec_class"] for r in result["all_results"]]
    colors = [EC_COLORS[int(c) - 1] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    ax.set_facecolor("#fafafa")
    bars = ax.barh(range(7), probs, color=colors, edgecolor="white", height=0.6, linewidth=1.5)
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"EC {c}: {n}" for c, n in zip(classes, names)], fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Probability (%)", fontweight="bold")
    ax.set_title("EC Enzyme Class Prediction", fontweight="bold", fontsize=14, pad=12)
    ax.set_xlim(0, max(probs) * 1.2 + 5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontweight="bold", fontsize=10, color="#374151")

    plt.tight_layout(pad=2)
    tmp = tempfile.mktemp(suffix=".png")
    fig.savefig(tmp, format="png", dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return tmp


# ============================================================================
# CSS 样式
# ============================================================================
CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
body, .gradio-container { background: #f8f9fa !important; color: #111827 !important; }
h1 { font-size: 1.8rem !important; font-weight: 800 !important; }
h3 { font-weight: 600 !important; color: #374151 !important; }
label { font-weight: 600 !important; color: #374151 !important; font-size: 14px !important; }
textarea { font-family: 'JetBrains Mono', 'Consolas', monospace !important; font-size: 14px !important;
    background: #fff !important; border: 1.5px solid #d1d5db !important; border-radius: 8px !important; padding: 12px !important; }
textarea:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important; }
button.primary { background: #6366f1 !important; border: none !important; font-weight: 700 !important;
    font-size: 15px !important; border-radius: 8px !important; }
button.primary:hover { background: #4f46e5 !important; }
.tab-nav button { font-weight: 600 !important; font-size: 15px !important; }
footer { display: none !important; }
"""

# ============================================================================
# 界面
# ============================================================================
with gr.Blocks(title="🧬 Protein AI — 蛋白质智能分析", css=CSS) as demo:
    gr.Markdown(
        """
        # 🧬 Protein AI — 蛋白质序列智能分析
        ### 深度学习驱动的蛋白质二级结构预测 & 酶功能分类 | ESM-2 | Q3 89.8% | EC 87.3%
        """
    )

    with gr.Tabs():
        # ============ TAB 1: 二级结构 ============
        with gr.TabItem("🧬 二级结构预测"):
            with gr.Row():
                with gr.Column(scale=3):
                    input_ss = gr.Textbox(
                        label="氨基酸序列",
                        placeholder="例如: MKVLILACLVALALACTVQAKTENPKKT",
                        lines=4,
                    )
                    with gr.Row():
                        btn_ss = gr.Button("🔬 开始预测", variant="primary", scale=3)
                        btn_clear_ss = gr.Button("清空", scale=1)
                with gr.Column(scale=2):
                    gr.Markdown("""
                    #### 📖 使用说明
                    - 粘贴**氨基酸序列**，单字母大写
                    - 支持 **20 种标准氨基酸**
                    - 最多 **256 个残基**
                    - 结果：每残基结构标签 + 可视化
                    """)

            gr.Examples(
                examples=[
                    ["MKVLILACLVALALACTVQAKTENPKKT"],
                    ["ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"],
                    ["MVKVGINGFGRIGRLVTRAAFNSGKVDIVAIND"],
                ],
                inputs=input_ss,
                label="📋 示例序列",
            )

            gr.Markdown("---")
            gr.Markdown("### 预测结果")

            html_ss = gr.HTML()
            stats_ss = gr.HTML()
            struct_ss = gr.Markdown()
            plot_ss = gr.Image(label="📊 可视化分析", type="filepath")

            btn_ss.click(fn=predict_ss, inputs=[input_ss],
                         outputs=[html_ss, stats_ss, struct_ss, plot_ss])
            input_ss.submit(fn=predict_ss, inputs=[input_ss],
                            outputs=[html_ss, stats_ss, struct_ss, plot_ss])
            btn_clear_ss.click(fn=lambda: (None, "", "", None),
                               inputs=[], outputs=[html_ss, stats_ss, struct_ss, plot_ss])

        # ============ TAB 2: EC 分类 ============
        with gr.TabItem("🧪 EC 酶分类"):
            with gr.Row():
                with gr.Column(scale=3):
                    input_ec = gr.Textbox(
                        label="蛋白质序列",
                        placeholder="例如: MSKVQVTGSVLK...（输入任意酶蛋白序列）",
                        lines=4,
                    )
                    with gr.Row():
                        btn_ec = gr.Button("🧪 预测酶类别", variant="primary", scale=3)
                        btn_clear_ec = gr.Button("清空", scale=1)
                with gr.Column(scale=2):
                    gr.Markdown("""
                    #### 📖 使用说明
                    - 输入**酶蛋白氨基酸序列**
                    - 模型识别 **7 大 EC 类别**
                    - 基于 ESM-2 特征提取 + MLP
                    - 训练集：**33,331 条** UniProt 真实酶序列
                    - 测试准确率：**87.3%**
                    """)

            gr.Examples(
                examples=[
                    # Oxidoreductase examples
                    ["MSKVQVTGSVLKAAAVDAVAAAGYPVEITGDLKRLGFKGVFIEKGEWTSYDNKQAGS"],
                    # Transferase example fragment
                    ["MKTKLLLTLISVLVALALVITTAQAKTENPKKTEGKVKATKE"],
                    # Hydrolase example
                    ["MKVLILACLVALALACTVQAKTENPKKTEGKVKATKETVK"],
                ],
                inputs=input_ec,
                label="📋 示例序列",
            )

            gr.Markdown("---")
            gr.Markdown("### 分类结果")

            html_ec = gr.HTML()
            status_ec = gr.Markdown()
            plot_ec = gr.Image(label="📊 概率分布", type="filepath")

            btn_ec.click(fn=predict_ec_fn, inputs=[input_ec],
                         outputs=[html_ec, status_ec, plot_ec])
            input_ec.submit(fn=predict_ec_fn, inputs=[input_ec],
                            outputs=[html_ec, status_ec, plot_ec])
            btn_clear_ec.click(fn=lambda: (None, "", None),
                               inputs=[], outputs=[html_ec, status_ec, plot_ec])

    # Footer
    gr.Markdown("---")
    gr.Markdown(
        '<div style="text-align:center;color:#9ca3af;font-size:13px;">'
        '🧬 Protein AI | ESM-2 + CNN-BiLSTM Ensemble | '
        '<a href="https://github.com/jiabeizhu547-tech/Bell-bio-ai" target="_blank">GitHub</a> | '
        '© 2026 Bell Zhu</div>'
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
