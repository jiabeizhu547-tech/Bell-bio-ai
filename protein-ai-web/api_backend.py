"""
API Backend for Protein AI landing page.
Directly calls ML inference functions (no Gradio dependency at runtime).
Generates HTML and matplotlib plots for results.
"""
import os
import sys
import tempfile
import traceback
import numpy as np

# ---- Path setup ----
_INFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference')
_INFERENCE_DIR = os.path.normpath(_INFERENCE_DIR)
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

# ---- Matplotlib ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Fix Chinese font rendering on Windows
try:
    import matplotlib.font_manager as fm
    # Try common CJK fonts available on Windows
    for font_name in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']:
        for f in fm.fontManager.ttflist:
            if font_name in f.name:
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial']
                break
        else:
            continue
        break
except Exception:
    pass

# ---- Output directory for images ----
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---- Lazy-loaded inference modules ----
_inference = None

def _get_inference():
    global _inference
    if _inference is None:
        from inference import predict_secondary_structure
        from ec_inference import predict_ec
        from mutation_inference import predict_mutations_batch
        _inference = {
            'predict_ss': predict_secondary_structure,
            'predict_ec': predict_ec,
            'predict_mutations': predict_mutations_batch,
        }
    return _inference

# ============================================================================
# Theme constants (matching the cinematic dark theme)
# ============================================================================
BG = "#08080a"
SURFACE = "rgba(255,255,255,0.03)"
BORDER = "rgba(255,255,255,0.08)"
BORDER_STRONG = "rgba(255,255,255,0.12)"
TEXT_PRIMARY = "#f5f5f5"
TEXT_SECONDARY = "rgba(255,255,255,0.55)"
TEXT_MUTED = "rgba(255,255,255,0.35)"

# Matplotlib-compatible colors (hex only)
MPL_TEXT = "#8c8c8c"
MPL_TEXT_BRIGHT = "#f5f5f5"
MPL_SURFACE = "#0d0d0f"
MPL_BORDER = "#1a1a1c"
MPL_GRID = "#111113"
MPL_DIM_WHITE = "#2a2a2c"
MPL_DIM_RED = "#4a1e1e"
MPL_PALE_RED = "#3a1818"
MPL_NEUTRAL = "#353538"

SS_COLORS = {"H": "#FF6B6B", "E": "#60A5FA", "C": "#FBBF24"}
SS_BG_COLORS = {"H": "rgba(255,107,107,0.15)", "E": "rgba(96,165,250,0.15)", "C": "rgba(251,191,36,0.15)"}
SS_NAMES = {"H": "alpha-helix", "E": "beta-sheet", "C": "coil"}

EC_COLORS = ["#EF4444", "#F59E0B", "#10B981", "#3B82F6", "#8B5CF6", "#EC4899", "#06B6D4"]
EC_ICONS = ["🔴", "🟠", "🟢", "🔵", "🟣", "🩷", "🩵"]

# ============================================================================
# HTML Builders — Secondary Structure
# ============================================================================

def _build_colored_html(seq, ss):
    chars_per_row = 60
    parts = [
        f'<div style="font-family:\'JetBrains Mono\',\'Consolas\',monospace;font-size:13px;'
        f'line-height:2.2;background:{SURFACE};border:1px solid {BORDER};border-radius:18px;'
        f'padding:20px 24px;">'
    ]
    for row_start in range(0, len(seq), chars_per_row):
        chunk_seq = seq[row_start:row_start + chars_per_row]
        chunk_ss = ss[row_start:row_start + chars_per_row]
        if row_start == 0:
            parts.append(
                '<div style="display:flex;flex-wrap:wrap;color:rgba(255,255,255,0.3);'
                'font-size:9px;font-weight:500;margin-bottom:4px;letter-spacing:0.05em;">'
            )
            for j in range(len(chunk_seq)):
                pos = j + 1
                parts.append(
                    f'<span style="min-width:19px;text-align:left;">'
                    f'{pos if pos == 1 or pos % 10 == 0 else ""}</span>'
                )
            parts.append('</div>')
        parts.append('<div style="display:flex;flex-wrap:wrap;">')
        for j, (aa, s) in enumerate(zip(chunk_seq, chunk_ss)):
            bg_color = SS_BG_COLORS.get(s, "rgba(255,255,255,0.05)")
            text_color = SS_COLORS.get(s, TEXT_MUTED)
            parts.append(
                f'<span style="background:{bg_color};color:{text_color};font-weight:700;'
                f'min-width:19px;height:24px;line-height:24px;text-align:center;'
                f'border-radius:4px;margin:1px;display:inline-block;font-size:12px;" '
                f'title="Pos {row_start + j + 1}: {aa}>{SS_NAMES.get(s, s)}">{aa}</span>'
            )
        parts.append('</div>')
        parts.append('<div style="display:flex;flex-wrap:wrap;margin-bottom:6px;">')
        for s in chunk_ss:
            parts.append(
                f'<span style="color:{SS_COLORS.get(s, TEXT_MUTED)};font-weight:700;'
                f'min-width:19px;text-align:center;font-size:10px;margin:0 1px;'
                f'display:inline-block;">{s}</span>'
            )
        parts.append('</div>')
    parts.append('</div>')
    return "".join(parts)


def _build_stats_cards(counts, pct, length, avg_conf):
    configs = [
        ("H", "alpha-helix", "Local hydrogen-bonded spiral"),
        ("E", "beta-sheet", "Parallel/antiparallel sheets"),
        ("C", "coil", "Flexible loop regions"),
    ]
    cards = []
    for ss_type, full_name, desc in configs:
        color = SS_COLORS[ss_type]
        cards.append(
            f'<div style="flex:1;min-width:160px;background:{SURFACE};'
            f'border:1px solid {BORDER};border-radius:18px;padding:18px 20px;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<span style="background:{color};color:#08080a;font-weight:800;font-size:20px;'
            f'width:42px;height:42px;border-radius:12px;display:flex;align-items:center;'
            f'justify-content:center;">{ss_type}</span>'
            f'<div><div style="font-weight:600;font-size:14px;color:{TEXT_PRIMARY};">{full_name}</div>'
            f'<div style="font-size:11px;color:{TEXT_SECONDARY};">{desc}</div></div></div>'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'border-top:1px solid {BORDER};padding-top:10px;">'
            f'<span style="font-size:11px;color:{TEXT_SECONDARY};">Residues</span>'
            f'<span style="font-weight:800;font-size:22px;color:{color};">{counts[ss_type]}</span>'
            f'<span style="font-size:11px;color:{TEXT_SECONDARY};">Ratio</span>'
            f'<span style="font-weight:800;font-size:22px;color:{color};">{pct[ss_type]}%</span>'
            f'</div></div>'
        )
    summary = (
        f'<div style="display:flex;gap:32px;margin-top:12px;background:{SURFACE};'
        f'border:1px solid {BORDER};border-radius:14px;padding:12px 24px;justify-content:center;">'
        f'<div><span style="color:{TEXT_SECONDARY};font-size:11px;">Length </span>'
        f'<span style="font-weight:700;font-size:18px;color:{TEXT_PRIMARY};">{length}</span>'
        f'<span style="color:{TEXT_MUTED};font-size:10px;"> aa</span></div>'
        f'<div><span style="color:{TEXT_SECONDARY};font-size:11px;">Avg Confidence </span>'
        f'<span style="font-weight:700;font-size:18px;color:{TEXT_PRIMARY};">{avg_conf:.3f}</span></div>'
        f'</div>'
    )
    return (
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;">'
        + "".join(cards) + '</div>' + summary
    )


def _build_ss_plot(seq, ss, confidences, counts, pct):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "text.color": MPL_TEXT, "axes.edgecolor": MPL_BORDER,
        "axes.facecolor": MPL_SURFACE, "figure.facecolor": BG,
        "axes.labelcolor": MPL_TEXT, "xtick.color": MPL_TEXT,
        "ytick.color": MPL_TEXT, "grid.color": MPL_GRID,
    })
    fig = plt.figure(figsize=(14, 6), facecolor=BG)

    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.set_facecolor(MPL_SURFACE)
    display_len = min(len(seq), 250)
    colors = [SS_COLORS[s] for s in ss[:display_len]]
    ax1.bar(range(display_len), [1] * display_len, color=colors, width=1.0, linewidth=0)
    ax1.set_xlim(-1, display_len); ax1.set_ylim(0, 1.15); ax1.set_yticks([])
    ax1.set_xlabel(f"Residue position (first {display_len} of {len(seq)})", fontweight="500")
    ax1.set_title("Per-Residue Secondary Structure", fontweight="600", fontsize=13, pad=10, color=TEXT_PRIMARY)
    legend_elements = [
        patches.Patch(facecolor=SS_COLORS["H"], label=f"H  alpha-Helix ({pct['H']}%)"),
        patches.Patch(facecolor=SS_COLORS["E"], label=f"E  beta-Sheet ({pct['E']}%)"),
        patches.Patch(facecolor=SS_COLORS["C"], label=f"C  Coil ({pct['C']}%)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", framealpha=0.9,
               facecolor=BG, edgecolor=MPL_BORDER, fontsize=9, labelcolor=MPL_TEXT)

    ax2 = fig.add_subplot(2, 2, 3)
    sizes = [counts["H"], counts["E"], counts["C"]]
    pie_colors = [SS_COLORS["H"], SS_COLORS["E"], SS_COLORS["C"]]
    wedges, _ = ax2.pie(sizes, labels=[f"H  {pct['H']}%", f"E  {pct['E']}%", f"C  {pct['C']}%"],
                        colors=pie_colors, explode=(0.03, 0.03, 0.03),
                        startangle=90, textprops={"fontsize": 11, "fontweight": "600", "color": TEXT_PRIMARY})
    for w in wedges: w.set_edgecolor(BG); w.set_linewidth(1.5)
    ax2.set_title("Structure Distribution", fontweight="600", fontsize=13, pad=10, color=TEXT_PRIMARY)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.set_facecolor(MPL_SURFACE)
    ax3.hist(confidences, bins=25, color="#818CF8", edgecolor=BG, alpha=0.85)
    ax3.axvline(x=np.mean(confidences), color="#FCA5A5", linestyle="--", linewidth=2.5,
                label=f"Mean: {np.mean(confidences):.3f}")
    ax3.set_xlabel("Confidence", fontweight="500")
    ax3.set_ylabel("Count", fontweight="500")
    ax3.set_title("Confidence Distribution", fontweight="600", fontsize=13, pad=10, color=TEXT_PRIMARY)
    ax3.legend(facecolor=BG, edgecolor=MPL_BORDER, fontsize=9, labelcolor=MPL_TEXT)
    ax3.set_xlim(0.25, 1.05); ax3.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout(pad=2)
    return fig


# ============================================================================
# HTML Builders — EC Classification
# ============================================================================

def _build_ec_html(result):
    top = result["all_results"]
    bars = []
    for i, item in enumerate(top):
        pct_val = item["probability"] * 100
        is_top = (i == 0)
        ec_color = EC_COLORS[int(item["ec_class"]) - 1]
        bars.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;'
            f'{"background:rgba(129,140,248,0.08);border-radius:14px;padding:8px 12px;" if is_top else ""}">'
            f'<span style="font-weight:600;font-size:12px;min-width:55px;color:{TEXT_PRIMARY};">'
            f'{EC_ICONS[i]} EC {item["ec_class"]}</span>'
            f'<span style="font-size:11px;min-width:130px;color:{TEXT_SECONDARY};">{item["name"]}</span>'
            f'<div style="flex:1;background:rgba(255,255,255,0.06);border-radius:8px;height:20px;overflow:hidden;">'
            f'<div style="background:{ec_color};width:{int(pct_val)}%;height:100%;'
            f'border-radius:8px;display:flex;align-items:center;justify-content:flex-end;padding-right:6px;">'
            f'<span style="color:#08080a;font-size:10px;font-weight:700;">{pct_val:.1f}%</span>'
            f'</div></div></div>'
        )
    return (
        f'<div style="background:{SURFACE};border:1px solid {BORDER_STRONG};border-radius:18px;'
        f'padding:22px 26px;">'
        f'<div style="text-align:center;margin-bottom:16px;">'
        f'<span style="font-size:48px;">{EC_ICONS[0]}</span><br>'
        f'<span style="font-weight:700;font-size:20px;color:{TEXT_PRIMARY};">'
        f'EC {result["predicted_class"]}: {result["predicted_name"]}</span><br>'
        f'<span style="font-size:12px;color:{TEXT_SECONDARY};">{result["description"]}</span><br>'
        f'<span style="font-weight:600;font-size:14px;color:#818CF8;">'
        f'Confidence: {result["confidence"]:.1%}</span>'
        f'</div>' + "".join(bars) + '</div>'
    )


def _build_ec_plot(result):
    names = [r["name"].split(" ")[0][:6] for r in result["all_results"]]
    probs = [r["probability"] * 100 for r in result["all_results"]]
    classes = [r["ec_class"] for r in result["all_results"]]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "text.color": MPL_TEXT, "axes.edgecolor": MPL_BORDER,
        "axes.facecolor": MPL_SURFACE, "figure.facecolor": BG,
        "axes.labelcolor": MPL_TEXT, "xtick.color": MPL_TEXT,
        "ytick.color": MPL_TEXT, "grid.color": MPL_GRID,
    })
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=BG)
    ax.set_facecolor(MPL_SURFACE)
    bars = ax.barh(range(7), probs, color=[EC_COLORS[int(c) - 1] for c in classes],
                   edgecolor=BG, height=0.6, linewidth=1.5)
    ax.set_yticks(range(7))
    ax.set_yticklabels([f"EC {c}: {n}" for c, n in zip(classes, names)], fontsize=11, fontweight="600")
    ax.invert_yaxis()
    ax.set_xlabel("Probability (%)", fontweight="500")
    ax.set_title("EC Enzyme Class Prediction", fontweight="600", fontsize=13, pad=10, color=TEXT_PRIMARY)
    ax.set_xlim(0, max(probs) * 1.2 + 5)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontweight="600", fontsize=10, color=TEXT_PRIMARY)
    plt.tight_layout(pad=2)
    return fig


# ============================================================================
# HTML Builders — Mutation Effect
# ============================================================================

def _build_mutation_html(batch_result, sequence):
    results = batch_result["results"]
    errors = batch_result["errors"]
    n = len(results)
    import html as _html
    seq_display = _html.escape(sequence[:40]) + ("..." if len(sequence) > 40 else "")
    pathogenic = sum(1 for r in results if "pathogenic" in r["prediction"].lower() or "致病" in r["prediction"])
    damaging = sum(1 for r in results if "damaging" in r["prediction"].lower() or "影响功能" in r["prediction"])
    benign = sum(1 for r in results if "benign" in r["prediction"].lower() or "良性" in r["prediction"])
    uncertain = n - pathogenic - damaging - benign

    summary_html = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">'
        f'<div style="flex:1;min-width:120px;background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:16px;padding:14px 18px;text-align:center;">'
        f'<div style="font-size:28px;font-weight:800;color:#FF6B6B;">{pathogenic}</div>'
        f'<div style="font-size:11px;color:{TEXT_SECONDARY};">Likely Pathogenic</div></div>'
        f'<div style="flex:1;min-width:120px;background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:16px;padding:14px 18px;text-align:center;">'
        f'<div style="font-size:28px;font-weight:800;color:#FBBF24;">{damaging}</div>'
        f'<div style="font-size:11px;color:{TEXT_SECONDARY};">Possibly Damaging</div></div>'
        f'<div style="flex:1;min-width:120px;background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:16px;padding:14px 18px;text-align:center;">'
        f'<div style="font-size:28px;font-weight:800;color:#34D399;">{benign}</div>'
        f'<div style="font-size:11px;color:{TEXT_SECONDARY};">Likely Benign</div></div>'
        f'<div style="flex:1;min-width:120px;background:{SURFACE};border:1px solid {BORDER};'
        f'border-radius:16px;padding:14px 18px;text-align:center;">'
        f'<div style="font-size:28px;font-weight:800;color:{TEXT_MUTED};">{uncertain}</div>'
        f'<div style="font-size:11px;color:{TEXT_SECONDARY};">Uncertain</div></div>'
        f'</div>'
        f'<div style="margin-bottom:14px;font-size:12px;color:{TEXT_SECONDARY};">'
        f'Sequence: <code style="color:{TEXT_PRIMARY};background:{SURFACE};padding:2px 8px;'
        f'border-radius:6px;">{seq_display}</code> | {n} mutations predicted</div>'
    )

    result_cards = []
    for r in results:
        pred = r["prediction"].lower()
        if "pathogenic" in pred or "致病" in pred:
            border, badge_color, badge_text = "#FF6B6B", "#FF6B6B", "Pathogenic"
        elif "damaging" in pred or "影响功能" in pred:
            border, badge_color, badge_text = "#FBBF24", "#FBBF24", "Damaging"
        elif "benign" in pred or "良性" in pred:
            border, badge_color, badge_text = "#34D399", "#34D399", "Benign"
        else:
            border, badge_color, badge_text = "rgba(255,255,255,0.12)", TEXT_MUTED, "Uncertain"

        score = r["score"]
        bar_pos = max(0, min(1, (score + 5) / 7))
        if score < -1:
            bar_color = "#FF6B6B"
        elif score < 0:
            bar_color = "#FBBF24"
        else:
            bar_color = "#34D399"

        result_cards.append(
            f'<div style="background:{SURFACE};border:1px solid {border};border-radius:16px;'
            f'padding:16px 20px;margin-bottom:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'flex-wrap:wrap;gap:8px;">'
            f'<div>'
            f'<span style="font-weight:700;font-size:16px;color:{TEXT_PRIMARY};">{r["variant"]}</span>'
            f'<span style="color:{TEXT_SECONDARY};font-size:11px;margin-left:8px;">'
            f'{r["wildtype_aa"]} > {r["mutant_aa"]} @ pos {r["position"]}</span>'
            f'</div>'
            f'<span style="background:{badge_color};color:#08080a;font-size:11px;font-weight:700;'
            f'padding:4px 12px;border-radius:9999px;">{badge_text}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:10px;margin-top:10px;">'
            f'<span style="font-weight:700;font-size:13px;min-width:48px;color:{TEXT_PRIMARY};">'
            f'LLR: {score:+.2f}</span>'
            f'<div style="flex:1;background:rgba(255,255,255,0.06);border-radius:6px;'
            f'height:18px;overflow:hidden;">'
            f'<div style="background:{bar_color};width:{bar_pos * 100:.0f}%;height:100%;'
            f'border-radius:6px;"></div></div>'
            f'<span style="font-size:10px;color:{TEXT_MUTED};min-width:80px;">Harmful | Beneficial</span>'
            f'</div>'
            f'<div style="display:flex;gap:22px;margin-top:8px;font-size:11px;'
            f'color:{TEXT_SECONDARY};">'
            f'<span>Structure disruption: {r["structure_disruption"]:.4f}</span>'
            f'<span>Confidence: {r["confidence"]:.0%}</span>'
            f'<span>WT logP: {r["wt_logp"]:.2f} | MT logP: {r["mt_logp"]:.2f}</span>'
            f'</div></div>'
        )

    error_html = ""
    if errors:
        error_items = "".join(
            f'<li><code style="color:{TEXT_PRIMARY};">{e["mutation"]}</code>: {e["error"]}</li>'
            for e in errors
        )
        error_html = (
            f'<div style="margin-top:10px;background:rgba(255,107,107,0.08);'
            f'border:1px solid rgba(255,107,107,0.25);border-radius:14px;padding:12px 18px;">'
            f'<span style="font-weight:600;color:#FF6B6B;">Parse Errors</span>'
            f'<ul style="margin:4px 0 0 16px;font-size:12px;color:{TEXT_SECONDARY};">'
            f'{error_items}</ul></div>'
        )

    return summary_html + "".join(result_cards) + error_html


def _build_mutation_plot(batch_result):
    results = batch_result["results"]
    if not results:
        return None

    n = len(results)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10,
        "text.color": MPL_TEXT, "axes.edgecolor": MPL_BORDER,
        "axes.facecolor": MPL_SURFACE, "figure.facecolor": BG,
        "axes.labelcolor": MPL_TEXT, "xtick.color": MPL_TEXT,
        "ytick.color": MPL_TEXT_BRIGHT, "grid.color": MPL_GRID,
    })
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(8, n * 0.5 + 4), max(4.5, n * 0.38)),
        facecolor=BG, gridspec_kw={"width_ratios": [2, 1]}
    )

    variants = [r["variant"] for r in results]
    scores = [r["score"] for r in results]

    bar_colors = []
    for s in scores:
        if s < -1: bar_colors.append("#FF6B6B")
        elif s < 0: bar_colors.append("#FBBF24")
        else: bar_colors.append("#34D399")

    ax1.set_facecolor(MPL_SURFACE)
    y_pos = range(n)
    ax1.barh(y_pos, scores, color=bar_colors, edgecolor=BG, height=0.6, linewidth=1.5)
    ax1.axvline(x=0, color=MPL_DIM_WHITE, linewidth=1.5, linestyle="-")
    ax1.axvline(x=-0.5, color=MPL_DIM_RED, linewidth=1, linestyle="--",
                label="Pathogenic threshold (-0.5)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(variants, fontweight="600", fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mutation Effect Score (LLR)", fontweight="500")
    ax1.set_title("Mutation Effect Scores", fontweight="600", fontsize=13, pad=10, color=TEXT_PRIMARY)
    ax1.legend(fontsize=9, loc="lower left", facecolor=BG, edgecolor=MPL_BORDER,
               labelcolor=MPL_TEXT)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    for i, s in enumerate(scores):
        offset = 0.15
        ax1.text(s + (offset if s >= 0 else -offset), i,
                 f"{s:+.2f}", va="center",
                 ha="left" if s >= 0 else "right",
                 fontweight="600", fontsize=9, color=TEXT_PRIMARY)

    ax2.set_facecolor(MPL_SURFACE)
    disruptions = [r["structure_disruption"] for r in results]

    scatter_colors = []
    for r in results:
        pred = r["prediction"].lower()
        if "pathogenic" in pred or "致病" in pred:
            scatter_colors.append("#FF6B6B")
        elif "damaging" in pred or "影响功能" in pred:
            scatter_colors.append("#FBBF24")
        elif "benign" in pred or "良性" in pred:
            scatter_colors.append("#34D399")
        else:
            scatter_colors.append(MPL_NEUTRAL)

    ax2.scatter(scores, disruptions, c=scatter_colors, s=120, edgecolors=BG,
                linewidth=1.5, zorder=5, alpha=0.85)
    ax2.axhline(y=0.08, color=MPL_DIM_RED, linewidth=1, linestyle="--",
                label="Disruption threshold (0.08)")
    ax2.axvline(x=-0.5, color=MPL_PALE_RED, linewidth=1, linestyle="--")

    for i, v in enumerate(variants):
        ax2.annotate(v, (scores[i], disruptions[i]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=8, fontweight="600", color=MPL_TEXT, alpha=0.8)

    ax2.set_xlabel("LLR Score", fontweight="500")
    ax2.set_ylabel("Structure Disruption", fontweight="500")
    ax2.set_title("Score vs. Structure Disruption", fontweight="600", fontsize=13, pad=10,
                  color=TEXT_PRIMARY)
    ax2.legend(fontsize=9, facecolor=BG, edgecolor=MPL_BORDER, labelcolor=MPL_TEXT)
    ax2.grid(alpha=0.3, linestyle="--")

    plt.tight_layout(pad=2)
    return fig


def _save_plot(fig, prefix):
    """Save matplotlib figure to static dir, return URL path."""
    import uuid
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    fig.savefig(filepath, format="png", dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    return f"/static/{filename}"


# ============================================================================
# Public API functions
# ============================================================================

def parse_fasta(text: str):
    """Parse FASTA text into list of (name, sequence) tuples."""
    results = []
    current_name = ""
    current_seq = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_seq:
                results.append((current_name, "".join(current_seq)))
            current_name = line[1:].split()[0]
            current_seq = []
        elif line:
            current_seq.append(line)
    if current_seq:
        results.append((current_name, "".join(current_seq)))
    return results if results else [("", text.strip().replace("\n", ""))]


def predict_ss_batch(fasta_text: str):
    """Batch SS prediction from FASTA text."""
    from inference import predict_secondary_structure
    sequences = parse_fasta(fasta_text)
    results_list = []
    for name, seq in sequences:
        try:
            r = predict_secondary_structure(seq)
            results_list.append({"name": name, "seq": seq[:60], "counts": r["counts"], "structure": r["structure"]})
        except Exception as e:
            results_list.append({"name": name, "seq": seq[:60], "error": str(e)})
    total = len(results_list)
    ok = sum(1 for r in results_list if "error" not in r)
    return {"results": results_list, "total": total, "ok": ok, "fail": total - ok}


def predict_ss(sequence: str):
    """Run secondary structure prediction. Returns dict with HTML and plot URL."""
    inf = _get_inference()
    try:
        result = inf['predict_ss'](sequence)
    except ValueError as e:
        return {"error": f"Input error: {e}"}
    except FileNotFoundError as e:
        return {"error": f"Model not found: {e}"}
    except Exception as e:
        return {"error": f"Prediction failed: {e}\n```\n{traceback.format_exc()}\n```"}

    seq = result["sequence"]
    ss = result["structure"]
    pct = result["percentages"]
    counts = result["counts"]
    confidences = [r["confidence"] for r in result["per_residue"]]
    avg_conf = np.mean(confidences)

    colored_html = _build_colored_html(seq, ss)
    stats_html = _build_stats_cards(counts, pct, result["length"], avg_conf)
    structure_text = f"Sequence: {seq}\nStructure: {ss}\nLegend: H=alpha-helix  E=beta-sheet  C=coil"

    fig = _build_ss_plot(seq, ss, confidences, counts, pct)
    plot_url = _save_plot(fig, "ss")

    return {
        "colored_html": colored_html,
        "stats_html": stats_html,
        "structure_text": structure_text,
        "plot_url": plot_url,
    }


def predict_ec(sequence: str):
    """Run EC enzyme classification. Returns dict with HTML and plot URL."""
    inf = _get_inference()
    try:
        result = inf['predict_ec'](sequence)
    except Exception as e:
        return {"error": f"Prediction failed: {e}\n```\n{traceback.format_exc()}\n```"}

    html = _build_ec_html(result)
    fig = _build_ec_plot(result)
    plot_url = _save_plot(fig, "ec")

    return {"html": html, "plot_url": plot_url}


def predict_mutation(sequence: str, mutations_str: str):
    """Run mutation effect prediction. Returns dict with HTML and plot URL."""
    inf = _get_inference()
    mut_list = [m.strip() for m in mutations_str.split(",") if m.strip()]
    if not mut_list:
        return {"error": "Please enter at least one mutation (e.g. R175H or A67T, G128R)"}

    try:
        batch_result = inf['predict_mutations'](sequence, mut_list)
    except Exception as e:
        return {"error": f"Prediction failed: {e}\n```\n{traceback.format_exc()}\n```"}

    html = _build_mutation_html(batch_result, sequence)
    fig = _build_mutation_plot(batch_result)
    plot_url = _save_plot(fig, "mut") if fig else None

    return {"html": html, "plot_url": plot_url}
