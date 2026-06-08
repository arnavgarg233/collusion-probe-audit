"""Publication figures - WeakPINN / solver-forensics house style (seaborn, paper context,
muted palette, regular-weight titles, faint dotted grid, despined, compact).

figures/:
  fig_boundary.png     (a) regime map  (b) controlled-delta forest
  fig_reliability.png  (a) ROC executed -> text dominates  (b) ROC withheld -> white-box wins
  fig_main.png         2x2 graphical abstract

Run: python scripts/make_paper_figures.py
"""
import csv, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)
TAB = os.path.join(ROOT, "results", "tables")

sns.set_theme(context="paper", style="white", font="DejaVu Sans")
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.titleweight": "regular", "axes.titlepad": 8,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.9,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 150,
})
BLUE, CYAN, RED, ORANGE, GREEN, GREY = "#4C72B0", "#64B5CD", "#C44E52", "#DD8452", "#55A868", "#8a8a8a"
INK = "#222222"


def panel(ax, letter):
    ax.text(-0.015, 1.05, letter, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, ha="right", va="bottom")


def ygrid(ax):
    ax.grid(axis="y", color="#cfcfcf", ls=(0, (1, 2)), lw=0.8); ax.set_axisbelow(True)
    sns.despine(ax=ax)


# ---------- (a) regime map -------------------------------------------------
def draw_regime_map(ax):
    conf = [(0.861, 0.102), (0.980, -0.118), (0.719, 0.018),
            (0.997, -0.094), (0.940, 0.060), (0.900, 0.013), (0.959, 0.009),
            (0.919, 0.033), (0.999, 0.001), (0.999, -0.016)]   # 10 executed-deception cells
    gpwd = (0.141, 0.076)   # Gemma-pwd: withheld but NOT decodable -> confounded
    sand = [(0.228, 0.685), (0.250, 0.660), (0.122, 0.598)]   # Mistral, Gemma-cb, Mistral-instruct
    apps = (0.675, 0.226); floor = 0.11

    ax.axhspan(-floor, floor, color=BLUE, alpha=0.09, zorder=0, lw=0)
    ax.axhline(0, color=GREY, lw=0.9, zorder=1)
    ax.scatter([p[0] for p in conf] + [gpwd[0]], [p[1] for p in conf] + [gpwd[1]], s=46, color=BLUE,
               edgecolor="white", linewidth=0.7, zorder=4, label="executed / non-decodable - confounded (11)")
    ax.scatter([p[0] for p in sand] + [apps[0]], [p[1] for p in sand] + [apps[1]], s=170,
               marker="*", color=RED, edgecolor="white", linewidth=0.6, zorder=6,
               label="withheld content - white-box wins (4)")

    ax.annotate("Mistral · Gemma-cb · Mistral-instruct\nsandbag decode", (0.20, 0.66),
                textcoords="offset points", xytext=(0, 15), ha="center", fontsize=8.0, color=RED, linespacing=1.1)
    ax.annotate("b2w apps_backdoor", apps, textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=8.5, color=RED)
    ax.annotate("Gemma-pwd\n(withheld, not decodable)", gpwd, textcoords="offset points", xytext=(13, -3),
                ha="left", fontsize=7.3, color=GREY, linespacing=1.0)
    ax.annotate("executed-deception cluster (10):\nNARCBench · Liars' · sleeper · BackdoorLLM · b2w×4",
                xy=(0.905, -0.03), xytext=(0.33, -0.245), fontsize=8, color=GREY, ha="left",
                linespacing=1.25, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, alpha=0.6))
    ax.text(0.50, 0.075, "shaded: ± median floor", fontsize=8, color=GREY, va="center", style="italic")

    ax.set_xlabel("full-access visible / behaviour baseline  (AUROC)")
    ax.set_ylabel("controlled white-box lift   $\\Delta$AUROC")
    ax.set_title("The information-access boundary", color=INK)
    ax.set_xlim(0.07, 1.05); ax.set_ylim(-0.31, 0.80)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax.legend(loc="upper left", bbox_to_anchor=(0.30, 1.005), frameon=False,
              handletextpad=0.3, labelspacing=0.35, fontsize=8.5)
    sns.despine(ax=ax)


# ---------- ROC (b2w boundary view) ----------------------------------------
def draw_roc(ax, stem, title):
    fp = os.path.join(TAB, "b2w_roc_preds.npz")
    if not os.path.exists(fp):
        ax.text(0.5, 0.5, "no b2w_roc_preds.npz", ha="center"); return
    d = np.load(fp); y = d[f"{stem}__y"].astype(int); tp = d[f"{stem}__text"]; ap = d[f"{stem}__act"]
    for name, pr, col in [("full-access text monitor", tp, BLUE), ("activation probe", ap, RED)]:
        fpr, tpr, _ = roc_curve(y, pr); au = roc_auc_score(y, pr)
        ax.plot(fpr, tpr, color=col, lw=2.2, solid_capstyle="round", zorder=4,
                label=f"{name}  ·  AUROC {au:.3f}")
        for f in (0.01, 0.05):
            ax.scatter([f], [float(np.interp(f, fpr, tpr))], s=34, color=col,
                       edgecolor="white", linewidth=0.9, zorder=6)
    ax.plot([0, 1], [0, 1], color=GREY, ls=(0, (4, 3)), lw=1.1, zorder=2, label="chance")
    for f in (0.01, 0.05):
        ax.axvline(f, color=GREY, ls=(0, (1, 3)), lw=0.8, zorder=1)
    ax.text(0.065, 0.04, "1% / 5% FPR", fontsize=8, color=GREY)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(title, color=INK); ax.set_xlim(-0.008, 1.0); ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, handletextpad=0.5)
    ygrid(ax)


# ---------- (b) controlled-delta forest ------------------------------------
def draw_forest(ax, right_labels=False):
    rows = [("gpt-oss core", 0.102, 0.219, "c"), ("qwen3 core", -0.118, 0.120, "c"),
            ("BackdoorLLM", 0.013, 0.10, "c"),
            ("Liars' convincing", 0.018, 0.119, "c"), ("Liars' gender", -0.094, 0.131, "c"),
            ("sleeper-agent", 0.060, 0.20, "c"), ("b2w ai_liar", 0.009, 0.131, "c"),
            ("b2w dilution", 0.033, 0.132, "c"), ("b2w insider", 0.001, 0.075, "c"),
            ("b2w sandbagging", -0.016, 0.059, "c"), ("Gemma-pwd", 0.076, 0.082, "c"),
            ("b2w apps_backdoor", 0.226, 0.218, "w"), ("Gemma-cb decode", 0.660, 0.080, "w"),
            ("Mistral-instruct decode", 0.598, 0.080, "w"), ("Mistral decode", 0.685, 0.080, "w")][::-1]
    for i, (lab, d, fl, grp) in enumerate(rows):
        ax.plot([-fl, fl], [i, i], color="#c7ccd1", lw=1.4, solid_capstyle="butt", zorder=1)
        ax.plot([-fl, -fl], [i - 0.16, i + 0.16], color="#c7ccd1", lw=1.0, zorder=1)
        ax.plot([fl, fl], [i - 0.16, i + 0.16], color="#c7ccd1", lw=1.0, zorder=1)
        ax.scatter([d], [i], s=46, color=(RED if grp == "w" else BLUE),
                   edgecolor="white", linewidth=0.7, zorder=4)
    ax.axvline(0, color=GREY, lw=0.9, zorder=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    if right_labels:
        ax.yaxis.tick_right()
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("controlled white-box lift $\\Delta$   (bar = ± permutation floor)")
    ax.set_title("Every executed lift sits inside its floor", color=INK)
    ax.set_xlim(-0.33, 0.80)
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=6, label="executed / non-decodable - confounded"),
         plt.Line2D([], [], marker="o", ls="", color=RED, ms=6, label="withheld - white-box wins")]
    ax.legend(handles=h, loc="lower left", frameon=False, fontsize=8.5, bbox_to_anchor=(0.0, 0.0))
    ax.grid(axis="x", color="#e3e3e3", lw=0.8); ax.set_axisbelow(True)
    sns.despine(ax=ax, left=right_labels, right=not right_labels)


# ---------- (d) sandbagging decode vs layer --------------------------------
def draw_sandbag(ax):
    fp = os.path.join(TAB, "sandbag_layer_ci.csv")
    if not os.path.exists(fp):
        ax.text(0.5, 0.5, "no sandbag_layer_ci.csv", ha="center"); return
    rows = list(csv.DictReader(open(fp)))
    L = [int(r["layer"]) for r in rows]; acc = [float(r["decode_acc"]) for r in rows]
    lo = [float(r["ci_lo"]) for r in rows]; hi = [float(r["ci_hi"]) for r in rows]
    ax.fill_between(L, lo, hi, color=BLUE, alpha=0.15, zorder=1, lw=0, label="95% bootstrap CI")
    ax.plot(L, acc, "-o", color=BLUE, lw=2.2, markersize=6, markeredgecolor="white",
            markeredgewidth=1.0, zorder=4, label="activation decode of withheld answer")
    ax.axhline(0.25, color=GREY, ls=(0, (4, 3)), lw=1.1, zorder=2, label="chance (0.25)")
    ax.axhline(0.0, color=RED, ls=(0, (1, 2)), lw=1.3, zorder=2, label="visible output accuracy (0.00)")
    for x, a in zip(L, acc):
        ax.annotate(f"{a:.2f}", (x, a), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=BLUE)
    ax.set_xlabel("transformer layer"); ax.set_ylabel("decode AUROC of withheld answer")
    ax.set_title("Positive control: withheld answer decodes across late layers", color=INK)
    ax.set_xticks(L); ax.set_xlim(14.5, 31.5); ax.set_ylim(-0.05, 1.04)
    ax.legend(loc="center right", frameon=False, fontsize=8.5)
    ygrid(ax)


def main():
    # Each figure is a self-contained unit for its in-text section (single-column unless noted).

    # regime map - the information-access boundary (Sec. regime map)
    fig, ax = plt.subplots(figsize=(6.6, 5.0)); draw_regime_map(ax)
    fig.savefig(os.path.join(OUT, "fig_regime_map.png")); plt.close(fig); print("wrote fig_regime_map.png")

    # controlled-delta forest - every setting vs its floor (Sec. collusion reliability / breadth)
    fig, ax = plt.subplots(figsize=(7.0, 5.2)); draw_forest(ax)
    fig.savefig(os.path.join(OUT, "fig_forest.png")); plt.close(fig); print("wrote fig_forest.png")

    # ROC boundary (double-column, 2-panel: the one coherent pair) - operating points (Sec. detector reliability)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw=dict(wspace=0.24))
    draw_roc(a0, "sandbagging_wmdp_mmlu", "Executed deception (n=931):\ntext monitor dominates at every FPR")
    draw_roc(a1, "apps_backdoors", "Withheld content (n=150):\nonly here does the white-box probe win")
    panel(a0, "a"); panel(a1, "b")
    fig.savefig(os.path.join(OUT, "fig_roc_boundary.png")); plt.close(fig); print("wrote fig_roc_boundary.png")

    # sandbagging layer profile - the positive control (Sec. sandbagging)
    fig, ax = plt.subplots(figsize=(6.6, 4.6)); draw_sandbag(ax)
    fig.savefig(os.path.join(OUT, "fig_sandbag_layers.png")); plt.close(fig); print("wrote fig_sandbag_layers.png")
    print("done.")


if __name__ == "__main__":
    main()
