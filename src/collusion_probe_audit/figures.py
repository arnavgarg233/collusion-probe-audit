"""The four figures of the paper, drawn from the stored records.

fig_regime_map.png (Fig. 1), fig_forest.png (Fig. 2), fig_roc_boundary.png (Fig. 3) and
fig_sandbag_layers.png (Fig. 4).
"""

import csv
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from .paths import EVIDENCE, FIGURES  # noqa: E402

warnings.filterwarnings("ignore")
TAB = str(EVIDENCE)

sns.set_theme(context="paper", style="white", font="DejaVu Sans")
plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.titleweight": "regular",
        "axes.titlepad": 8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    }
)
BLUE, CYAN, RED, ORANGE, GREEN, GREY = "#4C72B0", "#64B5CD", "#C44E52", "#DD8452", "#55A868", "#8a8a8a"
INK = "#222222"


def panel(ax, letter):
    ax.text(
        -0.015,
        1.05,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        ha="right",
        va="bottom",
    )


def ygrid(ax):
    ax.grid(axis="y", color="#cfcfcf", ls=(0, (1, 2)), lw=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


# ---------- shared AUROC-unit cells ---------------------------------------
# Executed-deception cells: (label, full-access observable-channel AUROC, controlled delta, floor95).
# Sources under results/tables/evidence/: narcbench/natural_subsat.csv and gapfill_corrected.csv,
# breadth/liars.log, breadth/b2w_result.json, breadth/sleeper_result.json and
# breadth/backdoorllm_result.json (layer 16). tests/test_figures.py checks each value against its record.
EXECUTED = [
    ("gpt-oss core", 0.861, 0.102, 0.219),
    ("Qwen3 core", 0.980, -0.118, 0.120),
    ("Liars' convincing", 0.719, 0.018, 0.119),
    ("Liars' gender", 0.997, -0.094, 0.131),
    ("sleeper-agent", 0.940, 0.060, 0.206),
    ("BackdoorLLM", 0.987, 0.013, 0.060),
    ("b2w ai_liar", 0.959, 0.009, 0.131),
    ("b2w dilution", 0.919, 0.033, 0.132),
    ("b2w insider", 0.999, 0.001, 0.075),
    ("b2w sandbagging", 0.999, -0.001, 0.059),
]
APPS = ("b2w apps_backdoor", 0.675, 0.226, 0.218)


def mistral_decode_cell(layer="24"):
    """Mistral-7B withheld-answer decode in AUROC units, all from the one fp16 run:
    x = question-text-only macro one-vs-rest AUROC (the access-fair baseline),
    y = decode AUROC minus that baseline, with its group-block permutation floor."""
    import json

    d = json.load(open(os.path.join(TAB, "sandbagging", "sandbag_access_fair.json")))
    c = d["layers"][layer]
    return ("Mistral-7B decode", d["text_auroc"], c["delta"], c["floor95"])


# ---------- (a) regime map -------------------------------------------------
def draw_regime_map(ax):
    mis = mistral_decode_cell()
    floor = float(np.median([r[3] for r in EXECUTED + [APPS, mis]]))
    ax.axhspan(-floor, floor, color=BLUE, alpha=0.09, zorder=0, lw=0)
    ax.axhline(0, color=GREY, lw=0.9, zorder=1)
    ax.scatter(
        [r[1] for r in EXECUTED],
        [r[2] for r in EXECUTED],
        s=46,
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label=f"content in observable channel ({len(EXECUTED)})",
    )
    ax.scatter(
        [APPS[1], mis[1]],
        [APPS[2], mis[2]],
        s=170,
        marker="*",
        color=RED,
        edgecolor="white",
        linewidth=0.6,
        zorder=6,
        label="content withheld (2)",
    )
    ax.annotate(
        "Mistral-7B sandbagging decode",
        (mis[1], mis[2]),
        textcoords="offset points",
        xytext=(-8, 11),
        ha="left",
        fontsize=8.5,
        color=RED,
    )
    ax.annotate(
        "code backdoor",
        (APPS[1], APPS[2]),
        textcoords="offset points",
        xytext=(0, 11),
        ha="center",
        fontsize=8.5,
        color=RED,
    )
    ax.annotate(
        "executed-deception cluster (10)",
        xy=(0.93, -0.03),
        xytext=(0.42, -0.22),
        fontsize=8.5,
        color=GREY,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, alpha=0.6),
    )
    ax.set_xlabel("full-access observable-channel AUROC")
    ax.set_ylabel("controlled white-box lift  $\\Delta$AUROC")
    ax.set_xlim(0.40, 1.03)
    ax.set_ylim(-0.31, 0.62)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.legend(loc="upper right", frameon=False, handletextpad=0.3, labelspacing=0.35, fontsize=8.5)
    sns.despine(ax=ax)


# ---------- ROC (b2w boundary view) ----------------------------------------
def draw_roc(ax, stem, title):
    fp = os.path.join(TAB, "breadth", "b2w_roc_preds.npz")
    if not os.path.exists(fp):
        ax.text(0.5, 0.5, "no b2w_roc_preds.npz", ha="center")
        return
    d = np.load(fp)
    y = d[f"{stem}__y"].astype(int)
    tp = d[f"{stem}__text"]
    ap = d[f"{stem}__act"]
    for name, pr, col, ls in [("full-access text monitor", tp, BLUE, (0, (5, 1.5))), ("activation probe", ap, RED, "-")]:
        fpr, tpr, _ = roc_curve(y, pr)
        au = roc_auc_score(y, pr)
        ax.plot(
            fpr,
            tpr,
            color=col,
            lw=2.2,
            ls=ls,
            solid_capstyle="round",
            zorder=4,
            label=f"{name}\nAUROC {au:.3f}",
        )
        for f in (0.01, 0.05):
            ax.scatter(
                [f],
                [float(np.interp(f, fpr, tpr))],
                s=34,
                color=col,
                edgecolor="white",
                linewidth=0.9,
                zorder=6,
            )
    ax.plot([0, 1], [0, 1], color=GREY, ls=(0, (4, 3)), lw=1.1, zorder=2, label="chance")
    for f in (0.01, 0.05):
        ax.axvline(f, color=GREY, ls=(0, (1, 3)), lw=0.8, zorder=1)
    ax.text(0.065, 0.36, "1% / 5% FPR", fontsize=8, color=GREY)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title, color=INK)
    ax.set_xlim(-0.008, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, handletextpad=0.5, labelspacing=0.6)
    ygrid(ax)


# ---------- (b) controlled-delta forest ------------------------------------
def draw_forest(ax, right_labels=False):
    rows = (
        [(r[0], r[2], r[3], "c") for r in EXECUTED]
        + [(APPS[0], APPS[2], APPS[3], "w"), (lambda m: (m[0], m[2], m[3], "w"))(mistral_decode_cell())]
    )[::-1]
    for i, (lab, d, fl, grp) in enumerate(rows):
        ax.plot([-fl, fl], [i, i], color="#c7ccd1", lw=1.4, solid_capstyle="butt", zorder=1)
        ax.plot([-fl, -fl], [i - 0.16, i + 0.16], color="#c7ccd1", lw=1.0, zorder=1)
        ax.plot([fl, fl], [i - 0.16, i + 0.16], color="#c7ccd1", lw=1.0, zorder=1)
        ax.scatter(
            [d], [i], s=46, color=(RED if grp == "w" else BLUE), edgecolor="white", linewidth=0.7, zorder=4
        )
    ax.axvline(0, color=GREY, lw=0.9, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    if right_labels:
        ax.yaxis.tick_right()
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("controlled white-box lift $\\Delta$AUROC   (bar = ± permutation floor)")
    ax.set_xlim(-0.33, 0.62)
    h = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=6, label="content in observable channel"),
        plt.Line2D([], [], marker="o", ls="", color=RED, ms=6, label="content withheld"),
    ]
    ax.legend(handles=h, loc="upper right", frameon=False, fontsize=8.5)
    ax.grid(axis="x", color="#e3e3e3", lw=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=right_labels, right=not right_labels)


# ---------- (d) sandbagging decode vs layer --------------------------------
def draw_sandbag(ax, metric):
    """One panel of the Mistral-7B withheld-answer decode from the single fp16 run
    (results/tables/sandbag_one_run.csv). metric='acc': 4-way accuracy, chance 0.25, with the
    visible-answer accuracy; metric='auroc': macro one-vs-rest AUROC, chance 0.5, with the
    question-text-only baseline. Both share the decoder and the bootstrap resamples."""
    import json

    rows = [
        r
        for r in csv.DictReader(open(os.path.join(TAB, "sandbagging", "sandbag_one_run.csv")))
        if r["quant"] == "fp16"
    ]
    L = [int(r["layer"]) for r in rows]
    v = [float(r[metric]) for r in rows]
    lo = [float(r[f"{metric}_lo"]) for r in rows]
    hi = [float(r[f"{metric}_hi"]) for r in rows]
    ax.fill_between(L, lo, hi, color=BLUE, alpha=0.15, zorder=1, lw=0, label="95% bootstrap CI")
    ax.plot(
        L,
        v,
        "-o",
        color=BLUE,
        lw=2.2,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
        label="activation decode",
    )
    if metric == "acc":
        ax.axhline(0.25, color=GREY, ls=(0, (4, 3)), lw=1.1, zorder=2, label="chance (0.25)")
        ax.axhline(0.0, color=RED, ls=(0, (1, 2)), lw=1.3, zorder=2, label="visible answer (0.00)")
        ax.set_ylabel("4-way decode accuracy")
        ax.set_ylim(-0.05, 1.04)
    else:
        t = json.load(open(os.path.join(TAB, "sandbagging", "sandbag_access_fair.json")))["text_auroc"]
        ax.axhline(0.5, color=GREY, ls=(0, (4, 3)), lw=1.1, zorder=2, label="chance (0.50)")
        ax.axhline(t, color=RED, ls=(0, (1, 2)), lw=1.3, zorder=2, label=f"question text only ({t:.2f})")
        ax.set_ylabel("macro one-vs-rest AUROC")
        ax.set_ylim(0.35, 1.03)
    for x, a in zip(L, v):
        ax.annotate(
            f"{a:.2f}",
            (x, a),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            color=BLUE,
        )
    ax.set_xlabel("transformer layer")
    ax.set_xticks(L)
    ax.set_xlim(14.5, 31.5)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), frameon=False, fontsize=8.5)
    ygrid(ax)


def main(out=None):
    OUT = str(out) if out else str(FIGURES)
    os.makedirs(OUT, exist_ok=True)
    # Each figure is a self-contained unit for its in-text section (single-column unless noted).

    # regime map — the information-access boundary (Sec. regime map)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    draw_regime_map(ax)
    fig.savefig(os.path.join(OUT, "fig_regime_map.png"))
    plt.close(fig)
    print("wrote fig_regime_map.png")

    # controlled-delta forest — every setting vs its floor (Sec. collusion reliability / breadth)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    draw_forest(ax)
    fig.savefig(os.path.join(OUT, "fig_forest.png"))
    plt.close(fig)
    print("wrote fig_forest.png")

    # ROC boundary (double-column, 2-panel: the one coherent pair) — operating points (Sec. detector reliability)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw=dict(wspace=0.24))
    draw_roc(a0, "sandbagging_wmdp_mmlu", "Executed deception (n=931)")
    draw_roc(a1, "apps_backdoors", "Withheld content (n=150)")
    panel(a0, "a")
    panel(a1, "b")
    fig.savefig(os.path.join(OUT, "fig_roc_boundary.png"))
    plt.close(fig)
    print("wrote fig_roc_boundary.png")

    # sandbagging layer profile — the positive control (Sec. sandbagging)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(wspace=0.24))
    draw_sandbag(a0, "acc")
    draw_sandbag(a1, "auroc")
    panel(a0, "a")
    panel(a1, "b")
    fig.savefig(os.path.join(OUT, "fig_sandbag_layers.png"))
    plt.close(fig)
    print("wrote fig_sandbag_layers.png")
    print("done.")
