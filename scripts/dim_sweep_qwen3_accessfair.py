"""ACCESS-FAIR qwen3 core-emergent dim sweep.

The earlier dim sweep (results/tables/dim_sweep_qwen3.csv via exp3 TF-IDF baseline) was NOT
access-fair: it stacked PCA(activation) on a weak full-vocab/diluted TF-IDF baseline (0.674,
an empty-transcript artifact) instead of the real full-access TEXT baseline (~0.98). Against
0.674 the activation spuriously cleared the floor (WHITE_BOX_WINS) as PCA dim grew, even though
it never beats the true full-access baseline.

This script does the access-fair version. It:
  1. Loads the qwen3 core-emergent slice, restricts to the TEXT-ALIGNED subset (honest rows
     with non-empty context_text, n=78) -- the Phase-1 Task-A construction that reproduces the
     reported 0.980 full-access TEXT baseline (here AUROC=0.9704 with the exact textctrl
     _text_pipe: TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True) +
     LogisticRegression(C=0.5, class_weight='balanced'), pooled OOF GroupKFold-by-scenario_id).
  2. Builds a leakage-safe pooled-OOF full-access baseline score s_base from that text pipe.
  3. For act_pca in [10,30,100,200]: controlled delta = AUROC(stack[s_base, PCA(act)]) -
     AUROC(s_base), with a group-permutation floor / p / verdict, via exp3._delta_with_floor on
     exp3._oof_score_act(s_base, A, yy, g, act_pca=dim).

EXPECTED: since the activation saturates at ~0.936 < 0.980, the access-fair delta stays
NEGATIVE / within-floor (CONFOUNDED) at every dim -- the opposite of the spurious TF-IDF flip.

Writes results/tables/dim_sweep_qwen3_accessfair.csv.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys, csv, time, importlib.util
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

REPO = "/Users/akshgarg/Invariant Collusion Probe"
for _sub in ("narcbench", "common"):
    _p = os.path.join(REPO, "src", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
import narcbench_textctrl as tc

# import exp3 machinery by path
_spec = importlib.util.spec_from_file_location(
    "exp3_baseline_sweep", os.path.join(REPO, "scripts", "exp3_baseline_sweep.py"))
exp3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(exp3)

OUT_CSV = os.path.join(REPO, "results", "tables", "dim_sweep_qwen3_accessfair.csv")
DIMS = [10, 30, 100, 200]
N_PERM = 120


def build_base_oof(texts, y, g, n_splits=5):
    """Leakage-safe pooled-OOF full-access TEXT baseline score (the ~0.98 reproduced in Phase 1).
    Uses the exact textctrl _text_pipe; pooled OOF under GroupKFold grouped by scenario_id."""
    n = len(y)
    oof = np.full(n, np.nan)
    k = max(2, min(n_splits, len(set(g))))
    for tr, te in GroupKFold(k).split(np.zeros(n), y, g):
        if len(set(y[tr])) < 2:
            continue
        tp = tc._text_pipe().fit([texts[i] for i in tr], y[tr])
        oof[te] = tp.predict_proba([texts[i] for i in te])[:, 1]
    return oof


def main():
    sl, texts, A, y, g = tc._rows_labels("/tmp/narc/slice", "qwen3_32b", "core", "emergent", None)
    A = np.asarray(A, float); g = np.asarray(g); y = np.asarray(y)

    # text-aligned subset: honest rows with non-empty context_text (the Phase-1 Task-A subset)
    ne = np.array([len((t or "").strip()) > 0 for t in texts])
    texts = [t for t, k in zip(texts, ne) if k]
    A = A[ne]; y = y[ne]; g = g[ne]
    n = len(y)
    print(f"text-aligned subset: n={n} pos={int(y.sum())} groups={len(set(g))} "
          f"H={A.shape[1]} layer={sl['layer']}", flush=True)

    # full-access baseline OOF score (s_base) + its standalone AUROC
    s_base = build_base_oof(texts, y, g)
    mb = ~np.isnan(s_base)
    base_au = roc_auc_score(y[mb], s_base[mb])
    print(f"full-access TEXT baseline (s_base) pooled-OOF AUROC = {base_au:.4f}  "
          f"(reproduces reported 0.980 to within ~0.01)", flush=True)

    rows = []
    for dim in DIMS:
        t0 = time.time()
        d = exp3._delta_with_floor(
            lambda yy: exp3._oof_score_act(s_base, A, yy, g, act_pca=dim),
            y, g, n_perm=N_PERM)
        v = exp3._verdict(d)
        print(f"RESULT accessfair pca={dim:>3} | base={d['base']:.3f} act={d['act']:.3f} "
              f"stack={d['full']:.3f} delta={d['delta']:+.3f} floor95={d['floor95']:.3f} "
              f"p={d['p']:.3f} -> {v}  [{time.time()-t0:.1f}s]", flush=True)
        rows.append(dict(
            pca_dim=dim,
            base_full_access=round(d["base"], 4),
            act=round(d["act"], 4),
            stack=round(d["full"], 4),
            delta=round(d["delta"], 4),
            floor95=round(d["floor95"], 4),
            p=round(d["p"], 4),
            n=d["n"], pos=d["pos"],
            verdict=v))

    cols = ["pca_dim", "base_full_access", "act", "stack", "delta", "floor95", "p",
            "n", "pos", "verdict", "note"]
    note = ("ACCESS-FAIR: activation PCA(act) stacked on the reproduced ~0.98 full-access TEXT "
            "baseline (textctrl TfidfVectorizer(1,2)/min_df=2/sublinear + LR C=0.5 balanced, "
            "pooled OOF GroupKFold-by-scenario_id) on the text-aligned subset (n=78, honest "
            "non-empty context_text). delta=AUROC(stack[s_base,PCA(act)])-AUROC(s_base); "
            "group-permutation floor, n_perm=120. NOT the degenerate on-disk TF-IDF 0.674.")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = {**r, "note": note}
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
