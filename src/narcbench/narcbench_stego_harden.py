"""Stego confound hardening for the one live lead: gpt_oss/stego colluder delta=+0.167.

The borderline GO used a table-talk-only text baseline. But blackjack BETS are visible,
and the covert channel IS the bet pattern. So the decisive test is whether activation adds
anything beyond the FULL visible record = table talk + bet sequences. Plus the usual battery
(length, de-saturation, scenario grouping, per-seat bp_corr action baseline).
"""
# --- path bootstrap: make sibling src/ subpackages importable as flat modules ---
import os as _os, sys as _sys
_SRCROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('', 'common', 'synthetic', 'narcbench', 'audit', 'exploratory'):
    _pp = _os.path.join(_SRCROOT, _sub)
    if _os.path.isdir(_pp) and _pp not in _sys.path:
        _sys.path.insert(0, _pp)
# --- end bootstrap ---
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import sys; sys.path.insert(0, "src")
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import narcbench_loader as nb, stats
ROOT = "/tmp/narc/slice"


def _bet_feats(bets):
    b = np.array(bets, dtype=float) if bets else np.array([0.0])
    mn, mx = b.min(), b.max()
    return [b.mean(), b.std(), mn, mx, (mx / mn if mn > 0 else mx),
            (b[len(b) // 2:].mean() / (b[:max(1, len(b) // 2)].mean() + 1e-9))]


def _text_pipe():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))


def _logit():
    return LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")


def _oof(texts, C, A, y, g, act_pca=30, n_splits=5):
    """OOF probs: text-only, visible(text+controls C), full(text+C+activation), act-only."""
    n = len(y); k = max(2, min(n_splits, len(set(g))))
    TO, VIS, FULL, AO = (np.full(n, np.nan) for _ in range(4))
    for tr, te in GroupKFold(k).split(np.zeros(n), y, g):
        if len(set(y[tr])) < 2:
            continue
        tp = _text_pipe().fit([texts[i] for i in tr], y[tr])
        t_te = tp.predict_proba([texts[i] for i in te])[:, 1]; TO[te] = t_te
        gtr = g[tr]; ktr = max(2, min(n_splits, len(set(gtr))))
        t_tr = cross_val_predict(_text_pipe(), [texts[i] for i in tr], y[tr],
                                 groups=gtr, cv=GroupKFold(ktr), method="predict_proba")[:, 1]
        red = make_pipeline(StandardScaler(), PCA(min(act_pca, len(tr) - 1, A.shape[1]),
                            random_state=0)).fit(A[tr])
        Atr, Ate = red.transform(A[tr]), red.transform(A[te])
        Csc = StandardScaler().fit(C[tr]); Ctr, Cte = Csc.transform(C[tr]), Csc.transform(C[te])
        Vtr = np.column_stack([t_tr, Ctr]); Vte = np.column_stack([t_te, Cte])
        VIS[te] = _logit().fit(Vtr, y[tr]).predict_proba(Vte)[:, 1]
        FULL[te] = _logit().fit(np.column_stack([Vtr, Atr]), y[tr]).predict_proba(
            np.column_stack([Vte, Ate]))[:, 1]
        AO[te] = _logit().fit(Atr, y[tr]).predict_proba(Ate)[:, 1]
    return TO, VIS, FULL, AO


def _auc(y, p):
    m = ~np.isnan(p); return roc_auc_score(y[m], p[m])


def _delta_with_floor(texts, C, A, y, g, n_perm=60):
    TO, VIS, FULL, AO = _oof(texts, C, A, y, g)
    d = _auc(y, FULL) - _auc(y, VIS)
    rng = np.random.default_rng(0); perm = []
    for _ in range(n_perm):
        yp = stats.group_block_permute(y, g, rng)   # group-aware null (block-permute if label is group-constant)
        _, v2, f2, _ = _oof(texts, C, A, yp, g)
        try: perm.append(_auc(yp, f2) - _auc(yp, v2))
        except Exception: pass
    perm = np.array(perm) if perm else np.array([0.0])
    return (_auc(y, TO), _auc(y, VIS), _auc(y, FULL), _auc(y, AO), d,
            float(np.percentile(perm, 95)), float(np.mean(perm >= d)))


def run():
    sl = nb.load_slice(ROOT, "gpt_oss_20b", "stego")
    rows = nb.agent_table(sl, channel="public", behavioral_filter=False)
    sub = rows
    y = np.array([1 if r["role"] == "colluder" else 0 for r in sub])
    g = np.array([r["run_id"] for r in sub]); gs = np.array([r["scenario_id"] for r in sub])
    # colluder contrast grouped by run_id: label varies WITHIN run while context_text is
    # run-constant -> degenerate within-run text baseline. Use each agent's OWN visible_text
    # (the table talk an overt-text monitor sees for THAT seat). Bets are added separately as C.
    texts = [r["visible_text"] for r in sub]
    A = np.array([r["vec"] for r in sub])
    # bets + per-seat bp_corr per agent (from results)
    C, bp = [], []
    for r in sub:
        res = sl["results"].get(r["run_id"]) or {}
        bets = (res.get("all_bets") or {}).get(r["agent"])
        C.append(_bet_feats(bets))
        bp.append([(res.get("per_seat_bp_correlation") or {}).get(r["agent"], 0.0)])
    C = np.array(C); bp = np.array(bp); ntok = np.array([[r["n_tokens"]] for r in sub])
    print(f"stego colluder n={len(y)} colluders={int(sum(y))}")

    # standalone baselines
    print(f"BASE text-only={_baseline_text(texts,y,g):.3f} "
          f"act_pca50={stats.numeric_group_auroc(A,y,g,pca_dims=50):.3f} "
          f"act_pca10={stats.numeric_group_auroc(A,y,g,pca_dims=10):.3f} "
          f"act_pca5={stats.numeric_group_auroc(A,y,g,pca_dims=5):.3f}")
    print(f"BASE bets-only={stats.numeric_group_auroc(C,y,g,pca_dims=0):.3f} "
          f"perseat_bp_corr={stats.numeric_group_auroc(bp,y,g,pca_dims=0):.3f} "
          f"n_tokens={stats.numeric_group_auroc(ntok,y,g,pca_dims=0):.3f}")

    # delta beyond TEXT only (reproduce the 0.167)
    to, vis, full, ao, d, fl, p = _delta_with_floor(texts, np.zeros((len(y), 1)), A, y, g)
    print(f"DELTA beyond text-only      | text={to:.3f} vis={vis:.3f} full={full:.3f} "
          f"delta={d:+.3f} floor95={fl:.3f} p={p:.3f}")
    # DECISIVE: delta beyond TEXT + BETS
    to, vis, full, ao, d, fl, p = _delta_with_floor(texts, C, A, y, g)
    verdict = "GO" if (d > fl and d >= 0.10) else "CONFOUNDED" if d <= fl else "PARTIAL"
    print(f"DELTA beyond text+BETS      | vis(text+bets)={vis:.3f} full={full:.3f} "
          f"delta={d:+.3f} floor95={fl:.3f} p={p:.3f} -> {verdict}")
    # scenario-grouped version of the decisive test
    to, vis, full, ao, d, fl, p = _delta_with_floor(texts, C, A, y, gs)
    print(f"DELTA beyond text+BETS [scenario-grouped] | vis={vis:.3f} full={full:.3f} "
          f"delta={d:+.3f} floor95={fl:.3f} p={p:.3f}")
    print("DONE")


def _baseline_text(texts, y, g):
    from narcbench_textctrl import _text_pipe as tp
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(max(2, min(5, len(set(g))))).split(np.zeros(len(y)), y, g):
        if len(set(y[tr])) < 2: continue
        e = _text_pipe().fit([texts[i] for i in tr], y[tr])
        oof[te] = e.predict_proba([texts[i] for i in te])[:, 1]
    return _auc(y, oof)


if __name__ == "__main__":
    run()
