"""Confirm the borderline 'CHECK' cells collapse to confounded under a FAIR text baseline.

The fast pass inflated some deltas because empty transcripts crippled the text baseline.
Here we (a) restrict to NON-EMPTY transcripts, (b) add the n_tokens behaviour feature to the
visible baseline, and (c) compute a permutation-null floor. Single process, thread-pinned.
"""
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "common")):
    if _p not in sys.path: sys.path.insert(0, _p)
import narcbench_loader as nb
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

ROOT = "/tmp/narc/slice"
RNG = np.random.RandomState(0)


def g(rows, k, d=None):
    return [r.get(k, d) if isinstance(r, dict) else getattr(r, k, d) for r in rows]


def oof_auroc(Xnum, y, grp):
    y = np.asarray(y).astype(int); grp = np.asarray(grp)
    ng = len(set(grp)); k = max(2, min(5, ng))
    if len(set(y)) < 2 or ng < 2: return float("nan")
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    p = cross_val_predict(clf, Xnum, y, cv=GroupKFold(k), groups=grp, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def text_oof(texts, y, grp):
    y = np.asarray(y).astype(int); grp = np.asarray(grp)
    k = max(2, min(5, len(set(grp))))
    # liblinear converges on sparse high-dim text; cap features. FAIR baseline = full visible record.
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True),
                        LogisticRegression(solver="liblinear", C=1.0, max_iter=5000))
    return cross_val_predict(clf, texts, y, cv=GroupKFold(k), groups=grp, method="predict_proba")[:, 1]


def cell(model, tier, contrast, n_perm=50):
    sl = nb.load_slice(ROOT, model, tier)
    rows = nb.agent_table(sl, channel="public", behavioral_filter=False)
    role = np.array(g(rows, "role")); mode = np.array(g(rows, "mode"))
    vec = np.array([np.asarray(v, float) for v in g(rows, "vec")])
    # FAIR visible baseline = the full visible record the agent saw + produced (context + own output),
    # because the activation reads that visible context. Excludes the hidden prompt.
    _ctx = g(rows, "context_text", ""); _vis = g(rows, "visible_text", "")
    txt = np.array([f"{c or ''} {v or ''}".strip() for c, v in zip(_ctx, _vis)], dtype=object)
    ntok = np.array([float(x or 0) for x in g(rows, "n_tokens", 0)]).reshape(-1, 1)
    grp = np.array([str(s) for s in (g(rows, "scenario_id") or g(rows, "run_id"))])

    if contrast == "colluder_vs_honest":
        m = np.ones(len(rows), bool); y = (role == "colluder").astype(int)
    else:  # emergent: honest agents, collusion vs control
        m = role != "colluder"; y = (mode == "collusion").astype(int)
    # restrict to NON-EMPTY transcripts (the fair text baseline)
    nonempty = np.array([len(str(t).split()) >= 3 for t in txt])
    m = m & nonempty
    y, V, T, N, G = y[m], vec[m], txt[m], ntok[m], grp[m]
    if len(set(y)) < 2 or m.sum() < 20:
        return dict(model=model, tier=tier, contrast=contrast, n=int(m.sum()), note="too few non-empty")

    # visible baseline = OOF text pred + n_tokens ; full = visible + activation PCA
    tp = text_oof(list(T), y, G).reshape(-1, 1)
    Vpca = PCA(min(30, V.shape[1], max(2, V.shape[0]-1))).fit_transform(StandardScaler().fit_transform(V))
    vis = oof_auroc(np.column_stack([tp, N]), y, G)
    full = oof_auroc(np.column_stack([tp, N, Vpca]), y, G)
    delta = full - vis
    # permutation floor: shuffle y within nothing (global), recompute delta
    floors = []
    for _ in range(n_perm):
        yp = RNG.permutation(y)
        tpp = text_oof(list(T), yp, G).reshape(-1, 1)
        v0 = oof_auroc(np.column_stack([tpp, N]), yp, G)
        f0 = oof_auroc(np.column_stack([tpp, N, Vpca]), yp, G)
        if v0 == v0 and f0 == f0: floors.append(f0 - v0)
    floor95 = float(np.percentile(floors, 95)) if floors else float("nan")
    p = float(np.mean([fl >= delta for fl in floors])) if floors else float("nan")
    verdict = "CONFOUNDED" if (delta <= floor95 or delta < 0.10) else "signal>visible(!)"
    return dict(model=model, tier=tier, contrast=contrast, n=int(m.sum()),
                visible_auroc=round(vis, 3), full_auroc=round(full, 3),
                delta=round(delta, 3), floor95=round(floor95, 3), p=round(p, 3), verdict=verdict)


CELLS = [("gpt_oss_20b", "core", "emergent"),
         ("qwen3_32b", "core", "colluder_vs_honest"),
         ("qwen3_32b", "core", "emergent")]

if __name__ == "__main__":
    for m, t, c in CELLS:
        try:
            r = cell(m, t, c)
        except Exception as e:
            r = dict(model=m, tier=t, contrast=c, err=str(e))
        print(r, flush=True)
