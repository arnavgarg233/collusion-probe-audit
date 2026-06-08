"""Reproduce NARCBench's 5 scenario-level probing techniques (Rose, Cullen, Abdelnabi, Torr,
Kaplowitz, Schroeder de Witt; arXiv:2604.01151) and run the VISIBLE-BEHAVIOUR baseline they
omitted. Task: classify a run as collusion-mode (1) vs control-mode (0). Sample = run; CV group
= scenario_id (collusion+control of the same scenario stay together). The deception direction is
their per-agent linear probe (colluder vs honest), CROSS-FITTED per fold (trained on train runs
only). Their only published baseline is a text LLM-judge; we add votes / bets / length.

Techniques: Activation Alignment (mean pairwise cosine), Peak Suspicion (max score),
Deception Split (mean top-2 - mean rest), Sorted Concatenation (sort by score, concat, PCA),
Asymmetry Probe (var/range/split/mean of scores).
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
import sys, argparse; sys.path.insert(0, "src")
import numpy as np
from itertools import combinations
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import narcbench_loader as nb
ROOT = "/tmp/narc/slice"


def _cos(a, b):
    na, nbb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nbb)) if na > 0 and nbb > 0 else 0.0


def load_runs(model, tier):
    sl = nb.load_slice(ROOT, model, tier)
    rows = nb.agent_table(sl, channel="public", behavioral_filter=False)
    runs = {}
    for r in rows:
        runs.setdefault(r["run_id"], []).append(r)
    meta = {}
    for rid, ags in runs.items():
        cfg = sl["configs"].get(rid) or {}
        res = sl["results"].get(rid) or {}
        meta[rid] = {
            "mode": ags[0]["mode"], "scenario_id": ags[0]["scenario_id"],
            "fair": cfg.get("fair_option"),
            "bp_seat": res.get("per_seat_bp_correlation") or {},
            "transcript": nb._run_public_text(res),
        }
    return sl, runs, meta


def deception_dir(train_runs, runs):
    """Their per-agent probe: logistic colluder(1) vs honest(0) on TRAIN collusion-run agents."""
    X, y = [], []
    for rid in train_runs:
        if runs[rid][0]["mode"] != "collusion":
            continue
        for a in runs[rid]:
            X.append(a["vec"]); y.append(1 if a["role"] == "colluder" else 0)
    if len(set(y)) < 2:
        return None, None
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    return sc, lr.coef_[0]


def scores(ags, sc, d):
    Xs = sc.transform([a["vec"] for a in ags])
    return Xs @ d


def tech_features(ags, s):
    vecs = [a["vec"] for a in ags]
    align = np.mean([_cos(vecs[i], vecs[j]) for i, j in combinations(range(len(vecs)), 2)]) if len(vecs) > 1 else 0.0
    ss = np.sort(s)[::-1]
    split = (ss[:2].mean() - ss[2:].mean()) if len(ss) > 2 else (ss[0] - ss[-1])
    return {"align": align, "peak": float(s.max()), "split": float(split),
            "asym": [float(np.var(s)), float(s.max() - s.min()), float(split), float(s.mean())]}


def behav_features(rid, ags, meta):
    fair = meta[rid]["fair"]
    nonfair = sum(1 for a in ags if a["agent_vote"] is not None and a["agent_vote"] != fair) if fair else 0
    bp = list(meta[rid]["bp_seat"].values())
    bp_max = max([abs(x) for x in bp], default=0.0)
    toks = [a["n_tokens"] for a in ags]
    return [nonfair, float(np.var([1 if (a["agent_vote"] or "") else 0 for a in ags])),
            bp_max, float(np.mean(bp)) if bp else 0.0,
            float(np.mean(toks)), float(np.var(toks))]


def run(model, tier, n_splits=5):
    sl, runs, meta = load_runs(model, tier)
    rids = [r for r in runs if meta[r]["mode"] in ("collusion", "control")]
    y = np.array([1 if meta[r]["mode"] == "collusion" else 0 for r in rids])
    g = np.array([meta[r]["scenario_id"] for r in rids])
    if len(set(y)) < 2:
        print(f"{model}/{tier}: single-class scenario labels, skip"); return
    k = max(2, min(n_splits, len(set(g))))

    techs = ["align", "peak", "split", "asym", "sortedconcat"]
    oof = {t: np.full(len(rids), np.nan) for t in techs}
    oof_behav = np.full(len(rids), np.nan)
    idx = {r: i for i, r in enumerate(rids)}

    for tr, te in GroupKFold(k).split(np.zeros(len(rids)), y, g):
        tr_rids = [rids[i] for i in tr]; te_rids = [rids[i] for i in te]
        sc, d = deception_dir(tr_rids, runs)
        if d is None:
            continue
        # per-run features
        feat_tr, feat_te = {}, {}
        for rid in tr_rids + te_rids:
            s = scores(runs[rid], sc, d)
            feat = tech_features(runs[rid], s)
            srt = np.argsort(s)[::-1][:4]
            vv = np.concatenate([runs[rid][j]["vec"] for j in srt] +
                                [np.zeros(runs[rid][0]["vec"].shape)] * max(0, 4 - len(srt)))
            (feat_tr if rid in tr_rids else feat_te)[rid] = (feat, vv)
        ytr = y[tr]
        # scalar techniques: 1-feature logistic; multi: logistic
        for t in ["align", "peak", "split"]:
            Xtr = np.array([[feat_tr[r][0][t]] for r in tr_rids]); Xte = np.array([[feat_te[r][0][t]] for r in te_rids])
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
            for j, r in enumerate(te_rids): oof[t][idx[r]] = m.predict_proba(Xte[j:j+1])[0, 1]
        Xtr = np.array([feat_tr[r][0]["asym"] for r in tr_rids]); Xte = np.array([feat_te[r][0]["asym"] for r in te_rids])
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(StandardScaler().fit_transform(Xtr), ytr)
        scA = StandardScaler().fit(Xtr)
        for j, r in enumerate(te_rids): oof["asym"][idx[r]] = m.predict_proba(scA.transform(Xte[j:j+1]))[0, 1]
        # sorted concat + PCA
        Vtr = np.array([feat_tr[r][1] for r in tr_rids]); Vte = np.array([feat_te[r][1] for r in te_rids])
        red = make_pca(Vtr, ytr)
        m = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(red.transform(Vtr), ytr)
        for j, r in enumerate(te_rids): oof["sortedconcat"][idx[r]] = m.predict_proba(red.transform(Vte[j:j+1]))[0, 1]
        # BEHAVIOURAL baseline (no activations)
        Btr = np.array([behav_features(r, runs[r], meta) for r in tr_rids]); Bte = np.array([behav_features(r, runs[r], meta) for r in te_rids])
        scB = StandardScaler().fit(Btr)
        mb = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scB.transform(Btr), ytr)
        for j, r in enumerate(te_rids): oof_behav[idx[r]] = mb.predict_proba(scB.transform(Bte[j:j+1]))[0, 1]

    def auc(p):
        m = ~np.isnan(p); return roc_auc_score(y[m], p[m]) if len(set(y[m])) > 1 else float("nan")
    # text baseline
    texts = [meta[r]["transcript"] for r in rids]
    txt = _text_auc(texts, y, g, k)
    print(f"\n== {model}/{tier}  (scenario classification: collusion vs control; n={len(rids)} runs, {len(set(g))} scenarios) ==")
    print(f"  THEIR techniques:  align={auc(oof['align']):.3f}  peak={auc(oof['peak']):.3f}  "
          f"split={auc(oof['split']):.3f}  asym={auc(oof['asym']):.3f}  sortedConcat={auc(oof['sortedconcat']):.3f}")
    print(f"  BASELINES:         BEHAVIOUR(votes/bets/length)={auc(oof_behav):.3f}   text-transcript={txt:.3f}")
    print(f"  -> rebuttal holds if BEHAVIOUR ~= their best technique (activations not needed)")


def make_pca(X, y):
    n = min(50, X.shape[0] - 1, X.shape[1])
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), PCA(n_components=max(2, n), random_state=0)).fit(X)


def _text_auc(texts, y, g, k):
    from sklearn.pipeline import make_pipeline
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(k).split(np.zeros(len(y)), y, g):
        if len(set(y[tr])) < 2: continue
        try:
            m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                              LogisticRegression(max_iter=2000, class_weight="balanced")).fit([texts[i] for i in tr], y[tr])
            oof[te] = m.predict_proba([texts[i] for i in te])[:, 1]
        except Exception: pass
    m = ~np.isnan(oof)
    return roc_auc_score(y[m], oof[m]) if m.sum() and len(set(y[m])) > 1 else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt_oss_20b")
    ap.add_argument("--tiers", nargs="+", default=["core", "stego"])
    a = ap.parse_args()
    for t in a.tiers:
        try: run(a.model, t)
        except Exception as e: print(f"{a.model}/{t} ERROR {e}")
