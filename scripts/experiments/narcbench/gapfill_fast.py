"""Controlled, single-process gap-fillers (no permutation explosion).

Computes pooled-OOF AUROC for text / activation / text+activation and the delta, for the
off-gpt_oss text baselines (core + transfer) and a PCA-dim sweep on gpt_oss core-emergent.
Fast: no permutation floors (delta magnitude tells the story; floor only matters if delta>0.1).
Writes results/tables/gapfill.csv. Thread-pinned by the caller (OMP_NUM_THREADS=1).
"""
import os, sys, csv as _csv
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

ROOT = "/tmp/narc/slice"


def pooled_oof_auroc(X, y, g, kind, pca_dim=30):
    """pooled out-of-fold AUROC. kind: 'num' (X=array) or 'text' (X=list of str)."""
    y = np.asarray(y).astype(int)
    g = np.asarray(g)
    ng = len(set(g))
    k = max(2, min(5, ng))
    if len(set(y)) < 2 or ng < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=k)
    if kind == "text":
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                            LogisticRegression(max_iter=2000, C=1.0))
        Xin = X
    else:
        Xs = StandardScaler().fit_transform(X)
        d = min(pca_dim, Xs.shape[1], max(2, Xs.shape[0] - 1))
        Xin = PCA(d).fit_transform(Xs)
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    try:
        proba = cross_val_predict(clf, Xin, y, cv=gkf, groups=g, method="predict_proba")[:, 1]
        return roc_auc_score(y, proba)
    except Exception as e:
        print("  auroc err:", e); return float("nan")


def stacked_auroc(texts, X_act, y, g, pca_dim=30):
    """pooled-OOF AUROC of [text_pred, activation_PCA] meta-model (leakage-safe via OOF text pred)."""
    y = np.asarray(y).astype(int); g = np.asarray(g)
    ng = len(set(g)); k = max(2, min(5, ng))
    if len(set(y)) < 2 or ng < 2: return float("nan")
    gkf = GroupKFold(n_splits=k)
    from sklearn.pipeline import make_pipeline
    tclf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000))
    try:
        tpred = cross_val_predict(tclf, texts, y, cv=gkf, groups=g, method="predict_proba")[:, 1]
    except Exception as e:
        print("  text oof err:", e); return float("nan")
    Xs = StandardScaler().fit_transform(X_act)
    d = min(pca_dim, Xs.shape[1], max(2, Xs.shape[0] - 1))
    Xact = PCA(d).fit_transform(Xs)
    Xstack = np.column_stack([tpred, Xact])
    return pooled_oof_auroc(Xstack, y, g, "num", pca_dim=Xstack.shape[1])


def get(rows, key, default=None):
    return [r.get(key, default) if isinstance(r, dict) else getattr(r, key, default) for r in rows]


def contrasts(model, tier):
    """yield (name, mask, label) for colluder-vs-honest and emergent honest-collusion-vs-control."""
    sl = nb.load_slice(ROOT, model, tier)
    rows = nb.agent_table(sl, channel="public", behavioral_filter=False)
    role = np.array(get(rows, "role"))
    mode = np.array(get(rows, "mode"))
    vec = np.array([np.asarray(v, float) for v in get(rows, "vec")])
    txt = [(t or "") for t in get(rows, "visible_text", "")]
    grp = np.array([str(s) for s in (get(rows, "scenario_id") or get(rows, "run_id"))])
    out = []
    # colluder vs honest
    y1 = (role == "colluder").astype(int)
    if y1.sum() and (1 - y1).sum():
        out.append(("colluder_vs_honest", np.ones(len(rows), bool), y1, vec, txt, grp))
    # emergent: among honest agents, collusion-mode vs control-mode
    hon = role != "colluder"
    if hon.sum() > 4:
        y2 = (mode[hon] == "collusion").astype(int)
        if y2.sum() and (1 - y2).sum():
            out.append(("emergent_honest_coll_vs_ctrl", hon, y2,
                        vec[hon], [txt[i] for i in np.where(hon)[0]], grp[hon]))
    return out


def main():
    out_rows = []
    MODELS = ["gpt_oss_20b", "qwen3_32b", "deepseek_r1_32b", "llama_3_1_70b"]
    for model in MODELS:
        for tier in ["core", "transfer"]:
            try:
                cs = contrasts(model, tier)
            except Exception as e:
                print(f"{model}/{tier}: load err {e}"); continue
            for name, _m, y, vec, txt, grp in cs:
                n = len(y)
                a = pooled_oof_auroc(vec, y, grp, "num")
                t = pooled_oof_auroc(txt, y, grp, "text")
                ta = stacked_auroc(txt, vec, y, grp)
                delta = (ta - t) if (ta == ta and t == t) else float("nan")
                verdict = ("confounded" if (delta != delta or delta <= 0.10) else "CHECK_floor")
                print(f"{model:16s} {tier:8s} {name:28s} n={n:3d} act={a:.3f} text={t:.3f} t+a={ta:.3f} delta={delta:+.3f} -> {verdict}", flush=True)
                out_rows.append(dict(model=model, tier=tier, contrast=name, n=n,
                                     act_auroc=round(a, 3), text_auroc=round(t, 3),
                                     text_plus_act=round(ta, 3), delta=round(delta, 3), verdict=verdict))

    # PCA-dim sweep on gpt_oss core emergent (the crashed cell)
    print("\n--- PCA de-saturation: gpt_oss_20b core emergent ---", flush=True)
    try:
        cs = contrasts("gpt_oss_20b", "core")
        em = [c for c in cs if c[0].startswith("emergent")]
        if em:
            _, _m, y, vec, txt, grp = em[0]
            for d in [5, 10, 25, 50]:
                ta = stacked_auroc(txt, vec, y, grp, pca_dim=d)
                t = pooled_oof_auroc(txt, y, grp, "text")
                delta = ta - t
                print(f"  pca={d:3d}  text={t:.3f} t+a={ta:.3f} delta={delta:+.3f}", flush=True)
                out_rows.append(dict(model="gpt_oss_20b", tier="core", contrast=f"emergent_pca{d}",
                                     n=len(y), act_auroc="", text_auroc=round(t, 3),
                                     text_plus_act=round(ta, 3), delta=round(delta, 3),
                                     verdict=("confounded" if delta <= 0.10 else "CHECK_floor")))
    except Exception as e:
        print("  pca sweep err:", e)

    od = "results/tables"; os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "gapfill.csv")
    with open(fp, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["model", "tier", "contrast", "n", "act_auroc",
                                           "text_auroc", "text_plus_act", "delta", "verdict"])
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {fp} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
