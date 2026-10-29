"""Withheld-answer decoders for the sandbagging analysis.

These functions are copied without change in behaviour from the extraction script
(scripts/experiments/pod/sandbag_quantization_transfer.py) and the subspace script
(scripts/experiments/pod/defender_subspace_recovery_and_cost.py), so that the CPU recomputation
in sandbagging.py does not depend on the GPU code.
"""

from __future__ import annotations

import csv

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LETTERS = ["A", "B", "C", "D"]
COL_NOPWD = "question_prompt"
COL_PWD = "question_prompt_with_password"
COL_ANSWER = "answer"
COL_SUBJECT = "subject"
CHANCE4 = 0.25


def parse_rows(csv_path, n):
    """Return (nopwd_prompts, pwd_prompts, answers, subjects) for the first n WMDP rows."""
    rows = list(csv.DictReader(open(csv_path)))[:n]
    no, pw, ans, sub = [], [], [], []
    for r in rows:
        a_ = (r.get(COL_ANSWER) or "").strip()
        q_no = r.get(COL_NOPWD) or ""
        q_pw = r.get(COL_PWD) or r.get("question_prompt_pw") or ""
        if q_no and a_ in LETTERS:
            no.append(q_no)
            pw.append(q_pw)
            ans.append(a_)
            sub.append(r.get(COL_SUBJECT) or "?")
    return no, pw, ans, sub


def text_clf():
    """Question-text-only decoder: TF-IDF on the multiple-choice prompt, logistic regression."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True),
        LogisticRegression(solver="liblinear", C=1.0, max_iter=5000),
    )


def block_perm_labels(y, g, rng):
    """Shuffle the answer letters within each subject block."""
    y = np.asarray(y)
    g = np.asarray(g)
    out = y.copy()
    for gg in set(g):
        idx = np.where(g == gg)[0]
        out[idx] = rng.permutation(y[idx])
    return out


def safe_auroc(p, y):
    y = np.asarray(y).astype(int)
    m = ~np.isnan(p)
    return roc_auc_score(y[m], p[m]) if (m.sum() > 4 and len(set(y[m])) == 2) else float("nan")


def tpr_at_fpr(scores, y, fpr=0.01):
    """TPR at the largest ROC threshold whose empirical FPR does not exceed the target."""
    scores = np.asarray(scores, float)
    y = np.asarray(y).astype(int)
    m = ~np.isnan(scores)
    s, yy = scores[m], y[m]
    if len(set(yy)) < 2 or m.sum() < 5:
        return float("nan")
    fp, tp, _ = roc_curve(yy, s)
    ok = np.where(fp <= fpr + 1e-12)[0]
    i = ok[-1] if len(ok) else 0
    return float(tp[i])


def bootstrap_ci(scores, y, stat="auroc", fpr=0.01, n_boot=1000, seed=0):
    """Stratified bootstrap interval for AUROC or TPR at a fixed FPR."""
    scores = np.asarray(scores, float)
    y = np.asarray(y).astype(int)
    m = ~np.isnan(scores)
    s, yy = scores[m], y[m]
    point = safe_auroc(s, yy) if stat == "auroc" else tpr_at_fpr(s, yy, fpr)
    if not (point == point) or len(set(yy)) < 2:
        return dict(point=None, lo=None, hi=None, n_boot=0)
    pos = np.where(yy == 1)[0]
    neg = np.where(yy == 0)[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        bi = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        v = safe_auroc(s[bi], yy[bi]) if stat == "auroc" else tpr_at_fpr(s[bi], yy[bi], fpr)
        if v == v:
            vals.append(v)
    if not vals:
        return dict(point=round(float(point), 4), lo=None, hi=None, n_boot=0)
    return dict(
        point=round(float(point), 4),
        lo=round(float(np.percentile(vals, 2.5)), 4),
        hi=round(float(np.percentile(vals, 97.5)), 4),
        n_boot=len(vals),
    )


def binarized_oof_proba(acts_m, y_letters_m, g_m, pca=40):
    """Group out-of-fold letter probabilities; scaler and PCA are fit inside each fold.

    Returns (proba[n, n_classes], classes, hard[n]).
    """
    y = np.asarray(y_letters_m)
    g = np.asarray(g_m)
    n = len(y)
    classes = sorted(set(y))
    if len(classes) < 2 or len(set(g)) < 2:
        return None, classes, np.full(n, "", dtype=object)
    proba = np.full((n, len(classes)), np.nan)
    cidx = {c: i for i, c in enumerate(classes)}
    hard = np.full(n, "", dtype=object)
    k = max(2, min(5, len(set(g))))
    for tr, te in GroupKFold(k).split(np.zeros(n), y, g):
        if len(set(y[tr])) < 2:
            continue
        red = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(pca, len(tr) - 1, acts_m.shape[1]), random_state=0),
        ).fit(acts_m[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(red.transform(acts_m[tr]), y[tr])
        pr = clf.predict_proba(red.transform(acts_m[te]))
        for j, c in enumerate(clf.classes_):
            proba[te, cidx[c]] = pr[:, j]
        hard[te] = clf.classes_[pr.argmax(1)]
    return proba, classes, hard


def ovr_macro_auroc(proba, classes, y_letters, n_boot=0, seed=0):
    """Macro one-versus-rest AUROC over the answer letters, with an optional bootstrap interval."""
    y = np.asarray(y_letters)
    rows = np.where(~np.isnan(proba).any(1))[0]
    if len(rows) < 5:
        return dict(point=None, lo=None, hi=None, per_class={})

    def macro(idx):
        aus = []
        for j, c in enumerate(classes):
            yb = (y[idx] == c).astype(int)
            if len(set(yb)) < 2:
                continue
            aus.append(roc_auc_score(yb, proba[idx, j]))
        return float(np.mean(aus)) if aus else float("nan")

    point = macro(rows)
    per_class = {}
    for j, c in enumerate(classes):
        yb = (y[rows] == c).astype(int)
        per_class[c] = round(roc_auc_score(yb, proba[rows, j]), 4) if len(set(yb)) == 2 else None
    out = dict(
        point=round(point, 4) if point == point else None, lo=None, hi=None, per_class=per_class
    )
    if n_boot and point == point:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            bi = rng.choice(rows, len(rows), replace=True)
            v = macro(bi)
            if v == v:
                vals.append(v)
        if vals:
            out["lo"] = round(float(np.percentile(vals, 2.5)), 4)
            out["hi"] = round(float(np.percentile(vals, 97.5)), 4)
            out["n_boot"] = len(vals)
    return out


def _label_idx(y):
    order = sorted(set(y))
    m = {c: i for i, c in enumerate(order)}
    return np.array([m[c] for c in y]), order


def _project_out(X, W):
    if W is None or len(W) == 0:
        return X
    return X - (X @ W.T) @ W


def pca_ablation_curve(A, y, g, ks, n_boot=200, n_perm=20, seed=0):
    """4-way decode accuracy after projecting out the top-k principal directions.

    The directions are fit on the training fold only, in raw centred space, with GroupKFold by
    subject. For each k it also returns a cluster-bootstrap interval and a permutation floor.
    """
    yi, order = _label_idx(y)
    g = np.asarray(g)
    groups = sorted(set(g))
    n_splits = len(groups)
    rng = np.random.default_rng(seed)

    def _oof_acc(labels, k):
        preds = np.full(len(labels), -1)
        for tr, te in GroupKFold(n_splits=n_splits).split(A, labels, g):
            if len(set(labels[tr])) < 2:
                preds[te] = rng.integers(0, len(order), size=len(te))
                continue
            mu = A[tr].mean(0)
            Xtr = A[tr] - mu
            Xte = A[te] - mu
            if k > 0:
                W = (
                    PCA(n_components=min(k, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=seed)
                    .fit(Xtr)
                    .components_
                )
                Xtr = _project_out(Xtr, W)
                Xte = _project_out(Xte, W)
            clf = LogisticRegression(max_iter=2000, C=0.5, random_state=seed).fit(Xtr, labels[tr])
            preds[te] = clf.predict(Xte)
        return preds

    out = []
    for k in ks:
        preds = _oof_acc(yi, k)
        acc = float((preds == yi).mean())
        bvals = []
        for _ in range(n_boot):
            chosen = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([np.where(g == gg)[0] for gg in chosen])
            bvals.append(float((preds[idx] == yi[idx]).mean()))
        lo, hi = (float(np.percentile(bvals, 2.5)), float(np.percentile(bvals, 97.5)))
        floors = []
        for _ in range(n_perm):
            yp = rng.permutation(yi)
            fp = _oof_acc(yp, k)
            floors.append(float((fp == yp).mean()))
        floor95 = float(np.percentile(floors, 95))
        p = float(np.mean([fl >= acc for fl in floors]))
        out.append(
            dict(
                k=int(k),
                decode_acc=round(acc, 4),
                ci_lo=round(lo, 4),
                ci_hi=round(hi, 4),
                floor95=round(floor95, 4),
                perm_p=round(p, 4),
                chance=CHANCE4,
            )
        )
    return out
