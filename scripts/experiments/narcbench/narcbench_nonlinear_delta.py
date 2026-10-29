"""NONLINEAR controlled delta.

Repeat the text-controlled delta (narcbench_textctrl.delta_test) but swap the LINEAR
activation probe (LogisticRegression) for a NONLINEAR one:
  - 'mlp' : sklearn MLPClassifier(hidden_layer_sizes=(64,))
  - 'svc' : sklearn RBF SVC(probability=True)
applied to the activation PCA(30) features. The visible TEXT baseline is left exactly
as-is (TF-IDF 1-2 gram + L2-logistic, pooled OOF), so the delta is still:

  delta = AUROC(text + activation) - AUROC(text-only)

and any gain over the linear delta must come from nonlinearity in the activation channel.

Decision rule (control discipline, non-negotiable):
  A method "works" ONLY if act/ta BEATS the matching VISIBLE baseline AND delta clears the
  permutation floor (delta > delta_floor95, delta_p small). We report delta vs the linear
  tc.delta_test delta on the SAME data.
"""
from __future__ import annotations
# --- path bootstrap: make sibling src/ subpackages importable as flat modules ---
import os as _os, sys as _sys
_SRCROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('', 'common', 'synthetic', 'narcbench', 'audit', 'exploratory'):
    _pp = _os.path.join(_SRCROOT, _sub)
    if _os.path.isdir(_pp) and _pp not in _sys.path:
        _sys.path.insert(0, _pp)
# --- end bootstrap ---
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score


# ---- probes ---------------------------------------------------------------
def _text_pipe():
    # IDENTICAL to tc._text_pipe -> the visible TEXT baseline is unchanged.
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))


def _act_clf(kind):
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, alpha=1e-3,
                             early_stopping=False, random_state=0)
    if kind == "svc":
        # CalibratedClassifierCV(ensemble=False) gives calibrated RBF-SVC probabilities
        # without the deprecated SVC(probability=True) internal CV (faster, no warning).
        return CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=0),
            method="sigmoid", ensemble=False, cv=3)
    if kind == "linear":
        return LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
    raise ValueError(kind)


def _oof(texts, A, y, g, act_pca, kind, n_splits=5):
    """Pooled OOF probs: text-only (linear, fixed), activation-only (nonlinear), and
    stacked text+activation. The stack head stays LINEAR-logistic over [PCA act feats,
    text_prob] so it can combine channels; the *activation channel* nonlinearity comes
    from the nonlinear `act_clf` whose OOF prob is one of the stacked features."""
    n = len(y)
    text_oof = np.full(n, np.nan); act_oof = np.full(n, np.nan); ta_oof = np.full(n, np.nan)
    k = max(2, min(n_splits, len(set(g))))
    for tr, te in GroupKFold(k).split(np.zeros(n), y, g):
        if len(set(y[tr])) < 2:
            continue
        # --- TEXT channel (unchanged) ---
        tp = _text_pipe().fit([texts[i] for i in tr], y[tr])
        text_te = tp.predict_proba([texts[i] for i in te])[:, 1]
        text_oof[te] = text_te
        gtr = g[tr]; ktr = max(2, min(n_splits, len(set(gtr))))
        text_tr = cross_val_predict(_text_pipe(), [texts[i] for i in tr], y[tr],
                                    groups=gtr, cv=GroupKFold(ktr), method="predict_proba")[:, 1]
        # --- ACTIVATION channel (nonlinear) ---
        red = make_pipeline(StandardScaler(),
                            PCA(n_components=min(act_pca, len(tr) - 1, A.shape[1]),
                                random_state=0)).fit(A[tr])
        Atr, Ate = red.transform(A[tr]), red.transform(A[te])
        act_oof[te] = _act_clf(kind).fit(Atr, y[tr]).predict_proba(Ate)[:, 1]
        # leakage-safe inner OOF nonlinear-activation prob for the stack feature
        act_tr = cross_val_predict(_act_clf(kind), Atr, y[tr], groups=gtr,
                                   cv=GroupKFold(ktr), method="predict_proba")[:, 1]
        # --- STACK: linear head over [nonlinear act prob, text prob] ---
        Mtr = np.column_stack([act_tr, text_tr]); Mte = np.column_stack([act_oof[te], text_te])
        ta_oof[te] = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced") \
            .fit(Mtr, y[tr]).predict_proba(Mte)[:, 1]
    return text_oof, act_oof, ta_oof


def delta_test(texts, A, y, g, act_pca=30, kind="mlp", n_perm=40, seed=0):
    to, ao, tao = _oof(texts, A, y, g, act_pca, kind)
    m = ~np.isnan(tao)
    text, act, ta = (roc_auc_score(y[m], to[m]), roc_auc_score(y[m], ao[m]),
                     roc_auc_score(y[m], tao[m]))
    delta = ta - text
    rng = np.random.default_rng(seed); perm = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        t2, a2, ta2 = _oof(texts, A, yp, g, act_pca, kind)
        mm = ~np.isnan(ta2)
        try:
            perm.append(roc_auc_score(yp[mm], ta2[mm]) - roc_auc_score(yp[mm], t2[mm]))
        except Exception:
            pass
    perm = np.array(perm) if perm else np.array([0.0])
    return {"kind": kind, "n": int(m.sum()), "pos": int(y[m].sum()),
            "text": text, "act": act, "ta": ta, "delta": delta,
            "delta_floor95": float(np.percentile(perm, 95)),
            "delta_p": float(np.mean(perm >= delta))}
