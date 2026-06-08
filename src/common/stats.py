"""Shared, leak-aware statistics: group-CV AUROC for numeric and text features,
plus the label-permutation null. Everything else imports from here so the probe,
the controls, and the text judge use identical CV machinery.
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
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import config


def _numeric_pipeline(pca_dims: int, n_features: int, min_train: int, C: float):
    steps = [StandardScaler()]
    eff = min(pca_dims, n_features, max(1, min_train - 1))
    if pca_dims and pca_dims > 0 and eff >= 2 and eff < n_features:
        steps.append(PCA(n_components=eff, random_state=config.SEED))
    steps.append(LogisticRegression(C=C, max_iter=2000))
    return make_pipeline(*steps)


def _folds(y, groups, n_splits):
    n_groups = len(set(groups))
    k = max(2, min(n_splits, n_groups))
    return GroupKFold(n_splits=k).split(np.zeros(len(y)), y, groups)


def numeric_group_auroc(X, y, groups, pca_dims=config.PCA_DIMS, C=config.LOGREG_C,
                        n_splits=config.CV_SPLITS) -> float:
    """Mean over folds of test AUROC, GroupKFold on `groups`, PCA+L2-logistic.
    PCA is fit inside each fold (no leakage). Set pca_dims<=0 for plain L2-logistic.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_groups = len(set(groups))
    k = max(2, min(n_splits, n_groups))
    min_train = int(len(y) * (k - 1) / k)
    pipe = _numeric_pipeline(pca_dims, X.shape[1], min_train, C)
    aurocs = []
    for tr, te in _folds(y, groups, n_splits):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        est = clone(pipe).fit(X[tr], y[tr])
        p = est.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y[te], p))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def text_group_auroc(texts, y, groups, C=config.LOGREG_C,
                     n_splits=config.CV_SPLITS) -> float:
    """Same CV protocol on raw transcript text via TF-IDF (1-2 grams) + L2-logistic."""
    texts = list(texts)
    y = np.asarray(y)
    groups = np.asarray(groups)
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=C, max_iter=2000),
    )
    aurocs = []
    for tr, te in _folds(y, groups, n_splits):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        est = clone(pipe).fit([texts[i] for i in tr], y[tr])
        p = est.predict_proba([texts[i] for i in te])[:, 1]
        aurocs.append(roc_auc_score(y[te], p))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def _label_constant_within_groups(y, groups) -> bool:
    """True iff every group has a single label value. When labels are constant within a
    CV group (e.g. emergent: label = run mode, grouped by scenario/run), the only valid
    null permutes the PER-GROUP label, not individual rows -- row-level shuffling breaks
    the block structure and inflates the apparent label/activation separability, making
    the floor anti-conservative."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    for gv in set(groups.tolist()):
        if len(set(y[groups == gv].tolist())) > 1:
            return False
    return True


def group_block_permute(y, groups, rng):
    """Group-aware label permutation. If the label is constant within each group, permute
    the labels ACROSS groups (one label per group, then broadcast back to rows) so the
    permuted labels keep the same block structure. Otherwise fall back to a plain row-level
    permutation (the label genuinely varies within group -> row shuffling is the valid null).
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if not _label_constant_within_groups(y, groups):
        return rng.permutation(y)
    uniq = list(dict.fromkeys(groups.tolist()))
    glab = {gv: y[groups == gv][0] for gv in uniq}        # one label per group
    perm_vals = rng.permutation(np.array([glab[gv] for gv in uniq]))
    perm_map = dict(zip(uniq, perm_vals))
    return np.array([perm_map[gv] for gv in groups])


def permutation_floor(X, y, groups, n_perm=config.N_PERM,
                      percentile=config.PERM_PERCENTILE, **kw):
    """Shuffle labels, rerun the full numeric group-CV, repeat. Return
    (floor_value, distribution). The floor is the given percentile of permuted AUROCs;
    with this n/dimensionality it sits above 0.5, which is exactly why we measure it.

    Permutation is GROUP-AWARE: when the label is constant within each CV group, the
    per-group label is permuted (block permutation) so the null respects the grouping;
    otherwise a plain row-level permutation is used. This keeps the floor valid regardless
    of whether the contrast's label varies within or across groups.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    rng = np.random.default_rng(config.SEED)
    dist = []
    for _ in range(n_perm):
        yp = group_block_permute(y, groups, rng)
        a = numeric_group_auroc(X, yp, groups, **kw)
        if not np.isnan(a):
            dist.append(a)
    dist = np.array(dist) if dist else np.array([0.5])
    return float(np.percentile(dist, percentile)), dist
