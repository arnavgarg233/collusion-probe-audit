"""Cross-model colluder-vs-honest probe transfer on NARCBench core activations.

Trains a probe on one model and tests it on another. Activation widths differ across models
(gpt_oss 2880, qwen3 5120, deepseek 5120, llama 8192), so zero-shot transfer uses a shared PCA
fit on the source model for same-width pairs, and probe-direction consistency is compared
across models alongside the transfer of the visible length feature.
"""
from __future__ import annotations
import sys, os, itertools
import numpy as np
_SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ('common', 'narcbench'):
    _pp = os.path.join(_SRCROOT, _sub)
    if _pp not in sys.path:
        sys.path.insert(0, _pp)
import narcbench_loader as nb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = '/tmp/narc/slice'
MODELS = ['gpt_oss_20b', 'qwen3_32b', 'deepseek_r1_32b', 'llama_3_1_70b']
SEED = 0


def load_model(m):
    sl = nb.load_slice(ROOT, m, 'core')
    rows = nb.agent_table(sl, channel='public', behavioral_filter=False)
    X = np.array([r['vec'] for r in rows], float)
    y = np.array([1 if r['role'] == 'colluder' else 0 for r in rows])
    g = np.array([r['run_id'] for r in rows])
    nt = np.array([r['n_tokens'] for r in rows], float)
    return X, y, g, nt, sl['H']


def pooled_oof_1d(feat, y, g, n_splits=5):
    feat = np.asarray(feat, float).reshape(-1, 1); y = np.asarray(y); g = np.asarray(g)
    ng = len(set(g)); k = max(2, min(n_splits, ng))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(feat, y, g):
        if len(set(y[tr])) < 2: continue
        sc = StandardScaler().fit(feat[tr])
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(feat[tr]), y[tr])
        oof[te] = lr.predict_proba(sc.transform(feat[te]))[:, 1]
    mm = ~np.isnan(oof)
    return roc_auc_score(y[mm], oof[mm]) if len(set(y[mm])) > 1 else float('nan')


# ---------- A. shared-PCA transfer (same-dim pairs) -------------------------
def shared_pca_transfer(src, dst, pca_dims=50, C=0.5):
    Xs, ys, gs, nts, Hs = load_model(src)
    Xd, yd, gd, ntd, Hd = load_model(dst)
    assert Hs == Hd, f'dim mismatch {Hs} vs {Hd}'
    # Fit scaler+PCA on the SOURCE only (zero-shot transfer: the target is held-out test
    # data, so fitting the basis on the source+target union leaks test info into the
    # projection). Project both through the source-fit basis.
    sc = StandardScaler().fit(Xs)
    eff = min(pca_dims, Xs.shape[1], Xs.shape[0] - 1)
    pca = PCA(n_components=eff, random_state=SEED).fit(sc.transform(Xs))
    Ps = pca.transform(sc.transform(Xs))
    Pd = pca.transform(sc.transform(Xd))
    lr = LogisticRegression(C=C, max_iter=2000).fit(Ps, ys)
    p = lr.predict_proba(Pd)[:, 1]
    act_transfer = roc_auc_score(yd, p)
    # visible-behaviour transfer baseline: fit n_tokens-logistic on src, apply to dst
    scn = StandardScaler().fit(nts.reshape(-1, 1))
    lrn = LogisticRegression(max_iter=2000).fit(scn.transform(nts.reshape(-1, 1)), ys)
    pn = lrn.predict_proba(scn.transform(ntd.reshape(-1, 1)))[:, 1]
    nt_transfer = roc_auc_score(yd, pn)
    return act_transfer, nt_transfer


# ---------- B. probe-direction consistency (all models, per-model PCA) ------
def model_colluder_dir(m, pca_dims=50, C=0.5):
    """Per-model PCA(50) basis + the logistic colluder direction expressed in
    that model's ORIGINAL feature space (back-projected through PCA)."""
    X, y, g, nt, H = load_model(m)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    eff = min(pca_dims, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=eff, random_state=SEED).fit(Xs)
    P = pca.transform(Xs)
    lr = LogisticRegression(C=C, max_iter=2000).fit(P, y)
    w_pca = lr.coef_.ravel()                       # direction in PCA space
    w_orig = pca.components_.T @ w_pca             # back to standardized-orig space
    w_orig = w_orig / (np.linalg.norm(w_orig) + 1e-12)
    return pca.components_, w_orig, H              # components_: (eff, H)


def cca_subspace_corr(m1, m2, k=10, pca_dims=50):
    """For two same-dim models: how aligned are their top-PCA subspaces and their
    colluder directions? Returns (mean top-k CCA corr of PCA subspaces,
    cosine between back-projected colluder directions)."""
    A1, w1, H1 = model_colluder_dir(m1, pca_dims)
    A2, w2, H2 = model_colluder_dir(m2, pca_dims)
    if H1 != H2:
        return None
    # CCA between the two PCA subspaces (use their loadings as feature sets over H dims)
    kk = min(k, A1.shape[0], A2.shape[0])
    cca = CCA(n_components=kk, max_iter=1000)
    cca.fit(A1.T, A2.T)              # (H, eff) each -> canonical correlations
    Xc, Yc = cca.transform(A1.T, A2.T)
    corrs = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(kk)]
    cos = float(abs(np.dot(w1, w2)))
    return float(np.mean(corrs)), cos, corrs


if __name__ == '__main__':
    print('=== A. SHARED-PCA ZERO-SHOT TRANSFER (same-dim pairs only) ===')
    same_dim_pairs = []
    Hs = {m: load_model(m)[4] for m in MODELS}
    for a, b in itertools.permutations(MODELS, 2):
        if Hs[a] == Hs[b]:
            same_dim_pairs.append((a, b))
    print('same-dim pairs:', [(a, b, Hs[a]) for a, b in same_dim_pairs])
    for src, dst in same_dim_pairs:
        act_t, nt_t = shared_pca_transfer(src, dst)
        verdict = 'BEYOND visible' if act_t > nt_t + 0.03 else 'within/below visible confound'
        print(f'{src:16s} -> {dst:16s} act_transfer={act_t:.3f} | n_tokens_transfer={nt_t:.3f}  [{verdict}]')

    print('\n=== B. PROBE-DIRECTION CONSISTENCY (same-dim pairs) ===')
    seen = set()
    for src, dst in same_dim_pairs:
        key = tuple(sorted([src, dst]))
        if key in seen: continue
        seen.add(key)
        r = cca_subspace_corr(src, dst)
        if r is None: continue
        mean_cca, cos, corrs = r
        print(f'{src:16s} <-> {dst:16s} top-10 PCA-subspace mean CCA={mean_cca:.3f} | colluder-dir |cos|={cos:.3f}')

    print('\n=== B2. cross-dim: report dims differ; n_tokens (visible) transfers across ALL ===')
    for src, dst in itertools.permutations(MODELS, 2):
        if Hs[src] == Hs[dst]: continue
        Xs, ys, gs, nts, _ = load_model(src)
        Xd, yd, gd, ntd, _ = load_model(dst)
        scn = StandardScaler().fit(nts.reshape(-1, 1))
        lrn = LogisticRegression(max_iter=2000).fit(scn.transform(nts.reshape(-1, 1)), ys)
        pn = lrn.predict_proba(scn.transform(ntd.reshape(-1, 1)))[:, 1]
        print(f'{src:16s} -> {dst:16s} (H {Hs[src]}->{Hs[dst]}, no act transfer)  n_tokens_transfer={roc_auc_score(yd, pn):.3f}')
