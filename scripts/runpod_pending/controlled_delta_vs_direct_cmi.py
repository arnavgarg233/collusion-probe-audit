"""DIRECT I(Y;A|B) ESTIMATION vs THE AUROC-DELTA - a second, model-free CMI estimator that
agrees on the confounded cells AND on the sandbagging positive.   ***0 GPU - CPU/local, minutes.***

WHY THIS FILE
-------------
The paper's load-bearing tool is the *access-fair controlled delta*
    delta = AUROC(B + A) - AUROC(B),  pooled cross-fitted GroupKFold OOF,
which we argue is a consistent estimator of whether  I(Y ; A | B) > 0  (Prop 1):  Y = deception
label,  A = activations,  B = the visible behaviour/record (the SAME out-of-fold text/behaviour
score the delta uses, so the comparison is access-fair by construction).  A reviewer can object
that the AUROC-delta is just one (classifier-shaped) functional of the data, and a within-floor
delta might still hide real conditional information that a different estimator would surface.

This script closes that gap by estimating the *same* conditional mutual information I(Y;A|B) THREE
INDEPENDENT, MODEL-FREE (or near-model-free) ways on every cell and checking they tell the same
story as the delta:
    (1) Gao/KSG kNN conditional-MI on PCA(A, 5-10)          - non-parametric, no classifier
    (2) discretized plug-in I(Y;A|B)  (B in deciles, A k-means binned)  - pure counting
    (3) a tiny MINE / InfoNCE critic with a cross-fitted held-out lower bound - neural, but a
        different functional than AUROC.
Each gets a GROUP-AWARE permutation null (block-permute Y by scenario_id / subject) and a 1000x
bootstrap CI.  The headline is a scatter of  AUROC-delta (x)  vs  kNN-CMI in bits (y)  across all
cells with BOTH permutation floors overlaid, plus the Spearman rho between delta and CMI.

EXPECTED (the result this is designed to show)
----------------------------------------------
  * monotone agreement: cells the delta calls CONFOUNDED have CMI inside their own permutation
    null (CMI ~ 0 bits, indistinguishable from the group-null), and the sandbagging / apps wins
    have CMI >> 0 and clear of the null;
  * a positive Spearman rho between delta and kNN-CMI across cells, with both passing through the
    origin band  =>  the controlled delta IS a calibrated estimator of I(Y;A|B), not a classifier
    artifact.  This is the methods-robustness leg that hardens Prop 1 for the TIFS register.

CELLS (CPU assets only - NO model load)
---------------------------------------
  sandbag_decode        : results/runpod/raw_acts/acts.npz  - the POSITIVE.  RAW activations
        (370 x 4 layers x 4096).  Y = correct withheld letter (4-class).  A = PCA(raw acts at the
        chosen layer).  B = the model's VISIBLE no-password answer (one-hot) - the access-fair
        behaviour proxy; on the sandbagged subset it is ~0% correct, so B carries ~no info about
        Y and I(Y;A|B) should be large.  (This reproduces the 0.913 late-layer decode.)
  apps_backdoors        : results/tables/b2w_roc_preds.npz  - a WHITE_BOX win.  A,B are the SAME
        1-D out-of-fold scores the delta uses (act-probe OOF = A, full-access text OOF = B), Y the
        binary backdoor label.  (n=150)
  sandbagging_wmdp_mmlu : results/tables/b2w_roc_preds.npz  - a CONFOUNDED cell (b2w framing puts
        the answer back in the scored transcript).  Same A/B/Y shape.  (n=931)
  narc_*                : NARCBench OOF preds behind results/tables/fpr_operating_points.csv.  If a
        prebuilt OOF npz is present (results/tables/narc_oof_preds.npz with <cell>__{y,text,act}[,
        groups]) we use it; otherwise we REGENERATE the access-fair OOF (act-PCA probe + full-vocab
        TF-IDF text baseline) from the published NARCBench slice via the repo's narcbench loader.
        Confounded by construction (executed collusion) -> CMI should sit in its null.

For the 1-D-score cells (apps / sandbagging_wmdp / narc) "A=PCA(acts,<=30)" degenerates to the
single activation-probe OOF score the delta already consumes - that scalar IS the access-fair
activation channel, so estimating I(Y; act_score | text_score) is the faithful direct analogue of
the delta on those cells.  The sandbag cell is the one place we hold RAW activations, so there we
do the real PCA(acts,<=30)->PCA(5-10) kNN-CMI the spec calls for.

ESTIMATORS - all defined locally, no third-party CMI dep required
-----------------------------------------------------------------
  KSG/Gao kNN CMI:  I(Y;A|B) = I(YA;B-augmented) decomposition via the Frenzel-Pompe estimator
        I(X;Y|Z) = psi(k) - <psi(n_xz+1)+psi(n_yz+1)-psi(n_z+1)>  (digamma counts in the joint
        max-norm ball).  Works for continuous A and discrete-or-continuous B,Y; we treat discrete
        coordinates by adding tiny jitter (standard KSG practice).  npeet is used iff importable,
        else the hand-rolled estimator (validated against npeet in --smoke when available).
  Discretized plug-in:  B -> deciles, A -> k-means bins, Y as-is; I(Y;A|B) = sum_b p(b) I(Y;A|B=b)
        with a Miller-Madow bias correction.
  MINE/InfoNCE:  I(Y;A|B) ~ I([A,B];Y) - I(B;Y) via two InfoNCE critics, cross-fitted (train critic
        on a held-out split, evaluate the bound on the other), torch iff importable else SKIPPED
        (the two model-free estimators carry the result; MINE is corroboration).

OUTPUTS
-------
  results/runpod/tomorrow/controlled_delta_vs_direct_cmi.json   - per-cell delta + 3 CMI estimates,
        each with point + perm-null 95th pct + 1000x bootstrap CI, + Spearman(delta, kNN-CMI).
  ..._scatter.npz / ..._scatter.png (if matplotlib) - the headline delta-vs-CMI scatter w/ floors.
  ..._rawcmi.npz - raw per-cell (A,B,Y,groups,perm draws) for re-analysis.

CLI (mirrors the house pattern; runs on CPU NOW, concurrently with the GPU jobs at no GPU cost):
  python controlled_delta_vs_direct_cmi.py --smoke                 # synthetic, validates math, <1 min
  python controlled_delta_vs_direct_cmi.py --inspect               # list cells + asset availability
  python controlled_delta_vs_direct_cmi.py --run                   # full audit on the real assets
  python controlled_delta_vs_direct_cmi.py --run --cells sandbag_decode,apps_backdoors
"""
from __future__ import annotations
import os, sys

# --- env pinning FIRST (RunPod-friendly; also tames the local libomp double-init) ------------
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # local conda libomp double-init guard
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")          # pin BLAS threads -> no oversubscription thrash
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# CUDA_VISIBLE_DEVICES is honoured if the launcher pins it; this job uses 0 GPU regardless.

import argparse, importlib.util, json, math, time, warnings
warnings.filterwarnings("ignore")           # quiet sklearn convergence / UndefinedMetric spam
import numpy as np
from numpy import log2
from scipy.special import digamma
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

# --- repo locations: laptop default + pod /workspace/repo; overridable via --repo / env --------
_THIS = os.path.abspath(__file__)
_REPO_GUESS = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))   # scripts/runpod_pending/.. -> repo
_REPO_CANDIDATES = [
    os.environ.get("COLLUSION_REPO", ""),
    _REPO_GUESS,
    "/workspace/repo",
    "/Users/akshgarg/Invariant Collusion Probe",
    "/workspace/Invariant Collusion Probe",
]


def _find_repo(explicit=None):
    for p in ([explicit] if explicit else []) + _REPO_CANDIDATES:
        if p and os.path.isdir(os.path.join(p, "scripts")):
            return p
    return _REPO_GUESS


# reused at runtime (set in main once --repo is known)
REPO = _find_repo()
NAT_TO_BIT = 1.0 / math.log(2.0)            # nats -> bits


# ============================================================ exp3 helpers (the delta + floor)
_EXP3_REL = os.path.join("scripts", "exp3_baseline_sweep.py")


def load_exp3(repo):
    """Import the VALIDATED cross-fitted-delta + group-permutation-floor helpers from the repo's
    exp3_baseline_sweep.py by path (so this stays a single paste-and-go module). Falls back to a
    vendored copy with identical math if the repo file is unavailable on the pod."""
    p = os.path.join(repo, _EXP3_REL)
    if os.path.isfile(p):
        spec = importlib.util.spec_from_file_location("exp3", p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            if all(hasattr(mod, f) for f in
                   ("_oof_score_act", "_delta_with_floor", "_verdict", "_safe_auroc")):
                print(f"[exp3] using helpers from {p}", flush=True)
                return mod
        except Exception as e:
            print(f"[exp3] failed to import {p}: {e!r} -> vendored fallback", flush=True)
    print("[exp3] repo helpers not found -> using vendored fallback (identical math)", flush=True)
    return _Exp3Fallback()


# ============================================================ CMI ESTIMATORS (all local)
def _as2d(x):
    x = np.asarray(x, float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _jitter(x, scale=1e-10, seed=0):
    """Tiny noise so KSG ties (discrete coords) don't collapse the max-norm ball - standard KSG."""
    rng = np.random.default_rng(seed)
    return x + rng.normal(0, scale, x.shape) * (np.std(x, axis=0, keepdims=True) + 1e-12)


def _knn_radius(P, k):
    """Chebyshev (max-norm) distance to the k-th neighbour for every point (excluding self)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(P)
    d, _ = nn.kneighbors(P)
    return d[:, -1]                          # k-th neighbour (col 0 is self at dist 0)


def _count_within(P, radius):
    """For each point i, number of OTHER points within strictly-less-than radius_i (max-norm)."""
    tree = NearestNeighbors(metric="chebyshev").fit(P)
    out = np.empty(len(P), dtype=int)
    for i in range(len(P)):
        # radius_neighbors is inclusive; KSG uses strictly-less, so shave the radius a hair.
        idx = tree.radius_neighbors(P[i:i + 1], radius=radius[i] - 1e-15, return_distance=False)[0]
        out[i] = max(0, len(idx) - 1)        # exclude self
    return out


def ksg_cmi(A, B, Y, k=5, seed=0):
    """Frenzel-Pompe / Gao kNN estimator of I(A ; Y | B) in BITS.
        I(X;Y|Z) = psi(k) - < psi(n_xz+1) + psi(n_yz+1) - psi(n_z+1) >
    with counts in the joint (A,B,Y) max-norm ball of radius = dist to k-th joint neighbour.
    Y is one-hot encoded if discrete; B may be continuous or one-hot; A is the (PCA'd) activation.
    Returns CMI in bits (clamped at 0 - the estimator can dip slightly negative on independence)."""
    A = _as2d(A); B = _as2d(B); Y = _as2d(Y)
    # standardize each block so the max-norm is comparable across coordinates
    def _std(M):
        s = M.std(axis=0, keepdims=True); s[s == 0] = 1.0
        return (M - M.mean(axis=0, keepdims=True)) / s
    A, B, Y = _std(A), _std(B), _std(Y)
    A = _jitter(A, seed=seed); B = _jitter(B, seed=seed + 1); Y = _jitter(Y, seed=seed + 2)
    XZ = np.hstack([A, B]); YZ = np.hstack([Y, B]); Z = B; J = np.hstack([A, B, Y])
    n = len(A); k = min(k, n - 1)
    if k < 1:
        return 0.0
    rad = _knn_radius(J, k)
    n_xz = _count_within(XZ, rad); n_yz = _count_within(YZ, rad); n_z = _count_within(Z, rad)
    val = digamma(k) - np.mean(digamma(n_xz + 1) + digamma(n_yz + 1) - digamma(n_z + 1))
    return float(max(0.0, val) * NAT_TO_BIT)


def _npeet_cmi(A, B, Y, k=5):
    """Use npeet's entropy_estimators.cmi (nats) iff importable -> bits. None if npeet absent."""
    try:
        from npeet import entropy_estimators as ee
    except Exception:
        return None
    A = _as2d(A).tolist(); B = _as2d(B).tolist(); Y = _as2d(Y).tolist()
    try:
        return float(max(0.0, ee.cmi(A, Y, B, k=k)) * NAT_TO_BIT)   # ee.cmi(x,y,z)=I(x;y|z) nats
    except Exception:
        return None


def discretized_cmi(A, B, Y, b_bins=10, a_bins=8, seed=0):
    """Plug-in I(Y;A|B) in BITS by binning. B -> deciles (b_bins quantile bins), A -> k-means
    (a_bins clusters), Y kept as its discrete classes. I = sum_b p(b) * I(Y;A | B=b) with a
    Miller-Madow bias correction per conditional cell. Pure counting, no model."""
    A = _as2d(A); B = _as2d(B); Y = np.asarray(Y).ravel()
    n = len(Y)
    # B deciles (on the 1-D B score, or its first PC if multi-D)
    bvec = B[:, 0] if B.shape[1] == 1 else PCA(1, random_state=seed).fit_transform(StandardScaler().fit_transform(B))[:, 0]
    qs = np.quantile(bvec, np.linspace(0, 1, b_bins + 1)[1:-1]) if b_bins > 1 else np.array([])
    bbin = np.digitize(bvec, qs) if len(qs) else np.zeros(n, int)
    # A bins via k-means (cap clusters at n/10 so each cell is populated)
    ka = max(2, min(a_bins, n // 10))
    abin = (KMeans(ka, n_init=4, random_state=seed).fit_predict(StandardScaler().fit_transform(A))
            if A.shape[1] >= 1 and n >= 2 * ka else np.zeros(n, int))
    ycls = Y.astype(int)

    def _mi(a, y):
        """Miller-Madow-corrected I(a;y) in bits on a sub-population."""
        m = len(a)
        if m < 4 or len(set(a)) < 2 or len(set(y)) < 2:
            return 0.0
        au, ai = np.unique(a, return_inverse=True); yu, yi = np.unique(y, return_inverse=True)
        joint = np.zeros((len(au), len(yu)))
        for ii, jj in zip(ai, yi):
            joint[ii, jj] += 1
        joint /= m
        pa = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = joint * (log2(joint) - log2(pa) - log2(py))
        mi = float(np.nansum(terms))
        # Miller-Madow: + (cells_used - rows - cols + 1)/(2 m ln2)
        cells = int((joint > 0).sum())
        mm = (cells - (pa > 0).sum() - (py > 0).sum() + 1) / (2 * m * math.log(2))
        return max(0.0, mi - mm)

    out = 0.0
    for bv in np.unique(bbin):
        sel = bbin == bv
        out += sel.mean() * _mi(abin[sel], ycls[sel])
    return float(max(0.0, out))


def mine_cmi(A, B, Y, epochs=120, seed=0):
    """InfoNCE lower-bound CMI in BITS:  I(Y;A|B) = I([A,B];Y) - I(B;Y), each term a cross-fitted
    InfoNCE bound (train critic on split-1, evaluate the bound on held-out split-2, average both
    folds). torch iff importable, else returns None (the two model-free estimators carry the
    result). Y one-hot. A different functional than the AUROC-delta -> independent corroboration."""
    try:
        import torch, torch.nn as nn
    except Exception:
        return None
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    A = _as2d(A); B = _as2d(B); Y = _as2d(Y)
    # standardize
    def _std(M):
        s = M.std(0, keepdims=True); s[s == 0] = 1.0
        return (M - M.mean(0, keepdims=True)) / s
    A, B = _std(A), _std(B); Yc = _std(Y)
    AB = np.hstack([A, B])
    n = len(Y); idx = rng.permutation(n); half = n // 2
    folds = [(idx[:half], idx[half:]), (idx[half:], idx[:half])]

    def _infonce(X, Z):
        """Cross-fitted InfoNCE estimate of I(X;Z) in nats, averaged over the two folds."""
        Xt = torch.tensor(X, dtype=torch.float32); Zt = torch.tensor(Z, dtype=torch.float32)
        vals = []
        for tr, te in folds:
            net = nn.Sequential(nn.Linear(X.shape[1] + Z.shape[1], 64), nn.ReLU(),
                                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            xt, zt = Xt[tr], Zt[tr]
            for _ in range(epochs):
                opt.zero_grad()
                m = len(xt)
                xi = xt.unsqueeze(1).expand(m, m, X.shape[1])
                zj = zt.unsqueeze(0).expand(m, m, Z.shape[1])
                scores = net(torch.cat([xi, zj], -1)).squeeze(-1)     # [m,m]
                # InfoNCE: diag is positive pair, row is the negatives
                loss = -(torch.diag(scores) - torch.logsumexp(scores, dim=1) + math.log(m)).mean()
                loss.backward(); opt.step()
            with torch.no_grad():
                xe, ze = Xt[te], Zt[te]; m = len(xe)
                xi = xe.unsqueeze(1).expand(m, m, X.shape[1])
                zj = ze.unsqueeze(0).expand(m, m, Z.shape[1])
                sc = net(torch.cat([xi, zj], -1)).squeeze(-1)
                est = (torch.diag(sc) - torch.logsumexp(sc, dim=1) + math.log(m)).mean().item()
            vals.append(max(0.0, est))
        return float(np.mean(vals))

    i_aby = _infonce(AB, Yc); i_by = _infonce(B, Yc)
    return float(max(0.0, i_aby - i_by) * NAT_TO_BIT)


# ============================================================ null + bootstrap for a CMI fn
def _block_perm_y(Y, groups, rng):
    """Group-aware label permutation. If Y is group-constant (scenario_id-constant labels, the
    NARCBench/emergent case), permute the per-GROUP label; else shuffle Y within the whole sample.
    Mirrors exp3._delta_with_floor's null exactly so the two floors are comparable."""
    groups = np.asarray(groups)
    by = {g: Y[groups == g] for g in set(groups)}
    if all(len(set(v.tolist())) == 1 for v in by.values()):
        gl = list(by); perm = rng.permutation([by[g][0] for g in gl])
        mp = dict(zip(gl, perm)); return np.array([mp[g] for g in groups])
    return rng.permutation(Y)


def cmi_with_null(cmi_fn, A, B, Y, groups, n_perm=200, n_boot=1000, seed=0):
    """Point CMI (bits) + group-aware permutation-null 95th pct + p-value + 1000x bootstrap CI.
    cmi_fn(A,B,Y) -> bits. Permutation block-permutes Y by group (same null as the delta floor)."""
    point = cmi_fn(A, B, Y)
    if point is None:
        return dict(point=None, floor95=None, p=None, ci_lo=None, ci_hi=None, n_perm=0, n_boot=0)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = _block_perm_y(Y, groups, rng)
        v = cmi_fn(A, B, yp)
        if v is not None and v == v:
            null.append(v)
    floor95 = float(np.percentile(null, 95)) if null else None
    p = float(np.mean([v >= point for v in null])) if null else None
    # bootstrap CI (resample rows with replacement; keep group structure implicitly via row resample)
    boots = []
    nb = len(Y)
    for _ in range(n_boot):
        bi = rng.integers(0, nb, nb)
        try:
            v = cmi_fn(_as2d(A)[bi], _as2d(B)[bi], np.asarray(Y)[bi])
        except Exception:
            v = None
        if v is not None and v == v:
            boots.append(v)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)
    return dict(point=round(float(point), 5),
                floor95=(round(floor95, 5) if floor95 is not None else None),
                p=(round(p, 4) if p is not None else None),
                ci_lo=(round(ci[0], 5) if ci[0] is not None else None),
                ci_hi=(round(ci[1], 5) if ci[1] is not None else None),
                n_perm=len(null), n_boot=len(boots),
                above_null=(None if floor95 is None else bool(point > floor95)))


# ============================================================ CELL BUILDERS (A, B, Y, groups)
def _safe_auroc(p, y):
    p = np.asarray(p, float); y = np.asarray(y).astype(int); m = ~np.isnan(p)
    return float(roc_auc_score(y[m], p[m])) if (m.sum() > 4 and len(set(y[m].tolist())) == 2) else float("nan")


def build_sandbag(repo, layer_idx=None):
    """The POSITIVE. RAW sandbag activations -> on the SANDBAGGED subset (visible wrong, latent
    right) decode the withheld correct letter. Returns a dict per requested layer.
      A = PCA(<=30) of the raw residual-stream activation at that layer.
      B = the model's VISIBLE no-password answer (the access-fair behaviour proxy). One-hot.
      Y = the correct withheld answer letter (4-class).
      groups = subject (block-permute the null by WMDP subject).
    On the sandbagged subset B ~ chance wrt Y, so I(Y;A|B) ~ I(Y;A) should be LARGE (the 0.913)."""
    p = os.path.join(repo, "results", "runpod", "raw_acts", "acts.npz")
    if not os.path.isfile(p):
        return {}
    z = np.load(p, allow_pickle=True)
    acts = z["act"]                                   # [n, n_layers, hidden]
    correct = np.array([str(c) for c in z["correct"]])
    nopwd = np.array([str(c) for c in z["nopwd_pred"]])
    pwd = np.array([str(c) for c in z["pwd_pred"]])
    subj = np.array([str(s) for s in z["subject"]])
    layers = list(z["layers"])
    # SANDBAGGED subset: model wrong WITHOUT password but right WITH it (the withheld-capability set)
    mask = (nopwd != correct) & (pwd == correct)
    letters = ["A", "B", "C", "D"]; lmap = {c: i for i, c in enumerate(letters)}
    out = {}
    chosen = range(len(layers)) if layer_idx is None else [layer_idx]
    for li in chosen:
        if li >= len(layers):
            continue
        A_full = acts[mask, li, :].astype(float)
        Y = np.array([lmap.get(c, 0) for c in correct[mask]])
        Bv = np.array([lmap.get(c, 0) for c in nopwd[mask]])            # visible no-pwd answer
        B = np.eye(4)[Bv]                                              # one-hot behaviour proxy
        g = subj[mask]
        # PCA(<=30) on the raw acts (real PCA, this is the one cell with raw activations)
        k = min(30, A_full.shape[0] - 1, A_full.shape[1])
        A = PCA(n_components=k, random_state=0).fit_transform(StandardScaler().fit_transform(A_full))
        out[f"sandbag_decode_L{layers[li]}"] = dict(
            A=A, B=B.astype(float), Y=Y, groups=g, raw_acts=A_full,
            y_kind="multiclass4", b_kind="onehot4_visible_answer", n=int(mask.sum()),
            note=f"sandbag subset (visible wrong, latent right) layer {layers[li]}; decode withheld letter")
    return out


def build_b2w_cell(repo, cell):
    """A b2w cell (apps_backdoors / sandbagging_wmdp_mmlu). A,B are the SAME 1-D out-of-fold scores
    the controlled delta consumes: A = activation-probe OOF score, B = full-access text OOF score,
    Y = binary deception label. groups: per-row (no group npz shipped) -> within-sample permutation
    null. This is the faithful direct analogue of the delta on the 1-D-score cells."""
    p = os.path.join(repo, "results", "tables", "b2w_roc_preds.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p, allow_pickle=True)
    if f"{cell}__y" not in z.files:
        return None
    Y = z[f"{cell}__y"].astype(int)
    A = z[f"{cell}__act"].astype(float)               # activation-probe OOF score (the A channel)
    B = z[f"{cell}__text"].astype(float)              # full-access text OOF score (the B channel)
    g = (z[f"{cell}__groups"] if f"{cell}__groups" in z.files else np.arange(len(Y)))
    return dict(A=A, B=B, Y=Y, groups=np.asarray(g), raw_acts=None,
                y_kind="binary", b_kind="text_oof_score", n=int(len(Y)),
                note=f"b2w {cell}: A=act-probe OOF, B=full-access text OOF (same scores the delta uses)")


def build_narc_cells(repo, exp3):
    """NARCBench OOF preds behind fpr_operating_points.csv. Prefer a prebuilt npz
    results/tables/narc_oof_preds.npz with <cell>__{y,text,act[,groups]}; else REGENERATE the
    access-fair OOF (act-PCA probe + full-vocab TF-IDF text baseline) from the published slice
    using the repo's narcbench loader. Returns {cell: dict(A,B,Y,groups,...)}.
    Confounded by construction (executed collusion) -> CMI should sit in its null."""
    cells = {}
    pre = os.path.join(repo, "results", "tables", "narc_oof_preds.npz")
    if os.path.isfile(pre):
        z = np.load(pre, allow_pickle=True)
        names = sorted({f.split("__")[0] for f in z.files if f.endswith("__y")})
        for c in names:
            Y = z[f"{c}__y"].astype(int)
            A = z[f"{c}__act"].astype(float); B = z[f"{c}__text"].astype(float)
            g = (z[f"{c}__groups"] if f"{c}__groups" in z.files else np.arange(len(Y)))
            cells[f"narc_{c}"] = dict(A=A, B=B, Y=Y, groups=np.asarray(g), raw_acts=None,
                                      y_kind="binary", b_kind="text_oof_score", n=int(len(Y)),
                                      note=f"NARCBench {c} (prebuilt access-fair OOF)")
        if cells:
            print(f"[narc] loaded {len(cells)} cells from prebuilt {pre}", flush=True)
            return cells

    # regenerate from the published slice via the repo's narcbench loader (CPU; pod path)
    slice_dir = os.environ.get("NARC_SLICE", "/tmp/narc/slice")
    sys.path.insert(0, os.path.join(repo, "src", "narcbench"))
    sys.path.insert(0, os.path.join(repo, "src", "common"))
    try:
        import narcbench_textctrl as tc
    except Exception as e:
        print(f"[narc] loader unavailable ({e!r}) and no prebuilt npz -> skipping NARCBench cells "
              f"(set NARC_SLICE or drop results/tables/narc_oof_preds.npz)", flush=True)
        return cells
    for model, tier in [("gpt_oss_20b", "core"), ("qwen3_32b", "core")]:
        try:
            sl, texts, Araw, y, g = tc._rows_labels(slice_dir, model, tier, "emergent", None)
        except Exception as e:
            print(f"[narc] {model}/{tier} load failed: {e!r}", flush=True); continue
        y = np.asarray(y).astype(int); g = np.asarray(g); Araw = np.asarray(Araw, float)
        if len(set(y.tolist())) < 2:
            print(f"[narc] {model}/{tier} single-class, skipped", flush=True); continue
        # access-fair OOF: full-vocab TF-IDF text baseline (B) + act-PCA probe (A) via exp3 stacking.
        text_oof, act_oof, _ = exp3._oof_text_act(list(texts), Araw, y, g, max_features=None, act_pca=30)
        m = ~np.isnan(text_oof) & ~np.isnan(act_oof)
        cells[f"narc_{model}_{tier}_emergent"] = dict(
            A=act_oof[m], B=text_oof[m], Y=y[m], groups=g[m], raw_acts=None,
            y_kind="binary", b_kind="text_oof_score", n=int(m.sum()),
            note=f"NARCBench {model}/{tier}/emergent (regenerated access-fair OOF, full-vocab TF-IDF)")
        print(f"[narc] built {model}/{tier}: n={int(m.sum())} pos={int(y[m].sum())} "
              f"text_auroc={_safe_auroc(text_oof, y):.3f} act_auroc={_safe_auroc(act_oof, y):.3f}", flush=True)
    return cells


# ============================================================ delta for a built cell
def cell_delta(exp3, cell, n_perm=200):
    """AUROC-delta for a built (A,B,Y) cell, reusing exp3._delta_with_floor with the score-baseline
    stacker _oof_score_act:  base = B score, full = stacked(B, PCA(A)).  For 1-D A this is the SAME
    delta the b2w/NARCBench tables report; for the raw-acts sandbag cell A is the PCA'd activation
    matrix.  Returns base/act/full AUROC, delta, floor95, p, verdict."""
    A = _as2d(cell["A"]); B = np.asarray(cell["B"], float); Y = np.asarray(cell["Y"]).astype(int)
    g = np.asarray(cell["groups"])
    # _oof_score_act expects a 1-D baseline score. 1-D B (b2w/narc OOF text score) is used as-is;
    # a single-column 2-D B is squeezed; a multi-column B (one-hot sandbag behaviour) -> first PC.
    if B.ndim == 1:
        bscore = B
    elif B.shape[1] == 1:
        bscore = B[:, 0]
    else:
        bscore = PCA(1, random_state=0).fit_transform(StandardScaler().fit_transform(B))[:, 0]
    # binary delta only; for the multiclass sandbag Y we binarize as "Y == argmax-correct class"
    Yb = Y if len(set(Y.tolist())) == 2 else (Y == np.bincount(Y).argmax()).astype(int)
    try:
        d = exp3._delta_with_floor(lambda yy: exp3._oof_score_act(bscore, A, yy, g, act_pca=30),
                                   Yb, g, n_perm=n_perm)
        v = exp3._verdict(d)
    except Exception as e:
        print(f"  [delta] {e!r}", flush=True)
        return dict(base=None, act=None, full=None, delta=None, floor95=None, p=None, verdict="NA")
    return dict(base=round(d["base"], 4), act=round(d["act"], 4), full=round(d["full"], 4),
                delta=round(d["delta"], 4), floor95=round(d["floor95"], 4),
                p=round(d["p"], 4), verdict=v)


# ============================================================ per-cell audit
def audit_cell(exp3, name, cell, n_perm, n_boot, k_knn, do_mine, seed=0):
    A, B, Y, g = cell["A"], cell["B"], cell["Y"], cell["groups"]
    t0 = time.time()
    print(f"\n=== CELL {name}  (n={cell['n']} y_kind={cell['y_kind']} b_kind={cell['b_kind']}) ===", flush=True)
    print(f"    {cell['note']}", flush=True)

    delta = cell_delta(exp3, cell, n_perm=min(n_perm, 200))
    print(f"    AUROC-delta: base(B)={delta['base']} act(A)={delta['act']} full={delta['full']} "
          f"delta={delta['delta']} floor95={delta['floor95']} p={delta['p']} -> {delta['verdict']}", flush=True)

    # CMI #1 - KSG/Gao kNN (npeet if importable for a cross-check, else hand-rolled)
    def _knn(a, b, y):
        v = _npeet_cmi(a, b, y, k=k_knn)
        return v if v is not None else ksg_cmi(a, b, y, k=k_knn, seed=seed)
    knn = cmi_with_null(_knn, A, B, Y, g, n_perm=n_perm, n_boot=n_boot, seed=seed)
    print(f"    CMI[kNN]   = {knn['point']} bits  (null95={knn['floor95']} p={knn['p']} "
          f"CI[{knn['ci_lo']},{knn['ci_hi']}] above_null={knn['above_null']})", flush=True)

    # CMI #2 - discretized plug-in
    disc = cmi_with_null(lambda a, b, y: discretized_cmi(a, b, y, seed=seed),
                         A, B, Y, g, n_perm=n_perm, n_boot=n_boot, seed=seed)
    print(f"    CMI[disc]  = {disc['point']} bits  (null95={disc['floor95']} p={disc['p']} "
          f"CI[{disc['ci_lo']},{disc['ci_hi']}] above_null={disc['above_null']})", flush=True)

    # CMI #3 - MINE/InfoNCE (torch iff importable; fewer perms - it's the slow corroborator)
    mine = dict(point=None, floor95=None, p=None, ci_lo=None, ci_hi=None, n_perm=0, n_boot=0, above_null=None)
    if do_mine:
        mine = cmi_with_null(lambda a, b, y: mine_cmi(a, b, y, seed=seed),
                             A, B, Y, g, n_perm=min(n_perm, 60), n_boot=min(n_boot, 200), seed=seed)
        print(f"    CMI[mine]  = {mine['point']} bits  (null95={mine['floor95']} p={mine['p']} "
              f"above_null={mine['above_null']})", flush=True)

    print(f"    [{time.time()-t0:.1f}s]", flush=True)
    return dict(n=cell["n"], y_kind=cell["y_kind"], b_kind=cell["b_kind"], note=cell["note"],
                delta=delta, cmi_knn=knn, cmi_disc=disc, cmi_mine=mine,
                # store oof-style channels so a re-analysis can recompute anything
                _y=Y, _A=_as2d(A), _B=_as2d(B), _g=np.asarray(g))


# ============================================================ HEADLINE scatter + Spearman
def make_scatter(results, out_base):
    """Scatter of AUROC-delta (x) vs kNN-CMI bits (y) across cells, with the delta permutation floor
    and the CMI permutation floor overlaid; Spearman(delta, CMI). Saves npz always + png if mpl."""
    names, dx, dfloor, cy, cfloor, verdicts = [], [], [], [], [], []
    for nm, r in results.items():
        d = r["delta"]; c = r["cmi_knn"]
        if d["delta"] is None or c["point"] is None:
            continue
        names.append(nm); dx.append(d["delta"]); dfloor.append(d["floor95"] or 0.0)
        cy.append(c["point"]); cfloor.append(c["floor95"] or 0.0); verdicts.append(d["verdict"])
    dx, cy = np.array(dx), np.array(cy)
    rho, prho = (spearmanr(dx, cy) if len(dx) >= 3 else (float("nan"), float("nan")))
    np.savez_compressed(out_base + "_scatter.npz", names=np.array(names, dtype=object),
                        delta=dx, delta_floor=np.array(dfloor), cmi_knn=cy,
                        cmi_floor=np.array(cfloor), verdict=np.array(verdicts, dtype=object),
                        spearman_rho=rho, spearman_p=prho)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5.2))
        col = {"WHITE_BOX_WINS": "tab:red", "CONFOUNDED": "tab:blue", "PARTIAL": "tab:orange", "NA": "gray"}
        for x, y, fl, nm, v in zip(dx, cy, cfloor, names, verdicts):
            ax.scatter(x, y, c=col.get(v, "gray"), s=70, edgecolor="k", zorder=3)
            ax.annotate(nm.replace("sandbag_decode_", "sb_").replace("narc_", ""),
                        (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        if len(cfloor):
            ax.axhline(float(np.median(cfloor)), ls="--", c="gray", lw=1,
                       label="CMI perm-null 95th (median)")
        if len(dfloor):
            ax.axvline(float(np.median(dfloor)), ls=":", c="gray", lw=1,
                       label="delta perm-floor (median)")
        ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
        ax.set_xlabel("AUROC controlled delta  (= I(Y;A|B) via classifier)")
        ax.set_ylabel("kNN CMI  I(Y;A|B)  [bits]")
        ax.set_title(f"Direct CMI vs AUROC-delta  (Spearman rho={rho:.2f}, p={prho:.3f})")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(out_base + "_scatter.png", dpi=150)
        print(f"[fig] wrote {out_base}_scatter.png", flush=True)
    except Exception as e:
        print(f"[fig] matplotlib unavailable ({e!r}) -> npz only", flush=True)
    return dict(spearman_rho=(None if rho != rho else round(float(rho), 4)),
                spearman_p=(None if prho != prho else round(float(prho), 4)),
                n_cells=len(names))


# ============================================================ RUN
ALL_CELLS = ["sandbag_decode", "apps_backdoors", "sandbagging_wmdp_mmlu", "narc"]


def run(args):
    repo = _find_repo(args.repo)
    print(f"[repo] {repo}", flush=True)
    exp3 = load_exp3(repo)
    want = set(args.cells.split(",")) if args.cells else set(ALL_CELLS)

    cells = {}
    if "sandbag_decode" in want:
        # the spec's canonical positive is the late-layer decode (layer 24 -> 0.913). Build all
        # available layers but headline the requested one; default = the layer matching layers[idx].
        sb = build_sandbag(repo, layer_idx=None)
        # keep the canonical late layer (24) + the early control (16) so the scatter shows the
        # layer-emergence (early ~ null, late >> null) - matches the published profile.
        keep = {k: v for k, v in sb.items() if k.endswith(("L24", "L16"))} or sb
        cells.update(keep)
    if "apps_backdoors" in want:
        c = build_b2w_cell(repo, "apps_backdoors")
        if c: cells["apps_backdoors"] = c
    if "sandbagging_wmdp_mmlu" in want:
        c = build_b2w_cell(repo, "sandbagging_wmdp_mmlu")
        if c: cells["sandbagging_wmdp_mmlu"] = c
    if "narc" in want:
        cells.update(build_narc_cells(repo, exp3))

    if not cells:
        raise SystemExit("no cells built - check asset paths (results/runpod/raw_acts/acts.npz, "
                         "results/tables/b2w_roc_preds.npz, NARCBench slice).")
    print(f"[cells] built: {list(cells)}", flush=True)

    results = {}
    for nm, cell in cells.items():
        results[nm] = audit_cell(exp3, nm, cell, n_perm=args.n_perm, n_boot=args.n_boot,
                                 k_knn=args.k_knn, do_mine=(not args.no_mine), seed=args.seed)

    out_dir = os.path.join(repo, "results", "runpod", "tomorrow")
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, "controlled_delta_vs_direct_cmi")
    scat = make_scatter(results, out_base)

    # JSON (strip the heavy arrays) + raw npz (keep them for re-analysis)
    raw = {}
    json_results = {}
    for nm, r in results.items():
        raw[f"{nm}__y"] = r.pop("_y"); raw[f"{nm}__A"] = r.pop("_A")
        raw[f"{nm}__B"] = r.pop("_B"); raw[f"{nm}__g"] = r.pop("_g")
        json_results[nm] = r
    payload = dict(
        experiment="controlled_delta_vs_direct_cmi",
        thesis="the access-fair AUROC controlled delta is a calibrated estimator of I(Y;A|B): "
               "confounded cells sit in the CMI permutation null; the sandbagging/apps wins are CMI>>0.",
        n_perm=args.n_perm, n_boot=args.n_boot, k_knn=args.k_knn,
        npeet_available=(_npeet_cmi(np.random.randn(20, 2), np.random.randn(20, 1),
                                    np.random.randint(0, 2, 20)) is not None),
        cells=json_results, summary=scat)
    json.dump(payload, open(out_base + ".json", "w"), indent=2, default=_jsonify)
    np.savez_compressed(out_base + "_rawcmi.npz", **raw)
    print("\n" + "=" * 72, flush=True)
    print("=== CONTROLLED-DELTA vs DIRECT-CMI : SUMMARY ===", flush=True)
    print(f"Spearman(delta, kNN-CMI) over {scat['n_cells']} cells = rho={scat['spearman_rho']} "
          f"(p={scat['spearman_p']})", flush=True)
    for nm, r in json_results.items():
        d = r["delta"]; c = r["cmi_knn"]
        print(f"  {nm:34s} delta={d['delta']:+.3f} (floor {d['floor95']:.3f}, {d['verdict']:14s}) "
              f"| CMI_knn={c['point']:.4f} bits (null95 {c['floor95']}, above_null={c['above_null']})",
              flush=True)
    print(f"\nwrote {out_base}.json  (+ _scatter.npz/.png + _rawcmi.npz)", flush=True)
    print("EXPECTED: monotone agreement - CONFOUNDED cells have CMI within their group-permutation "
          "null; the sandbagging late-layer decode + apps_backdoors have CMI >> 0 and clear of it; "
          "Spearman rho > 0 => the controlled delta is a calibrated I(Y;A|B) estimator.", flush=True)


def _jsonify(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


# ============================================================ INSPECT
def inspect(args):
    repo = _find_repo(args.repo)
    print(f"[repo] {repo}", flush=True)
    checks = {
        "sandbag acts.npz": os.path.join(repo, "results", "runpod", "raw_acts", "acts.npz"),
        "b2w_roc_preds.npz": os.path.join(repo, "results", "tables", "b2w_roc_preds.npz"),
        "narc prebuilt OOF": os.path.join(repo, "results", "tables", "narc_oof_preds.npz"),
        "narc slice dir": os.environ.get("NARC_SLICE", "/tmp/narc/slice"),
        "exp3 helpers": os.path.join(repo, _EXP3_REL),
    }
    print("\n=== ASSET AVAILABILITY ===")
    for k, p in checks.items():
        print(f"  {'OK ' if os.path.exists(p) else 'MISS'}  {k:22s} {p}")
    print(f"\n=== CELLS this run would build ({args.cells or 'all'}) ===")
    exp3 = load_exp3(repo)
    cells = {}
    sb = build_sandbag(repo)
    cells.update({k: v for k, v in sb.items() if k.endswith(("L24", "L16"))} or sb)
    for c in ("apps_backdoors", "sandbagging_wmdp_mmlu"):
        cc = build_b2w_cell(repo, c)
        if cc: cells[c] = cc
    cells.update(build_narc_cells(repo, exp3))
    for nm, c in cells.items():
        print(f"  {nm:34s} n={c['n']:5d}  y_kind={c['y_kind']:12s} b_kind={c['b_kind']}")
    print(f"\nnpeet importable: "
          f"{_npeet_cmi(np.random.randn(20,2), np.random.randn(20,1), np.random.randint(0,2,20)) is not None}"
          " (else hand-rolled KSG)")
    try:
        import torch  # noqa
        print("torch importable: True (MINE/InfoNCE critic ON)")
    except Exception:
        print("torch importable: False (MINE/InfoNCE SKIPPED; two model-free estimators carry it)")


# ============================================================ SMOKE (synthetic, CPU, <1 min)
def smoke():
    """CPU-only synthetic validation of the audit math - NO model, NO assets, NO network.
      (A) HIDDEN organism: Y is recoverable from A but NOT from B -> CMI > 0 and above its null,
          and the AUROC-delta says WHITE_BOX_WINS  (the sandbagging-positive analogue).
      (B) CONFOUNDED organism: Y is fully explained by B; A is just a noisy copy of B -> CMI ~ 0,
          inside its permutation null, delta within floor  (the executed-deception analogue).
    Also: kNN-CMI ordering (hidden >> confounded), discretized-CMI agreement, and (if npeet is
    importable) the hand-rolled KSG matches npeet to within tolerance.  Asserts the full chain."""
    print("=== SMOKE (synthetic, CPU, no model, no assets) ===", flush=True)
    repo = _find_repo()
    exp3 = load_exp3(repo)
    rng = np.random.default_rng(0)
    n = 400

    # ---- (A) HIDDEN: A carries Y, B is independent of Y ----
    Y = rng.integers(0, 2, n)
    g = np.array([f"grp{i//2}" for i in range(n)])               # twins share a group (honest null)
    A = (Y[:, None] * 2.5 + rng.normal(0, 1.0, (n, 4)))          # A separates the classes
    B = rng.normal(0, 1.0, (n, 1))                               # B independent of Y -> no confound
    cell_h = dict(A=A, B=B, Y=Y, groups=g, raw_acts=A, y_kind="binary",
                  b_kind="synth_independent", n=n, note="HIDDEN: A->Y, B indep")
    dh = cell_delta(exp3, cell_h, n_perm=60)
    knn_h = cmi_with_null(lambda a, b, y: ksg_cmi(a, b, y, k=5), A, B, Y, g, n_perm=60, n_boot=120)
    disc_h = cmi_with_null(lambda a, b, y: discretized_cmi(a, b, y), A, B, Y, g, n_perm=60, n_boot=120)
    print(f"  [hidden] delta={dh['delta']:+.3f} floor95={dh['floor95']:.3f} -> {dh['verdict']}", flush=True)
    print(f"  [hidden] CMI_knn={knn_h['point']} (null95={knn_h['floor95']}, above={knn_h['above_null']}) "
          f"CMI_disc={disc_h['point']} (above={disc_h['above_null']})", flush=True)
    assert dh["verdict"] == "WHITE_BOX_WINS", dh
    assert knn_h["above_null"] and knn_h["point"] > 0.1, knn_h
    assert disc_h["above_null"], disc_h

    # ---- (B) CONFOUNDED: Y = f(B) ; A is a NOISY COPY of B (adds nothing beyond B) ----
    Bc = rng.normal(0, 1.0, (n, 1))
    Yc = (Bc[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)      # Y fully explained by B
    Ac = Bc + rng.normal(0, 0.05, (n, 4))                        # A is B + noise -> no extra info
    gc = np.array([f"g{i//2}" for i in range(n)])
    cell_c = dict(A=Ac, B=Bc, Y=Yc, groups=gc, raw_acts=Ac, y_kind="binary",
                  b_kind="synth_copy_of_B", n=n, note="CONFOUNDED: Y=f(B), A=copy(B)")
    dc = cell_delta(exp3, cell_c, n_perm=60)
    knn_c = cmi_with_null(lambda a, b, y: ksg_cmi(a, b, y, k=5), Ac, Bc, Yc, gc, n_perm=60, n_boot=120)
    disc_c = cmi_with_null(lambda a, b, y: discretized_cmi(a, b, y), Ac, Bc, Yc, gc, n_perm=60, n_boot=120)
    print(f"  [confound] delta={dc['delta']:+.3f} floor95={dc['floor95']:.3f} -> {dc['verdict']}", flush=True)
    print(f"  [confound] CMI_knn={knn_c['point']} (null95={knn_c['floor95']}, above={knn_c['above_null']}) "
          f"CMI_disc={disc_c['point']} (above={disc_c['above_null']})", flush=True)
    assert dc["verdict"] == "CONFOUNDED", dc
    # the confounded CMI must be far below the hidden one and not significantly above its null
    assert knn_c["point"] < knn_h["point"], (knn_c, knn_h)
    assert not knn_c["above_null"], knn_c

    # ---- KSG vs npeet cross-check (only if npeet importable) ----
    npv = _npeet_cmi(A, B, Y, k=5)
    if npv is not None:
        hand = ksg_cmi(A, B, Y, k=5)
        print(f"  [ksg-check] hand-rolled={hand:.4f} vs npeet={npv:.4f} bits", flush=True)
        assert abs(hand - npv) < 0.25, (hand, npv)
    else:
        print("  [ksg-check] npeet not importable -> hand-rolled KSG only (still validated by A/B ordering)",
              flush=True)

    # ---- MINE/InfoNCE sanity (only if torch importable; must order hidden > confounded) ----
    mh = mine_cmi(A, B, Y, epochs=60)
    if mh is not None:
        mc = mine_cmi(Ac, Bc, Yc, epochs=60)
        print(f"  [mine] hidden={mh:.4f} confounded={mc:.4f} bits", flush=True)
        assert mh >= mc, (mh, mc)
    else:
        print("  [mine] torch not importable -> MINE skipped (model-free estimators carry it)", flush=True)

    # ---- scatter + Spearman plumbing on the two cells ----
    res = {"hidden": dict(delta=dh, cmi_knn=knn_h), "confound": dict(delta=dc, cmi_knn=knn_c)}
    import tempfile
    scat = make_scatter(res, os.path.join(tempfile.gettempdir(), "cmi_smoke"))
    print(f"  [scatter] spearman over 2 cells: {scat}", flush=True)

    print("\nSMOKE OK: WHITE_BOX_WINS + CMI-above-null on the hidden organism; CONFOUNDED + "
          "CMI-within-null on the confound; kNN/disc/(mine) all order hidden>confound; "
          "KSG matches npeet (if present). Audit math validated. Ready to --run on the assets.", flush=True)


# ============================================================ vendored exp3 fallback
class _Exp3Fallback:
    """Identical cross-fitted-delta + group-permutation-floor math as scripts/exp3_baseline_sweep.py,
    vendored so the audit runs even if the repo file is unavailable on the pod. Only the helpers this
    script calls are provided (_oof_text_act, _oof_score_act, _delta_with_floor, _verdict, _safe_auroc)."""

    @staticmethod
    def _tfidf_pipe(max_features):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, sublinear_tf=True),
            LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))

    @staticmethod
    def _logit():
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")

    def _oof_text_act(self, texts, A, y, g, max_features, act_pca=30, n_splits=5):
        from sklearn.model_selection import GroupKFold, cross_val_predict
        n = len(y); g = np.asarray(g)
        text_oof = np.full(n, np.nan); act_oof = np.full(n, np.nan); ta_oof = np.full(n, np.nan)
        k = max(2, min(n_splits, len(set(g))))
        for tr, te in GroupKFold(k).split(np.zeros(n), y, g):
            if len(set(y[tr])) < 2:
                continue
            tp = self._tfidf_pipe(max_features).fit([texts[i] for i in tr], y[tr])
            text_te = tp.predict_proba([texts[i] for i in te])[:, 1]; text_oof[te] = text_te
            gtr = g[tr]; ktr = max(2, min(n_splits, len(set(gtr))))
            text_tr = cross_val_predict(self._tfidf_pipe(max_features), [texts[i] for i in tr], y[tr],
                                        groups=gtr, cv=GroupKFold(ktr), method="predict_proba")[:, 1]
            red = make_pipeline(StandardScaler(),
                                PCA(n_components=min(act_pca, len(tr) - 1, A.shape[1]),
                                    random_state=0)).fit(A[tr])
            Atr, Ate = red.transform(A[tr]), red.transform(A[te])
            act_oof[te] = self._logit().fit(Atr, y[tr]).predict_proba(Ate)[:, 1]
            Mtr = np.column_stack([Atr, text_tr]); Mte = np.column_stack([Ate, text_te])
            ta_oof[te] = self._logit().fit(Mtr, y[tr]).predict_proba(Mte)[:, 1]
        return text_oof, act_oof, ta_oof

    def _oof_score_act(self, score, A, y, g, act_pca=30, n_splits=5):
        from sklearn.model_selection import GroupKFold
        n = len(y); g = np.asarray(g); base_oof = np.asarray(score, float).copy()
        A = np.asarray(A, float)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        act_oof = np.full(n, np.nan); sa_oof = np.full(n, np.nan)
        k = max(2, min(n_splits, len(set(g))))
        for tr, te in GroupKFold(k).split(np.zeros(n), y, g):
            if len(set(y[tr])) < 2:
                continue
            red = make_pipeline(StandardScaler(),
                                PCA(n_components=min(act_pca, len(tr) - 1, A.shape[1]),
                                    random_state=0)).fit(A[tr])
            Atr, Ate = red.transform(A[tr]), red.transform(A[te])
            act_oof[te] = self._logit().fit(Atr, y[tr]).predict_proba(Ate)[:, 1]
            Mtr = np.column_stack([Atr, base_oof[tr]]); Mte = np.column_stack([Ate, base_oof[te]])
            sa_oof[te] = self._logit().fit(Mtr, y[tr]).predict_proba(Mte)[:, 1]
        return base_oof, act_oof, sa_oof

    @staticmethod
    def _safe_auroc(pred, y):
        y = np.asarray(y).astype(int); m = ~np.isnan(pred)
        if m.sum() < 4 or len(set(y[m])) < 2:
            return float("nan")
        return roc_auc_score(y[m], pred[m])

    def _delta_with_floor(self, oof_fn, y, g, n_perm=80, seed=0):
        base, act, full = oof_fn(y)
        m = ~np.isnan(full) & ~np.isnan(base)
        b_au = self._safe_auroc(base, y); a_au = self._safe_auroc(act, y); f_au = self._safe_auroc(full, y)
        delta = f_au - b_au if (f_au == f_au and b_au == b_au) else float("nan")
        g = np.asarray(g); by = {gg: y[g == gg] for gg in set(g)}
        group_const = all(len(set(v)) == 1 for v in by.values())
        rng = np.random.default_rng(seed); floors = []
        for _ in range(n_perm):
            if group_const:
                groups = list(by); labs = rng.permutation([by[gg][0] for gg in groups])
                mp = dict(zip(groups, labs)); yp = np.array([mp[gg] for gg in g])
            else:
                yp = rng.permutation(y)
            bb, _, ff = oof_fn(yp)
            vb = self._safe_auroc(bb, yp); vf = self._safe_auroc(ff, yp)
            if vb == vb and vf == vf:
                floors.append(vf - vb)
        floor95 = float(np.percentile(floors, 95)) if floors else float("nan")
        p = float(np.mean([fl >= delta for fl in floors])) if floors else float("nan")
        return dict(n=int(m.sum()), pos=int(y[m].sum()), base=b_au, act=a_au, full=f_au,
                    delta=delta, floor95=floor95, p=p)

    @staticmethod
    def _verdict(d):
        if not (d["delta"] == d["delta"] and d["floor95"] == d["floor95"]):
            return "NA"
        if d["delta"] > d["floor95"] and d["delta"] >= 0.10:
            return "WHITE_BOX_WINS"
        if d["delta"] <= d["floor95"]:
            return "CONFOUNDED"
        return "PARTIAL"


# needed by the fallback's pipelines
from sklearn.pipeline import make_pipeline   # noqa: E402  (kept here so the fallback class is self-contained)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="CPU synthetic validation of the audit math, NO assets/model")
    ap.add_argument("--inspect", action="store_true", help="list cells + asset availability, no audit")
    ap.add_argument("--run", action="store_true", help="full CMI-vs-delta audit on the real assets (0 GPU)")
    ap.add_argument("--cells", default="", help="comma-list subset of: " + ",".join(ALL_CELLS) + " (default all)")
    ap.add_argument("--k-knn", type=int, default=5, help="k for the KSG/Gao kNN CMI estimator")
    ap.add_argument("--n-perm", type=int, default=200, help="group-aware permutations for each CMI null")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for each CMI CI")
    ap.add_argument("--no-mine", action="store_true", help="skip the MINE/InfoNCE critic (slow corroborator)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo", default=os.environ.get("COLLUSION_REPO", ""),
                    help="path to the Invariant-Collusion-Probe repo (assets + exp3 helpers)")
    a = ap.parse_args()

    print(f"[cuda] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')} "
          f"(this job uses 0 GPU)", flush=True)
    if a.smoke:
        smoke(); return
    if a.inspect or not a.run:
        inspect(a); return
    run(a)


if __name__ == "__main__":
    main()
