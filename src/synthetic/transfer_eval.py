"""Transfer: train the joint probe on one collusion protocol, test zero-shot on a
held-out protocol. This is where the paper's claim actually lives (the supervised
baselines collapse here). honest_secret is the shared negative anchor; items are
split disjointly so no item appears on both sides (no item-identity leak).

Reports in-distribution (direct->direct, held-out items) next to transfer
(direct->shift) so degradation is visible; collapse to ~0.5 is the kill.
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
import argparse
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import config, probes


def _split_items(episodes, frac=0.5):
    items = sorted({e["item_id"] for e in episodes.values()})
    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(items))
    cut = int(len(items) * frac)
    train = {items[i] for i in idx[:cut]}
    test = {items[i] for i in idx[cut:]}
    return train, test


def _joint_rows(episodes, cond, proto, item_set, label):
    X, y = [], []
    for e in episodes.values():
        if e["condition"] == cond and (proto is None or e["protocol"] == proto) \
                and e["item_id"] in item_set and "A" in e["vectors"] and "B" in e["vectors"]:
            X.append(np.concatenate([e["vectors"]["A"], e["vectors"]["B"]]))
            y.append(label)
    return X, y


def _fit_eval(Xtr, ytr, Xte, yte, pca_dims=config.PCA_DIMS):
    Xtr, Xte = np.array(Xtr), np.array(Xte)
    ytr, yte = np.array(ytr), np.array(yte)
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return float("nan")
    steps = [StandardScaler()]
    eff = min(pca_dims, Xtr.shape[1], len(ytr) - 1)
    if eff >= 2 and eff < Xtr.shape[1]:
        steps.append(PCA(n_components=eff, random_state=config.SEED))
    steps.append(LogisticRegression(C=config.LOGREG_C, max_iter=2000))
    est = make_pipeline(*steps).fit(Xtr, ytr)
    return roc_auc_score(yte, est.predict_proba(Xte)[:, 1])


def run_transfer(activations_path=config.ACTIVATIONS_PATH):
    episodes, meta = probes.load_activations(activations_path)
    tr_items, te_items = _split_items(episodes)

    # train pos = collusion(direct) on train items; neg = honest_secret on train items
    Xtr, ytr = [], []
    for X, y in [_joint_rows(episodes, "collusion", "direct", tr_items, 1),
                 _joint_rows(episodes, "honest_secret", None, tr_items, 0)]:
        Xtr += X; ytr += y

    def test_set(cond, proto):
        Xte, yte = [], []
        for X, y in [_joint_rows(episodes, cond, proto, te_items, 1),
                     _joint_rows(episodes, "honest_secret", None, te_items, 0)]:
            Xte += X; yte += y
        return Xte, yte

    in_dist = _fit_eval(Xtr, ytr, *test_set("collusion", "direct"))   # direct -> direct (held-out items)
    transfer = _fit_eval(Xtr, ytr, *test_set("collusion", "shift"))   # direct -> shift

    print("== Transfer (joint probe) ==")
    print(f"  in-distribution (direct->direct, held-out items): {in_dist:.3f}")
    print(f"  transfer        (direct->shift):                  {transfer:.3f}")
    print("  -> degradation is fine; collapse to ~0.5 is the kill")
    return {"in_distribution": in_dist, "transfer": transfer}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=config.ACTIVATIONS_PATH)
    run_transfer(ap.parse_args().activations)
