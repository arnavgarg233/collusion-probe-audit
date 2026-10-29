"""The controlled delta credits an injected channel only when it carries information beyond the
visible baseline (a small CPU version of the paper's calibrated positive control)."""

import json

import numpy as np

from collusion_probe_audit import audit as ca
from collusion_probe_audit.paths import ROOT


def _data(informative, seed=0, n_groups=24, per_group=6, d=40):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    y = rng.integers(0, 2, size=len(g))
    visible = (y + rng.normal(0, 1.0, len(g)))[:, None]  # a visible channel that partly reveals y
    X = rng.normal(0, 1.0, (len(g), d))
    carrier = y if informative else visible[:, 0]
    X[:, 0] += 5.0 * (carrier - carrier.mean())
    return X, y, g, visible


def test_redundant_channel_is_not_credited():
    X, y, g, B = _data(informative=False)
    r = ca.text_controlled_delta(X, y, g, B=B, n_perm=40, group_aware=False)
    assert ca._verdict(r["delta"], r["delta_floor95"]) != "GO"


def test_informative_channel_is_credited():
    X, y, g, B = _data(informative=True)
    r = ca.text_controlled_delta(X, y, g, B=B, n_perm=40, group_aware=False)
    assert ca._verdict(r["delta"], r["delta_floor95"]) == "GO"


def test_protocol_constants_match_config():
    cfg = json.loads((ROOT / "configs" / "estimator.json").read_text())
    assert ca.ACT_PCA == cfg["activation_pca_max"]
    assert ca.MIN_DELTA == cfg["min_effect"]
    assert ca.PERM_PCTL == cfg["permutation_floor"]["percentile"]
    assert ca.N_PERM == cfg["permutation_floor"]["n_perm_narcbench"]
    assert ca.LOGREG_C == cfg["logistic_C"]
    assert ca.CV_SPLITS == cfg["cross_validation"]["n_splits"]
