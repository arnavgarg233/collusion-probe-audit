"""Build leak-aware contrasts from the stored activations and run the A/B/joint
probes. `item_id` is the CV group so an item's two conditions never split across
train/test. Positive class (label 1) is always the first condition passed.
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
import argparse, json
import numpy as np
import config, stats


def load_activations(path: str = config.ACTIVATIONS_PATH):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    episodes: dict[str, dict] = {}
    for i in range(len(X)):
        eid = str(d["episode_id"][i])
        e = episodes.setdefault(eid, {
            "item_id": str(d["item_id"][i]),
            "condition": str(d["condition"][i]),
            "protocol": str(d["protocol"][i]),
            "vectors": {},
        })
        e["vectors"][str(d["agent"][i])] = X[i]
    meta = {"layer": int(d["layer"][0]), "n_layers": int(d["n_layers"][0]),
            "pool": str(d["pool"][0]), "H": X.shape[1]}
    return episodes, meta


def _match(e, cond, proto):
    return e["condition"] == cond and (proto is None or e["protocol"] == proto)


def build_numeric_contrast(episodes, condA, condB, feature_set,
                           protoA=None, protoB=None):
    X, y, groups = [], [], []
    for e in episodes.values():
        if _match(e, condA, protoA):
            label = 1
        elif _match(e, condB, protoB):
            label = 0
        else:
            continue
        v = e["vectors"]
        if feature_set == "A":
            if "A" not in v:
                continue
            feat = v["A"]
        elif feature_set == "B":
            if "B" not in v:
                continue
            feat = v["B"]
        elif feature_set == "joint":
            if "A" not in v or "B" not in v:
                continue
            feat = np.concatenate([v["A"], v["B"]])
        else:
            raise ValueError(feature_set)
        X.append(feat); y.append(label); groups.append(e["item_id"])
    return np.array(X), np.array(y), np.array(groups)


def build_text_contrast(records, condA, condB, protoA=None, protoB=None):
    texts, y, groups = [], [], []
    for r in records:
        ca = r["condition"] == condA and (protoA is None or r["protocol"] == protoA)
        cb = r["condition"] == condB and (protoB is None or r["protocol"] == protoB)
        if ca:
            label = 1
        elif cb:
            label = 0
        else:
            continue
        texts.append(r["visible_text"]); y.append(label); groups.append(r["item_id"])
    return texts, np.array(y), np.array(groups)


def probe_contrast(episodes, condA, condB, protoA=None, protoB=None,
                   pca_dims=config.PCA_DIMS):
    """Return {'A':auroc,'B':auroc,'joint':auroc,'n':n_pairs}."""
    out = {}
    for fs in ("A", "B", "joint"):
        X, y, g = build_numeric_contrast(episodes, condA, condB, fs, protoA, protoB)
        out[fs] = stats.numeric_group_auroc(X, y, g, pca_dims=pca_dims) if len(y) else float("nan")
        if fs == "joint":
            out["n"] = int(len(y))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=config.ACTIVATIONS_PATH)
    a = ap.parse_args()
    episodes, meta = load_activations(a.activations)
    print(f"[probe] {len(episodes)} episodes, residual dim {meta['H']}, "
          f"layer {meta['layer']}/{meta['n_layers']}, pool {meta['pool']}")
    r = probe_contrast(episodes, "collusion", "honest_secret", protoA=config.PRIMARY_PROTOCOL)
    print(f"  collusion(direct) vs honest_secret  n={r['n']}  "
          f"A={r['A']:.3f}  B={r['B']:.3f}  joint={r['joint']:.3f}")
