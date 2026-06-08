"""Breadth backbone for the (honest) paper: across ALL 4 models x {core, stego}, show that
activation probes separate colluder-vs-honest at ~0.9-1.0 AND so do cheap VISIBLE-behaviour
baselines (vote-relative-to-target for core; bp_corr + response length for stego). Works for
deepseek/llama too - their metadata carries votes/bp_corr/n_tokens even with no transcripts.

Claim it supports: the activation 'collusion signature' is visible behaviour, universally.
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
import sys, json; sys.path.insert(0, "src")
import numpy as np
import stats, narcbench_loader as nb
ROOT = "/tmp/narc/slice"
MODELS = ["gpt_oss_20b", "qwen3_32b", "deepseek_r1_32b", "llama_3_1_70b"]


def load(model, tier):
    adir = f"{ROOT}/activations/{model}/{tier}"
    meta = json.load(open(f"{adir}/metadata_gen.json"))
    L = nb.pick_layer(nb.npz_layers(f"{adir}/activations_gen.npz"), 0.7)
    with np.load(f"{adir}/activations_gen.npz") as d:
        X = d[f"layer_{L}"]
    return meta, X, L


def agg(meta, X, channel="public"):
    by = {}
    for i, r in enumerate(meta):
        if r.get("channel") != channel:
            continue
        by.setdefault((r["run_id"], r["agent_name"]), []).append(i)
    rows = []
    for (rid, ag), idx in by.items():
        r0 = meta[idx[0]]
        rows.append({
            "run_id": rid, "role": r0["role"], "vec": X[idx].mean(0),
            "vote": r0.get("agent_vote"), "target": r0.get("target_option"),
            "fair": r0.get("fair_option"),
            "ntok": float(np.mean([meta[i].get("n_tokens") or 0 for i in idx])),
            "bp": float(np.mean([meta[i].get("bp_corr") or 0 for i in idx])),
        })
    return rows


def behav_features(rows):
    X = []
    for r in rows:
        X.append([float(r["vote"] == r["target"]) if r["target"] else 0.0,
                  float(r["vote"] == r["fair"]) if r["fair"] else 0.0,
                  r["ntok"], r["bp"]])
    return np.array(X)


print(f"{'model':16s} {'tier':6s} {'n':>4s} {'col':>4s} {'act50':>6s} {'act5':>6s} "
      f"{'behav':>6s} {'floor':>6s}")
print("-" * 64)
for model in MODELS:
    for tier in ["core", "stego"]:
        try:
            meta, X, L = load(model, tier)
            rows = agg(meta, X)
            y = np.array([1 if r["role"] == "colluder" else 0 for r in rows])
            g = np.array([r["run_id"] for r in rows])
            if len(set(y)) < 2:
                print(f"{model:16s} {tier:6s} single-class"); continue
            A = np.array([r["vec"] for r in rows])
            B = behav_features(rows)
            act50 = stats.numeric_group_auroc(A, y, g, pca_dims=50)
            act5 = stats.numeric_group_auroc(A, y, g, pca_dims=5)
            beh = stats.numeric_group_auroc(B, y, g, pca_dims=0)
            floor, _ = stats.permutation_floor(A, y, g, pca_dims=50, n_perm=60)
            print(f"{model:16s} {tier:6s} {len(y):>4d} {int(sum(y)):>4d} {act50:>6.3f} "
                  f"{act5:>6.3f} {beh:>6.3f} {floor:>6.3f}")
            sys.stdout.flush()
        except Exception as e:
            print(f"{model:16s} {tier:6s} ERROR {e}")
print("DONE")
