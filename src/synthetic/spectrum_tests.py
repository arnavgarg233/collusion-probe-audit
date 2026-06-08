"""EXPLORATORY eigenspectrum diagnostics - NOT part of the gate.

The sign of the effect is ambiguous (colluders could compress into fewer invariant
directions OR expand into more), so this cannot be a kill-switch. It belongs in the
mechanism phase if the gate clears, as a "why", not a "whether". Printed for interest
only; nothing here feeds the go/kill decision.
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
import config, probes


def _spectrum_stats(X: np.ndarray) -> dict:
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    p = ev / ev.sum() if ev.sum() > 0 else np.ones_like(ev) / len(ev)
    spectral_entropy = float(-(p * np.log(p + 1e-12)).sum())
    eff_rank = float(np.exp(spectral_entropy))          # effective rank = exp(entropy)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return {"n": int(X.shape[0]), "eff_rank": eff_rank,
            "spectral_entropy": spectral_entropy, "cond_number": cond}


def run_spectrum(activations_path=config.ACTIVATIONS_PATH):
    episodes, meta = probes.load_activations(activations_path)
    print("== EXPLORATORY spectrum (NOT a gate criterion) ==")
    print(f"{'condition/proto':28s} {'n':>4s} {'eff_rank':>9s} {'spec_ent':>9s} {'cond#':>10s}")
    groups: dict[str, list] = {}
    for e in episodes.values():
        if "A" not in e["vectors"]:
            continue
        key = e["condition"] + (f"({e['protocol']})" if e["condition"] == "collusion" else "")
        groups.setdefault(key, []).append(np.concatenate(
            [e["vectors"]["A"], e["vectors"].get("B", e["vectors"]["A"])]))
    out = {}
    for key in sorted(groups):
        X = np.stack(groups[key])
        st = _spectrum_stats(X)
        out[key] = st
        print(f"{key:28s} {st['n']:>4d} {st['eff_rank']:>9.2f} "
              f"{st['spectral_entropy']:>9.2f} {st['cond_number']:>10.1f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=config.ACTIVATIONS_PATH)
    run_spectrum(ap.parse_args().activations)
