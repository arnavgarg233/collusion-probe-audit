"""End-to-end Week-1 synthetic gate:
    generate -> behavioural verify -> extract -> probe/controls -> transfer.

Real run (downloads Qwen2.5-1.5B on first use, runs on M2/MPS):
    python src/run_gate.py --n-items 150

Fast plumbing check, NO model (validates the entire stats/control stack):
    python src/run_gate.py --mock                 # expect everything near the floor
    python src/run_gate.py --mock --mock-signal 1.5   # expect a GO render

The synthetic gate passing only proves the pipeline works and that a signal CAN exist
in a setting where we built the hidden variable ourselves. The real go/no-go is NARCBench.
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
import config
import generate_conversations as gen
import extract_activations as extract
import behavioral_verify as bv
import controls
import transfer_eval
import spectrum_tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=config.N_ITEMS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pool", default=config.POOL, choices=["mean", "last", "max"])
    ap.add_argument("--mock", action="store_true", help="no model; templated text + random activations")
    ap.add_argument("--mock-signal", type=float, default=0.0, help="inject signal into collusion(direct)")
    ap.add_argument("--mock-dim", type=int, default=128)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--spectrum", action="store_true", help="also print exploratory spectrum")
    a = ap.parse_args()
    config.ensure_dirs()

    print("\n########## 1. GENERATE ##########")
    if a.skip_generate:
        print("(skipped; using existing transcripts)")
    else:
        gen.generate(a.n_items, a.limit, a.mock, config.TRANSCRIPTS_PATH)

    print("\n########## 2. BEHAVIOURAL VERIFY (before activations) ##########")
    bv.verify(config.TRANSCRIPTS_PATH)

    print("\n########## 3. EXTRACT ACTIVATIONS ##########")
    if a.skip_extract:
        print("(skipped; using existing activations)")
    else:
        extract.extract(config.TRANSCRIPTS_PATH, config.ACTIVATIONS_PATH, a.pool,
                        a.mock, a.mock_signal, a.mock_dim)

    print("\n########## 4. PROBES + CONTROLS + DECISION ##########")
    controls.run_controls(config.ACTIVATIONS_PATH, config.TRANSCRIPTS_PATH)

    print("\n########## 5. TRANSFER (held-out protocol) ##########")
    transfer_eval.run_transfer(config.ACTIVATIONS_PATH)

    if a.spectrum:
        print("\n########## 6. SPECTRUM (exploratory, not in gate) ##########")
        spectrum_tests.run_spectrum(config.ACTIVATIONS_PATH)


if __name__ == "__main__":
    main()
