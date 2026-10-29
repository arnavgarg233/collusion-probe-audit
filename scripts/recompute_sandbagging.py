#!/usr/bin/env python3
"""Recompute the sandbagging decode, worst-case subspace curve and (with --full) the
question-text baseline from the stored activations, on CPU."""

from collusion_probe_audit.cli import sandbag_main

if __name__ == "__main__":
    raise SystemExit(sandbag_main())
