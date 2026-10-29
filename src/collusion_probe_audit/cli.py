"""Command-line entry points used by scripts/ and reproduce.sh."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .paths import CLAIM_VERIFICATION, FIGURES, rel


def verify_main(argv=None) -> int:
    from .claims import verify

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(CLAIM_VERIFICATION),
                    help="where to write the verification record")
    a = ap.parse_args(argv)
    report = verify()
    Path(a.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for c in report["claims"]:
        for f in c["failures"]:
            print(f"FAIL {c['id']} ({c['where']}): paper {f['paper']}, record {f['observed']} {f['what']}")
    print(f"claims: {report['n_passing']}/{report['n_claims']} pass "
          f"({report['n_checks_passing']}/{report['n_checks']} numbers)")
    return 0 if report["all_pass"] else 1


def figures_main(argv=None) -> int:
    from . import figures

    ap = argparse.ArgumentParser(description="Draw the paper's four figures.")
    ap.add_argument("--check", action="store_true",
                    help="draw into a temporary folder and compare pixels with results/figures/")
    a = ap.parse_args(argv)
    if not a.check:
        figures.main(FIGURES)
        return 0
    import matplotlib.image as mpimg
    import numpy as np

    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        figures.main(tmp)
        for new in sorted(Path(tmp).glob("*.png")):
            old = FIGURES / new.name
            a_, b_ = mpimg.imread(new), mpimg.imread(old)
            same = a_.shape == b_.shape and float(np.abs(a_ - b_).max()) == 0.0
            print(f"{'ok  ' if same else 'DIFF'} {rel(old)}")
            bad += not same
    return 1 if bad else 0


def sandbag_main(argv=None) -> int:
    from . import sandbagging

    ap = argparse.ArgumentParser(description="Recompute the sandbagging decode from the activations.")
    ap.add_argument("--check", action="store_true", help="compare with the stored records instead of writing")
    ap.add_argument("--full", action="store_true",
                    help="also recompute the question-text baseline and its permutation floor "
                         "(about 9 minutes; downloads the WMDP prompt file)")
    a = ap.parse_args(argv)
    return sandbagging.run(check=a.check, full=a.full)


if __name__ == "__main__":
    sys.exit(verify_main())
