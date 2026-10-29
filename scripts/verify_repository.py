#!/usr/bin/env python3
"""Check file layout, checksums, file sizes, and that no secret or local path is committed.

With --write-checksums, rewrite results/CHECKSUMS.txt from the current files first.
"""

import hashlib
import json
import sys

from collusion_probe_audit.paths import ROOT
from collusion_probe_audit.repository_checks import run_all

COVERED = ["configs", "data/manifest.json", "results"]


def write_checksums():
    files = []
    for entry in COVERED:
        p = ROOT / entry
        files += [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ROOT).as_posix()}"
             for f in files if f.name not in ("CHECKSUMS.txt", ".DS_Store")]
    (ROOT / "results" / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if "--write-checksums" in sys.argv:
        write_checksums()
    report = run_all()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["all_pass"] else 1)
