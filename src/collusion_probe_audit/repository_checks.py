"""Checks on the repository itself: layout, checksums, file sizes and stray secrets or paths."""

from __future__ import annotations

import hashlib
import re
import subprocess

from .paths import ROOT

REQUIRED = [
    "README.md", "LICENSE", "pyproject.toml", "uv.lock", "reproduce.sh", "configs/estimator.json",
    "configs/sandbag_decoder.json", "configs/settings.json", "data/manifest.json",
    "results/CHECKSUMS.txt", "results/tables/claims.json", "results/tables/claim_verification.json",
    "results/figures/fig_regime_map.png", "results/figures/fig_forest.png",
    "results/figures/fig_roc_boundary.png", "results/figures/fig_sandbag_layers.png",
]
MAX_BYTES = 50 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".json", ".csv", ".log", ".md", ".txt", ".toml", ".sh", ".yml", ".cfg"}
# Secret-looking tokens and absolute home paths. Built from pieces so this file does not match itself.
PATTERNS = [
    re.compile("sk-" + r"[A-Za-z0-9_-]{16,}"),
    re.compile("hf_" + r"[A-Za-z0-9]{20,}"),
    re.compile("nvapi" + r"-[A-Za-z0-9_-]{10,}"),
    re.compile(r"(?i)(api[_-]?key|token)\s*=\s*['\"][A-Za-z0-9_-]{12,}['\"]"),
    re.compile("/" + "Users/" + r"[A-Za-z]"),
    re.compile("/" + "home/" + r"[a-z]+/"),
]


def _files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
        names = [n for n in out.stdout.splitlines() if n]
        if names:
            return [ROOT / n for n in names]
    except (OSError, subprocess.CalledProcessError):
        pass
    skip = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return [p for p in ROOT.rglob("*")
            if p.is_file() and not skip & set(p.relative_to(ROOT).parts)
            and "data/external" not in p.relative_to(ROOT).as_posix()]


def check_layout():
    missing = [r for r in REQUIRED if not (ROOT / r).exists()]
    return {"pass": not missing, "missing": missing}


def check_checksums():
    bad = []
    lines = (ROOT / "results" / "CHECKSUMS.txt").read_text().splitlines()
    for line in lines:
        digest, name = line.split(maxsplit=1)
        p = ROOT / name.strip()
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(name.strip())
    return {"pass": not bad, "n_files": len(lines), "mismatched": bad}


def check_sizes():
    big = [p.relative_to(ROOT).as_posix() for p in _files() if p.stat().st_size > MAX_BYTES]
    return {"pass": not big, "over_50MB": big}


def check_text_hygiene():
    hits = []
    for p in _files():
        if p.suffix not in TEXT_SUFFIXES or p.name == "uv.lock":
            continue
        text = p.read_text(errors="ignore")
        for pat in PATTERNS:
            if pat.search(text):
                hits.append(f"{p.relative_to(ROOT).as_posix()}: {pat.pattern}")
    return {"pass": not hits, "hits": hits}


def run_all():
    checks = {
        "layout": check_layout(),
        "checksums": check_checksums(),
        "sizes": check_sizes(),
        "text_hygiene": check_text_hygiene(),
    }
    checks["all_pass"] = all(c["pass"] for c in checks.values())
    return checks
