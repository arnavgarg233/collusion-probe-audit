"""Check every number the paper states against the stored result records.

results/tables/claims.json lists each claim with its place in the paper, the value as printed,
and where the number comes from. A source is one of

  {"json": <file>, "key": "a.b.0.c"}               a field of a JSON record
  {"csv": <file>, "where": {col: value}, "col": c}  one cell of a CSV table
  {"regex": <file>, "pattern": p}                   the first group of a regex on a log
  {"derive": <function>, "args": [...]}             a function in derive.py

with <file> relative to results/tables/evidence/. A source may also be a list of sources, for
claims about a range of values. The printed value is compared as follows:

  round   (default) the record rounds to the printed value at its printed precision
  exact   counts and identities
  range   "a-b": the smallest record rounds to a and the largest to b
  within  "a-b": every record lies in [a, b]
  lt, gt  the record is below or above the printed bound
  le      the record does not exceed the printed bound at its printed precision
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from . import derive
from .paths import CLAIMS, EVIDENCE, rel

EPS = 1e-9


def _number(text: str):
    """Value and number of decimals of a printed number such as '+0.102', '87.0%' or '1.00'."""
    t = text.strip().lstrip("+").replace("−", "-")
    scale = 1.0
    if t.endswith("%"):
        t, scale = t[:-1], 0.01
    decimals = len(t.split(".")[1]) if "." in t else 0
    if scale != 1.0:
        decimals += 2
    return float(t) * scale, decimals


def _tol(decimals: int) -> float:
    return 0.5 * 10.0 ** (-decimals) + EPS


def _read(source, used: set):
    if isinstance(source, list):
        out = []
        for s in source:
            v = _read(s, used)
            out.extend(v if isinstance(v, list) else [v])
        return out
    if "derive" in source:
        return getattr(derive, source["derive"])(*source.get("args", []))
    path = EVIDENCE / (source.get("json") or source.get("csv") or source.get("regex"))
    used.add(path)
    if "json" in source:
        d = json.loads(path.read_text())
        for part in source["key"].split("."):
            d = d[int(part)] if isinstance(d, list) else d[part]
        return d
    if "csv" in source:
        lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
        rows = [r for r in csv.DictReader(lines)
                if all(r[k] == str(v) for k, v in source["where"].items())]
        if len(rows) != 1:
            raise KeyError(f"{path.name}: {len(rows)} rows match {source['where']}")
        return float(rows[0][source["col"]])
    m = re.search(source["pattern"], path.read_text(), re.M)
    if not m:
        raise KeyError(f"{path.name}: no match for {source['pattern']!r}")
    return float(m.group(1))


def _check(paper: str, observed, mode: str) -> bool:
    if mode in ("range", "within"):
        lo_s, hi_s = re.split(r"(?<=\d)-(?=[\d+-])", paper)
        (lo, dlo), (hi, dhi) = _number(lo_s), _number(hi_s)
        vals = [float(v) for v in observed]
        if mode == "within":
            return all(lo - EPS <= v <= hi + EPS for v in vals)
        return abs(min(vals) - lo) <= _tol(dlo) and abs(max(vals) - hi) <= _tol(dhi)
    value, decimals = _number(paper)
    obs = float(observed)
    if mode == "exact":
        return abs(obs - value) <= EPS
    if mode == "lt":
        return obs < value
    if mode == "gt":
        return obs > value
    if mode == "le":
        return obs <= value + _tol(decimals)
    return abs(obs - value) <= _tol(decimals)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(claims_path: Path = CLAIMS) -> dict:
    registry = json.loads(Path(claims_path).read_text())
    results = []
    for claim in registry["claims"]:
        used: set = set()
        failures = []
        for chk in claim["checks"]:
            try:
                obs = _read(chk["source"], used)
                ok = _check(chk["paper"], obs, chk.get("cmp", "round"))
            except Exception as e:  # a missing record is a failed check, not a crash
                obs, ok = f"error: {e}", False
            if not ok:
                failures.append({"paper": chk["paper"], "observed": obs, "what": chk.get("what", "")})
        results.append({
            "id": claim["id"],
            "where": claim["where"],
            "checks": len(claim["checks"]),
            "passed": not failures,
            "failures": failures,
            "evidence_sha256": {rel(p).removeprefix("results/tables/evidence/"): _sha(p)
                                for p in sorted(used)},
        })
    n_checks = sum(r["checks"] for r in results)
    n_failed = sum(len(r["failures"]) for r in results)
    return {
        "all_pass": n_failed == 0,
        "n_claims": len(results),
        "n_passing": sum(r["passed"] for r in results),
        "n_checks": n_checks,
        "n_checks_passing": n_checks - n_failed,
        "claims": results,
    }
