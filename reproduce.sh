#!/usr/bin/env bash
# Replays every number in the paper from the stored records, on CPU, offline.
# FULL=1 also recomputes the question-text baseline of the sandbagging decode with its
# permutation floor (about 9 minutes; downloads the WMDP prompt file, see data/manifest.json).
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/collusion-replay.XXXXXX")
trap 'rm -rf "$TMP_ROOT"' EXIT

export SOURCE_DATE_EPOCH=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
for v in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS; do
  export "$v=${!v:-1}"
done

cd "$ROOT"

uv sync --locked
uv lock --check
uv run --frozen --no-sync python scripts/verify_claims.py --output "$TMP_ROOT/claim_verification.json" \
  | tee "$TMP_ROOT/claims.txt"
cmp "$TMP_ROOT/claim_verification.json" results/tables/claim_verification.json
uv run --frozen --no-sync ruff check src scripts/*.py tests
uv run --frozen --no-sync pytest -q -p no:cacheprovider | tee "$TMP_ROOT/pytest.txt"
uv run --frozen --no-sync python scripts/make_figures.py --check | grep -E '^(ok|DIFF)' | tee "$TMP_ROOT/figures.txt"
if [ "${FULL:-0}" = "1" ]; then
  uv run --frozen --no-sync python scripts/recompute_sandbagging.py --check --full | tee "$TMP_ROOT/sandbag.txt"
else
  uv run --frozen --no-sync python scripts/recompute_sandbagging.py --check | tee "$TMP_ROOT/sandbag.txt"
fi
uv run --frozen --no-sync python scripts/verify_repository.py >"$TMP_ROOT/repository_checks.json"
shasum -a 256 -c results/CHECKSUMS.txt >/dev/null

if [ -d .git ]; then
  git diff --check
  test -z "$(git status --porcelain --untracked-files=all)"
fi

echo "$(tail -1 "$TMP_ROOT/claims.txt")"
echo "tests passed: $(grep -Eo '[0-9]+ passed' "$TMP_ROOT/pytest.txt" | tail -1 | cut -d' ' -f1)"
echo "figures redrawn identically: $(grep -c '^ok' "$TMP_ROOT/figures.txt") of 4"
echo "sandbagging records reproduced: $(grep -c '^ok' "$TMP_ROOT/sandbag.txt") of 3"
echo "claim replay: PASS"
