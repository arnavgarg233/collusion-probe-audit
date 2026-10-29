"""Repository locations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
EVIDENCE = TABLES / "evidence"
FIGURES = RESULTS / "figures"
DATA = ROOT / "data"
EXTERNAL = DATA / "external"
CLAIMS = TABLES / "claims.json"
CLAIM_VERIFICATION = TABLES / "claim_verification.json"
SANDBAG_ACTS = EVIDENCE / "sandbagging" / "sandbag_quantization_transfer_rawacts.npz"


def rel(path) -> str:
    """Path relative to the repository root, with forward slashes."""
    return Path(path).resolve().relative_to(ROOT).as_posix()
