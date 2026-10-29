import copy
import json

from collusion_probe_audit.claims import _check, verify
from collusion_probe_audit.paths import CLAIMS


def test_every_claim_matches_its_record():
    report = verify()
    failures = [(c["id"], f) for c in report["claims"] for f in c["failures"]]
    assert report["all_pass"], failures
    assert report["n_checks"] > 300


def test_a_wrong_number_fails(tmp_path):
    reg = json.loads(CLAIMS.read_text())
    bad = copy.deepcopy(reg)
    claim = next(c for c in bad["claims"] if c["id"] == "C-T12-SANDBAG")
    claim["checks"][7]["paper"] = "0.913"  # layer-24 accuracy is 0.912
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(bad))
    report = verify(path)
    assert not report["all_pass"]
    assert [c["id"] for c in report["claims"] if not c["passed"]] == ["C-T12-SANDBAG"]


def test_comparison_rules():
    assert _check("0.912", 0.9124, "round")
    assert not _check("0.912", 0.9126, "round")
    assert _check("87.0%", 0.870, "round")
    assert _check("+0.102", 0.1022, "round")
    assert _check("0.98-0.99", [0.9791, 0.9875], "range")
    assert not _check("0.98-0.99", [0.9741, 0.9875], "range")
    assert _check("0.9-1.0", [0.963, 1.0], "within")
    assert _check("0.005", 0.0, "lt")
    assert _check("15", 15, "exact")
