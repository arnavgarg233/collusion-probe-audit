"""Quantities that the paper states but that are not stored as a single field.

Each function reads the stored records under results/tables/evidence/ and returns a number (or a
list of numbers) that a claim in results/tables/claims.json compares with the paper.
"""

from __future__ import annotations

import csv
import json
import re

import numpy as np
from sklearn.metrics import roc_auc_score

from .decoders import tpr_at_fpr
from .paths import EVIDENCE, ROOT


def _json(rel):
    return json.loads((EVIDENCE / rel).read_text())


def _csv(rel):
    lines = [ln for ln in (EVIDENCE / rel).read_text().splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def _liars(subset):
    text = (EVIDENCE / "breadth" / "liars.log").read_text()
    m = re.search(rf"^\s+{re.escape(subset)}: (\{{.*\}})$", text, re.M)
    return json.loads(m.group(1).replace("'", '"'))


def liars_field(subset, key):
    return _liars(subset)[key]


def _org_best(rel):
    """Organism record at the layer with the highest 4-way decode accuracy."""
    d = _json(rel)
    layer = max(d["layers"], key=lambda k: d["layers"][k]["decode_acc"])
    return int(layer), d["layers"][layer]


def setting_delta_floor_p(setting_id):
    """(delta, floor95, p) of one of the fifteen settings in configs/settings.json."""
    s = next(x for x in _settings() if x["id"] == setting_id)
    rel, key = s["delta"]
    if rel.endswith("gapfill_corrected.csv"):
        model, contrast = key.split("|")
        r = next(r for r in _csv(rel) if r["model"] == model and r["contrast"] == contrast)
        return float(r["delta"]), float(r["floor95"]), float(r["p"])
    if rel.endswith("natural_subsat.csv"):
        model, tier, contrast = key.split("|")
        r = next(r for r in _csv(rel) if (r["model"], r["tier"], r["contrast"]) == (model, tier, contrast))
        return float(r["delta"]), float(r["delta_floor95"]), float(r["delta_p"])
    if rel.endswith("liars.log"):
        r = _liars(key)
        return r["delta"], r["floor95"], r["p"]
    if key == "best_decode_layer":
        _, r = _org_best(rel)
        return r["delta"], r["floor95"], r["p"]
    d = _json(rel)
    for part in key.split("."):
        d = d[part]
    return d["delta"], d["floor95"], d["p"]


def _settings():
    return json.loads((ROOT / "configs" / "settings.json").read_text())["settings"]


def _regime_verdicts():
    rows = _csv("breadth/breadth_regime.csv")
    return {f"{r['dataset']}|{r['model']}": r["verdict"] for r in rows}


def n_settings():
    return len(_settings())


def n_verdict(verdict):
    v = _regime_verdicts()
    return sum(v[s["regime_row"]] == verdict for s in _settings())


def executed_auroc_cells():
    """The ten executed-deception settings scored in AUROC units (blue points of Figs. 2-3)."""
    v = _regime_verdicts()
    return [s["id"] for s in _settings()
            if s["units"] == "auroc" and v[s["regime_row"]] == "CONFOUNDED"]


def n_executed_auroc_cells():
    return len(executed_auroc_cells())


def n_executed_inside_floor():
    return sum(setting_delta_floor_p(i)[0] < setting_delta_floor_p(i)[1] for i in executed_auroc_cells())


def n_withheld_auroc_above_floor():
    ids = ["b2w_apps_backdoors", "mistral_decode"]
    return sum(setting_delta_floor_p(i)[0] > setting_delta_floor_p(i)[1] for i in ids)


def win_ids():
    v = _regime_verdicts()
    return [s["id"] for s in _settings() if v[s["regime_row"]] == "WHITE_BOX_WINS"]


def min_win_delta():
    return min(setting_delta_floor_p(i)[0] for i in win_ids())


def _wins_at(threshold):
    return {s["id"] for s in _settings()
            if (lambda d, f: d > f and d >= threshold)(*setting_delta_floor_p(s["id"])[:2])}


def only_extra_win_at(threshold, setting_id):
    """1 if the settings credited at this threshold are the four wins plus exactly setting_id."""
    return int(_wins_at(threshold) - set(win_ids()) == {setting_id} and set(win_ids()) <= _wins_at(threshold))


def n_confounded_below_floor():
    """Confounded settings rejected by the floor itself (delta below floor95)."""
    v = _regime_verdicts()
    return sum(
        setting_delta_floor_p(s["id"])[0] < setting_delta_floor_p(s["id"])[1]
        for s in _settings() if v[s["regime_row"]] == "CONFOUNDED"
    )


def max_gemma_pwd_delta():
    return max(r["delta"] for r in _json("sandbagging/org_gemma_pwd.json")["layers"].values())


def n_wins_at_threshold(threshold):
    """Settings whose delta clears both its floor and the given effect-size threshold."""
    n = 0
    for s in _settings():
        d, f, _ = setting_delta_floor_p(s["id"])
        n += d > f and d >= threshold
    return n


# The family of breadth delta-tests corrected for multiplicity in Section 5 (m = 12): the
# fifteen settings minus the three robust sandbagging decodes.
BREADTH_TESTS = [
    "narcbench_gptoss_core", "narcbench_qwen3_core", "liars_convincing_game", "liars_gender_secret",
    "sleeper_agent", "backdoorllm", "b2w_ai_liar", "b2w_deception_dilution", "b2w_insider_trading",
    "b2w_sandbagging", "gemma_pwd", "b2w_apps_backdoors",
]


def n_breadth_tests():
    return len(BREADTH_TESTS)


def _bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def bh_q(setting_id):
    p = [setting_delta_floor_p(i)[2] for i in BREADTH_TESTS]
    return float(_bh(p)[BREADTH_TESTS.index(setting_id)])


def min_bh_q():
    return float(_bh([setting_delta_floor_p(i)[2] for i in BREADTH_TESTS]).min())


def bonferroni_alpha():
    return 0.05 / len(BREADTH_TESTS)


def smallest_p_setting_is_apps():
    p = {i: setting_delta_floor_p(i)[2] for i in BREADTH_TESTS}
    return int(min(p, key=p.get) == "b2w_apps_backdoors")


def org_best_layer(rel):
    return _org_best(rel)[0]


def narcbench_n(stat):
    rows = _csv("narcbench/natural_subsat.csv")
    n = [int(r["n"]) for r in rows]
    neg = [int(r["n"]) - int(r["pos"]) for r in rows]
    return {"min_n": min(n), "max_n": max(n), "min_neg": min(neg)}[stat]


def roc(dataset, detector, stat):
    """AUROC or TPR at a fixed FPR recomputed from the stored out-of-fold scores."""
    z = np.load(EVIDENCE / "breadth" / "b2w_roc_preds.npz")
    y = z[f"{dataset}__y"].astype(int)
    s = z[f"{dataset}__{detector}"]
    if stat == "auroc":
        return float(roc_auc_score(y, s))
    return tpr_at_fpr(s, y, {"tpr1": 0.01, "tpr5": 0.05}[stat])


def roc_n(dataset):
    return int(len(np.load(EVIDENCE / "breadth" / "b2w_roc_preds.npz")[f"{dataset}__y"]))


def calibration_delta_from_oof(arm, cmi):
    """Controlled delta of the calibrated positive control recomputed from its out-of-fold scores."""
    z = np.load(EVIDENCE / "calibration" / "calibrated_doseresponse_positive_control_raw.npz")
    y = z["y"]
    base = roc_auc_score(y, z[f"oof__{arm}__cmi{cmi}__base"])
    full = roc_auc_score(y, z[f"oof__{arm}__cmi{cmi}__full"])
    return float(full - base)


def calibration_first_credited(arm):
    """Realized bits at the first strength the gate credits, or -1 if it never does."""
    rows = _json("calibration/calibrated_doseresponse_positive_control.json")["arms"][arm]["rows"]
    hit = [r for r in rows if r["gate_go"]]
    return hit[0]["realized_cmi_bits"] if hit else -1


def calibration_first_credited_field(arm, field):
    rows = _json("calibration/calibrated_doseresponse_positive_control.json")["arms"][arm]["rows"]
    return next(r for r in rows if r["gate_go"])[field]


def tfidf_realistic_min_p():
    """Smallest p over the TF-IDF baselines with text AUROC of at least 0.83."""
    rows = [r for r in _csv("narcbench/baseline_sensitivity.csv")
            if r["baseline"] == "tfidf" and float(r["base"]) >= 0.83]
    return min(float(r["p"]) for r in rows)


def tfidf_realistic_min_auroc():
    rows = [r for r in _csv("narcbench/baseline_sensitivity.csv")
            if r["baseline"] == "tfidf" and r["strength"] != "50"]
    return min(float(r["base"]) for r in rows)


def mec_total_n():
    d = _json("stego/mec_highpower.json")
    return d["n_stego"] + d["n_control"]


def sandbag_values(field, quants, layers):
    """A list of one field of sandbag_one_run.json over precisions and layers (range claims)."""
    d = _json("sandbagging/sandbag_one_run.json")["per_quant"]
    return [d[q]["layers"][str(L)][field] for q in quants for L in layers]


def org_decode_values(rels):
    out = []
    for rel in rels:
        if rel == "mistral":
            out.append(_json("sandbagging/sandbag_one_run.json")["per_quant"]["fp16"]["layers"]["24"]["acc"])
        else:
            out.append(_org_best(rel)[1]["decode_acc"])
    return out


def access_fair_floors():
    d = _json("sandbagging/sandbag_access_fair.json")["layers"]
    return [d[L]["floor95"] for L in sorted(d, key=int)]


def worstcase_max_from(k):
    rows = _csv("sandbagging/worstcase_subspace.csv")
    return max(float(r["decode_acc"]) for r in rows if int(r["subspace_removed_topk"]) >= k)


def colludebench_tpr1_equals_tpr5():
    op = _json("breadth/colludebench_fpr.json")["op_text"]
    return int(op["tpr@1%fpr"] == op["tpr@5%fpr"])


FUNCTIONS = {name: obj for name, obj in globals().items() if callable(obj) and not name.startswith("_")}
