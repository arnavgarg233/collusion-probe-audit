"""Driver for the NARCBench controlled-delta table.

Runs narcbench_textctrl on the core and transfer tiers, both contrasts, for the two models
that ship scenario text (gpt_oss_20b, qwen3_32b), and appends one row per cell to
results/tables/natural_subsat.csv. DeepSeek-R1-32B and Llama-3.1-70B ship activations without
transcripts, so no text control is defined for them.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, csv, datetime
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import narcbench_textctrl as tc

ROOT = "/tmp/narc/slice"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "results", "tables", "evidence", "narcbench", "natural_subsat.csv")
N_PERM = 60

# scenario text exists only for these two models (deepseek/llama have activations but
# no per-run results.json -> text-controlled delta undefined; gapfill.csv shows nan).
MODELS = ["gpt_oss_20b", "qwen3_32b"]
TIERS = ["transfer", "core"]
CONTRASTS = ["emergent", "colluder"]

FIELDS = ["timestamp", "model", "tier", "contrast", "layer", "n", "pos",
          "text_visible_auroc", "act_auroc", "ta_auroc", "delta",
          "delta_floor95", "delta_p", "verdict"]


def has_scenarios(model, tier):
    d = os.path.join(ROOT, "scenarios", model, tier)
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def main():
    rows = []
    for tier in TIERS:
        for model in MODELS:
            for contrast in CONTRASTS:
                if not has_scenarios(model, tier):
                    print(f"SKIP {model}/{tier}/{contrast}: no scenario text on disk")
                    continue
                sl0_layer = None
                r = tc.run(ROOT, model, tier, contrast, layer=None,
                           act_pca=30, n_perm=N_PERM)
                if r is None:
                    continue
                # tc.run prints; recover the layer it picked via a cheap re-load is wasteful,
                # so re-derive from the loader once (layer is deterministic for the slice).
                import narcbench_loader as nb
                layer = nb.load_slice(ROOT, model, tier)["layer"]
                rows.append({
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "model": model, "tier": tier, "contrast": contrast, "layer": layer,
                    "n": r["n"], "pos": r["pos"],
                    "text_visible_auroc": round(r["text"], 4),
                    "act_auroc": round(r["act"], 4),
                    "ta_auroc": round(r["ta"], 4),
                    "delta": round(r["delta"], 4),
                    "delta_floor95": round(r["delta_floor95"], 4),
                    "delta_p": round(r["delta_p"], 4),
                    "verdict": r["verdict"],
                })

    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nWROTE {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
