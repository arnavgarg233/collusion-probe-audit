"""NARCBench-Core loader, wired to the real published schema.

Repo:    github.com/aaronrose227/narcbench
Dataset: huggingface.co/datasets/aaronrose227/narcbench  (ships PRE-EXTRACTED activations,
         so the gate runs on CPU/M2 - no GPU generation/extraction needed for analysis.)

On-disk slice layout (mirrors the HF repo):
    <root>/scenarios/<model>/<tier>/<run_id>/{run_config.json, agent_prompts.json, results.json}
    <root>/activations/<model>/<tier>/{metadata_gen.json, activations_gen.npz}

Key facts (verified on gpt_oss_20b/core):
  - metadata_gen.json: one row per (agent, channel/phase, round); rows ALIGN by index to
    each `layer_<idx>` array in activations_gen.npz  (e.g. 612 rows x 2880 dims).
  - run_config.json carries the behavioural label already: collusion_success, votes,
    colluder_voted_target, target_option, roles{name->colluder/honest}, coalition[...].
  - the colluder's coalition instruction lives in agent_prompts[*].system_prompt -> the
    prompt-only control is the load-bearing confound check (honest agents lack it).
  - channel 'private' rows are COLLUDER-ONLY (coalition chat); colluder-vs-honest
    contrasts therefore use channel='public', where both roles speak.

Maps onto our schema so stats.py (numeric_group_auroc / text_group_auroc /
permutation_floor) runs unchanged. CV group = run_id.
"""
from __future__ import annotations
# --- path bootstrap: make sibling src/ subpackages importable as flat modules ---
import os as _os, sys as _sys
_SRCROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('', 'common', 'synthetic', 'narcbench', 'audit', 'exploratory'):
    _pp = _os.path.join(_SRCROOT, _sub)
    if _os.path.isdir(_pp) and _pp not in _sys.path:
        _sys.path.insert(0, _pp)
# --- end bootstrap ---
import json, os, re
import numpy as np


def _scn_dir(root, model, tier): return os.path.join(root, "scenarios", model, tier)
def _act_dir(root, model, tier): return os.path.join(root, "activations", model, tier)


def npz_layers(npz_path):
    with np.load(npz_path) as d:
        return sorted(int(re.search(r"(\d+)$", k).group(1)) for k in d.files if k.startswith("layer_"))


def pick_layer(layers, frac=0.7):
    """Layer nearest `frac` of the extracted band (the dataset ships a mid-depth band,
    e.g. 10..14 for gpt-oss-20b, so we target within what's available)."""
    lo, hi = min(layers), max(layers)
    target = lo + frac * (hi - lo)
    return min(layers, key=lambda L: abs(L - target))


def load_slice(root, model, tier, layer=None, frac=0.7):
    adir, sdir = _act_dir(root, model, tier), _scn_dir(root, model, tier)
    meta = json.load(open(os.path.join(adir, "metadata_gen.json")))
    npz_path = os.path.join(adir, "activations_gen.npz")
    layers = npz_layers(npz_path)
    layer = layer if layer is not None else pick_layer(layers, frac)
    with np.load(npz_path) as d:
        X = d[f"layer_{layer}"].astype(np.float32)
    assert X.shape[0] == len(meta), f"row mismatch {X.shape[0]} vs {len(meta)}"

    run_ids = sorted({r["run_id"] for r in meta})
    configs, prompts, results = {}, {}, {}
    for rid in run_ids:
        rd = os.path.join(sdir, rid)
        configs[rid] = _maybe(os.path.join(rd, "run_config.json"))
        prompts[rid] = _maybe(os.path.join(rd, "agent_prompts.json"))
        results[rid] = _maybe(os.path.join(rd, "results.json"))
    return {"meta": meta, "X": X, "layer": layer, "layers": layers,
            "H": int(X.shape[1]), "configs": configs, "prompts": prompts,
            "results": results, "model": model, "tier": tier}


def _maybe(path):
    return json.load(open(path)) if os.path.exists(path) else None


# ---- text joins -----------------------------------------------------------
def _agent_prompt_text(prompts_list, agent):
    """All prompt text the agent received (system+user, every turn) - INCLUDES the
    colluder's coalition instruction. This is the prompt-only confound feature."""
    if not prompts_list:
        return ""
    parts = []
    for t in prompts_list:
        if t.get("agent_name") == agent:
            parts.append((t.get("system_prompt", "") + " " + t.get("user_prompt", "")).strip())
    return " ".join(parts)


def _agent_visible_text(results, agent):
    """VISIBLE text only (public messages + final recommendation). Excludes private
    coalition chat - this is what an overt-text judge would see."""
    if not results:
        return ""
    out = []
    for m in results.get("public_messages", []) or []:
        if isinstance(m, str) and m.startswith(f"[{agent}]"):
            out.append(m.split("]:", 1)[-1].strip())
    ft = (results.get("final_texts", {}) or {}).get(agent, "")
    if ft:
        out.append(ft)
    return " ".join(out)


def _run_failed_collusion(cfg, row):
    if cfg is not None and "collusion_success" in cfg:
        return cfg.get("mode") == "collusion" and not cfg.get("collusion_success", False)
    return row.get("mode") == "collusion" and not row.get("collusion_success", False)


# ---- aggregation to per-(run, agent) -------------------------------------
def agent_table(sl, channel="public", behavioral_filter=True):
    """Mean-pool each agent's `channel` activation rows into one vector; attach role,
    mode, prompt text, visible text, run_id (CV group). One row per (run, agent).
    behavioral_filter drops collusion runs that did NOT coordinate (Core: collusion_success).
    """
    by_key: dict[tuple, list[int]] = {}
    for i, r in enumerate(sl["meta"]):
        if channel and r.get("channel") != channel:
            continue
        if behavioral_filter and _run_failed_collusion(sl["configs"].get(r["run_id"]), r):
            continue
        by_key.setdefault((r["run_id"], r["agent_name"]), []).append(i)

    rows = []
    for (rid, agent), idxs in by_key.items():
        r0 = sl["meta"][idxs[0]]
        rows.append({
            "run_id": rid, "agent": agent, "role": r0["role"], "mode": r0["mode"],
            "scenario_id": r0.get("scenario_id"),
            "n_tokens": float(np.mean([sl["meta"][i].get("n_tokens", 0) or 0 for i in idxs])),
            "vec": sl["X"][idxs].mean(0),
            "prompt_text": _agent_prompt_text(sl["prompts"].get(rid), agent),
            "visible_text": _agent_visible_text(sl["results"].get(rid), agent),
            "context_text": _run_public_text(sl["results"].get(rid)),
            "agent_vote": r0.get("agent_vote"),
            "bp_corr": float(np.mean([sl["meta"][i].get("bp_corr") or 0.0 for i in idxs])),
        })
    return rows


def _run_public_text(results):
    """FULL public discussion of the run (all agents' public messages) - the context an
    agent was conditioned on. The proper visible baseline for the emergent contrast:
    honest agents in collusion runs READ the colluders' visible arguments."""
    if not results:
        return ""
    return " ".join(m for m in (results.get("public_messages", []) or []) if isinstance(m, str))


# ---- contrast builders (return arrays for stats.py) -----------------------
def contrast(rows, label_fn, keep_fn=None):
    """label_fn(row)->1/0/None (None=skip). Returns X, y, groups, prompt_texts, visible_texts."""
    X, y, g, P, V = [], [], [], [], []
    for r in rows:
        if keep_fn and not keep_fn(r):
            continue
        lab = label_fn(r)
        if lab is None:
            continue
        X.append(r["vec"]); y.append(lab); g.append(r["run_id"])
        P.append(r["prompt_text"]); V.append(r["visible_text"])
    return np.array(X), np.array(y), np.array(g), P, V


def colluder_vs_honest(r):
    return 1 if r["role"] == "colluder" else 0          # honest pools both modes (per spec)


def honest_collusion_vs_control(r):
    if r["role"] != "honest":
        return None
    return 1 if r["mode"] == "collusion" else 0          # matched-prompt emergent contrast


def inspect(path):  # kept for convenience; now points at a slice root
    import glob
    print("scenarios:", len(glob.glob(os.path.join(path, "scenarios", "*", "*", "*"))))
    print("activation tiers:", glob.glob(os.path.join(path, "activations", "*", "*")))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="gpt_oss_20b")
    ap.add_argument("--tier", default="core")
    a = ap.parse_args()
    sl = load_slice(a.root, a.model, a.tier)
    print(f"layer {sl['layer']} of {sl['layers']}, H={sl['H']}, rows={len(sl['meta'])}")
    rows = agent_table(sl)
    from collections import Counter
    print("per-agent rows:", len(rows),
          "| role:", dict(Counter(r["role"] for r in rows)),
          "| mode:", dict(Counter(r["mode"] for r in rows)))
