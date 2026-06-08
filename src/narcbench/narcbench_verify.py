"""Adversarial confound battery for the NARCBench tiny-slice result.

The headline to break: EMERGENT (honest collusion vs control) shows act~0.90 with the
agent's OWN text/prompt at the floor. Each --check below attacks that from one angle.
Emits machine-readable `RESULT <check> | key=val ...` lines plus a one-line VERDICT.

Usage: python src/narcbench_verify.py --check <name> --root /tmp/narc/slice
checks: context_text | layers | scenario_group | nofilter | degenerate | ntokens | primary_votematch
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
import argparse
import numpy as np
import stats, narcbench_loader as nb

ROOT_DEFAULT = "/tmp/narc/slice"


def _load(root, model, tier, layer):
    sl = nb.load_slice(root, model, tier, layer=layer)
    return sl, nb.agent_table(sl, channel="public", behavioral_filter=True)


def _emergent(rows):
    sub = [r for r in rows if r["role"] == "honest"]
    y = np.array([1 if r["mode"] == "collusion" else 0 for r in sub])
    # emergent label = run mode (constant within run); collusion & control runs SHARE
    # scenarios, so grouping by run_id leaks scenario identity across folds. Group by
    # scenario_id (matches narcbench_textctrl.py / collusion_audit.py / narcbench_gate.py).
    g = np.array([r["scenario_id"] for r in sub])
    return sub, y, g


def _primary(rows):
    y = np.array([1 if r["role"] == "colluder" else 0 for r in rows])
    g = np.array([r["run_id"] for r in rows])
    return rows, y, g


def check_context_text(root, model, tier):
    """THE key confound: does the FULL public discussion separate the contrast, even though
    the agent's OWN text does not? If yes, the emergent activation 'signal' is reading the
    visible discussion (colluders argue in public), not a covert internal signature."""
    sl, rows = _load(root, model, tier, None)
    for nm, (sub, y, g) in [("EMERGENT", _emergent(rows)), ("PRIMARY", _primary(rows))]:
        own = stats.text_group_auroc([r["visible_text"] for r in sub], y, g)
        ctx = stats.text_group_auroc([r["context_text"] for r in sub], y, g)
        print(f"RESULT context_text {nm} | own_text={own:.3f} full_public_transcript={ctx:.3f} n={len(y)}")
    print("VERDICT: if EMERGENT full_public_transcript >> floor(~0.63), the emergent activation "
          "result is CONFOUNDED by visible discussion content (refuted as 'covert').")


def check_layers(root, model, tier):
    """Layer robustness: is the emergent signal stable across the extracted band or cherry-picked?"""
    for L in nb.npz_layers(f"{root}/activations/{model}/{tier}/activations_gen.npz"):
        sl, rows = _load(root, model, tier, L)
        es, ey, eg = _emergent(rows); ps, py, pg = _primary(rows)
        ea = stats.numeric_group_auroc(np.array([r["vec"] for r in es]), ey, eg, pca_dims=50)
        pa = stats.numeric_group_auroc(np.array([r["vec"] for r in ps]), py, pg, pca_dims=50)
        print(f"RESULT layers layer={L} | EMERGENT_act={ea:.3f} PRIMARY_act={pa:.3f}")
    print("VERDICT: emergent signal is robust only if act stays high across layers.")


def check_scenario_group(root, model, tier):
    """Group by scenario_id (not run_id): collusion & control share scenarios, so grouping by
    run_id could leak scenario content. Stricter grouping = does it survive?"""
    sl, rows = _load(root, model, tier, None)
    sub, y, _ = _emergent(rows)
    g_run = np.array([r["run_id"] for r in sub])
    g_scn = np.array([r["scenario_id"] for r in sub])
    X = np.array([r["vec"] for r in sub])
    a_run = stats.numeric_group_auroc(X, y, g_run, pca_dims=50)
    a_scn = stats.numeric_group_auroc(X, y, g_scn, pca_dims=50)
    fl_scn, _ = stats.permutation_floor(X, y, g_scn, pca_dims=50, n_perm=200)
    print(f"RESULT scenario_group | act_by_run={a_run:.3f} act_by_scenario={a_scn:.3f} floor_by_scenario={fl_scn:.3f} n={len(y)}")
    print("VERDICT: survives only if act_by_scenario stays well above floor_by_scenario.")


def check_nofilter(root, model, tier):
    """Behavioural filter removed: does the emergent signal depend on keeping only successful
    collusion runs (a selection effect)?"""
    sl = nb.load_slice(root, model, tier)
    rows = nb.agent_table(sl, channel="public", behavioral_filter=False)
    sub, y, g = _emergent(rows)
    X = np.array([r["vec"] for r in sub])
    a = stats.numeric_group_auroc(X, y, g, pca_dims=50)
    fl, _ = stats.permutation_floor(X, y, g, pca_dims=50, n_perm=200)
    print(f"RESULT nofilter | EMERGENT_act={a:.3f} floor={fl:.3f} n={len(y)}")
    print("VERDICT: robust if emergent act stays above floor without the behavioural filter.")


def check_degenerate(root, model, tier):
    """Explain the text=0.000 / prompt=0.000: degenerate (constant/empty) or genuine anti-signal?"""
    sl, rows = _load(root, model, tier, None)
    sub, y, g = _emergent(rows)
    from sklearn.feature_extraction.text import TfidfVectorizer
    for field in ("visible_text", "prompt_text"):
        texts = [r[field] for r in sub]
        lens = [len(t) for t in texts]
        try:
            vocab = len(TfidfVectorizer(min_df=2).fit([t for t in texts]).vocabulary_)
        except Exception as e:
            vocab = f"ERR {e}"
        print(f"RESULT degenerate field={field} | n={len(texts)} empties={sum(1 for t in texts if not t.strip())} "
              f"mean_len={np.mean(lens):.0f} tfidf_vocab={vocab} auroc={stats.text_group_auroc(texts, y, g):.3f}")
    print("VERDICT: 0.000 is a small-n extreme (perfectly anti-ranked on held-out groups), "
          "not a clean 0.5 floor - treat own-text as uninformative, lean on context_text instead.")


def check_ntokens(root, model, tier):
    """Output length (n_tokens) baseline - a behavioural correlate that is not coordination."""
    sl, rows = _load(root, model, tier, None)
    for nm, (sub, y, g) in [("EMERGENT", _emergent(rows)), ("PRIMARY", _primary(rows))]:
        X = np.array([[r["n_tokens"]] for r in sub])
        print(f"RESULT ntokens {nm} | ntokens_auroc={stats.numeric_group_auroc(X, y, g, pca_dims=0):.3f} n={len(y)}")
    print("VERDICT: confound if ntokens_auroc is well above floor.")


def check_primary_votematch(root, model, tier):
    """Can the PRIMARY (colluder vs honest) signal be salvaged by matching votes? Restrict honest
    to those who voted the TARGET (rare) vs colluders - removes the vote axis. Reports n."""
    sl, rows = _load(root, model, tier, None)
    cfg = sl["configs"]
    def voted_target(r):
        c = cfg.get(r["run_id"], {}) or {}
        return r.get("agent_vote") == c.get("target_option")
    col = [r for r in rows if r["role"] == "colluder"]
    hon_t = [r for r in rows if r["role"] == "honest" and voted_target(r)]
    print(f"RESULT primary_votematch | colluders={len(col)} honest_voted_target={len(hon_t)}")
    print("VERDICT: vote-matched PRIMARY is underpowered (few honest vote target); the clean "
          "test is EMERGENT, not a vote-matched colluder-vs-honest.")


CHECKS = {f.replace("check_", ""): globals()[f] for f in list(globals()) if f.startswith("check_")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=sorted(CHECKS))
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--model", default="gpt_oss_20b")
    ap.add_argument("--tier", default="core")
    a = ap.parse_args()
    CHECKS[a.check](a.root, a.model, a.tier)
