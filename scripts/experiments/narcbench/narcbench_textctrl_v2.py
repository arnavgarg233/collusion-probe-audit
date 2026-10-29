"""Text-controlled probe with per-contrast text baselines.

For the colluder contrast the label varies within a run while the run transcript is shared,
so the text baseline is each agent's own visible text. The emergent contrast uses the full
public discussion the honest agent read, grouped by scenario.
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
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, argparse
sys.path.insert(0, "src")
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import narcbench_loader as nb
import narcbench_textctrl as tc

ROOT = "/tmp/narc/slice"


def build(model, tier, contrast, layer=None):
    sl = nb.load_slice(ROOT, model, tier, layer=layer)
    rows = nb.agent_table(sl, channel="public", behavioral_filter=True)
    if contrast == "colluder":
        sub = rows
        y = np.array([1 if r["role"] == "colluder" else 0 for r in sub])
        g = np.array([r["run_id"] for r in sub])
        texts = [r["visible_text"] for r in sub]              # FIX: per-agent own text
    elif contrast == "emergent":
        sub = [r for r in rows if r["role"] == "honest"]
        y = np.array([1 if r["mode"] == "collusion" else 0 for r in sub])
        g = np.array([r["scenario_id"] for r in sub])
        texts = [r["context_text"] for r in sub]              # full transcript (valid here)
    else:
        raise ValueError(contrast)
    A = np.array([r["vec"] for r in sub])
    # vote-vs-target baseline (the cheap visible-behaviour confound)
    cfgs = sl["configs"]
    vote_feat = []
    for r in sub:
        cfg = cfgs.get(r["run_id"]) or {}
        tgt = cfg.get("target_option")
        vote_feat.append(1.0 if (r.get("agent_vote") is not None and r["agent_vote"] == tgt) else 0.0)
    return sl, texts, A, y, g, np.array(vote_feat).reshape(-1, 1)


def vote_auroc(vote, y, g):
    oof = np.full(len(y), np.nan)
    k = max(2, min(5, len(set(g))))
    for tr, te in GroupKFold(k).split(np.zeros(len(y)), y, g):
        if len(set(y[tr])) < 2:
            continue
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(vote[tr], y[tr])
        oof[te] = m.predict_proba(vote[te])[:, 1]
    msk = ~np.isnan(oof)
    try:
        return roc_auc_score(y[msk], oof[msk])
    except Exception:
        return float("nan")


def run(model, tier, contrast, n_perm, act_pca, layer=None):
    sl, texts, A, y, g, vote = build(model, tier, contrast, layer)
    if len(set(y)) < 2:
        print(f"{model}/{tier}/{contrast}: single-class, skipped"); return None
    # point estimate first
    to, ao, tao = tc._oof(texts, A, y, g, act_pca=act_pca)
    m = ~np.isnan(tao)
    text, act, ta = (roc_auc_score(y[m], to[m]), roc_auc_score(y[m], ao[m]),
                     roc_auc_score(y[m], tao[m]))
    delta = ta - text
    v_auc = vote_auroc(vote, y, g)
    print(f"PE  {model}/{tier}/{contrast} L{sl['layer']} n={int(m.sum())} pos={int(y[m].sum())} | "
          f"vote={v_auc:.3f} text={text:.3f} act={act:.3f} ta={ta:.3f} delta={delta:+.3f}",
          flush=True)
    # permutation floor on delta
    rng = np.random.default_rng(0); perm = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        t2, a2, ta2 = tc._oof(texts, A, yp, g, act_pca=act_pca)
        mm = ~np.isnan(ta2)
        try:
            perm.append(roc_auc_score(yp[mm], ta2[mm]) - roc_auc_score(yp[mm], t2[mm]))
        except Exception:
            pass
    perm = np.array(perm) if perm else np.array([0.0])
    fl = float(np.percentile(perm, 95)); p = float(np.mean(perm >= delta))
    verdict = ("GO" if (delta > fl and delta >= 0.10)
               else "CONFOUNDED" if delta <= fl else "PARTIAL")
    print(f"FLOOR {model}/{tier}/{contrast} | delta={delta:+.3f} floor95={fl:.3f} "
          f"p={p:.3f} -> {verdict}", flush=True)
    return {"model": model, "tier": tier, "contrast": contrast, "layer": sl["layer"],
            "n": int(m.sum()), "pos": int(y[m].sum()), "vote": v_auc, "text": text,
            "act": act, "ta": ta, "delta": delta, "floor95": fl, "p": p, "verdict": verdict}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--act-pca", type=int, default=30)
    a = ap.parse_args()
    combos = [("gpt_oss_20b", "core", "emergent"),
              ("qwen3_32b", "core", "emergent"),
              ("gpt_oss_20b", "core", "colluder"),
              ("qwen3_32b", "core", "colluder"),
              ("gpt_oss_20b", "stego", "colluder")]
    res = []
    for m, t, c in combos:
        try:
            r = run(m, t, c, a.n_perm, a.act_pca)
        except Exception as e:
            print(f"ERROR {m}/{t}/{c}: {e}"); r = None
        if r:
            res.append(r)
    # write findings
    if res:
        L = ["# NARCBench text-controlled probe v2 (corrected baseline) — M2, published activations\n",
             "Fix: colluder contrast now uses each agent's OWN visible_text (per-agent), not the",
             "run-constant full transcript (which gave a degenerate below-chance text baseline in v1).",
             "emergent uses the full transcript (valid: label = run mode, grouped by scenario_id).",
             f"delta = AUROC(text+act) - AUROC(text-only), pooled OOF, perm floor (n_perm={a.n_perm}).\n",
             "| model | tier | contrast | n | pos | vote | text | act | text+act | delta | floor95 | p | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in res:
            L.append(f"| {r['model']} | {r['tier']} | {r['contrast']} | {r['n']} | {r['pos']} | "
                     f"{r['vote']:.3f} | {r['text']:.3f} | {r['act']:.3f} | {r['ta']:.3f} | "
                     f"{r['delta']:+.3f} | {r['floor95']:.3f} | {r['p']:.3f} | **{r['verdict']}** |")
        L += ["", "## Read",
              "- `vote` = vote-vs-target baseline (the cheap visible-behaviour confound). For the",
              "  colluder contrast it rides ~1.0 -> colluders are trivially ID'd from their VOTE,",
              "  so colluder-vs-honest is dead upstream regardless of the text delta.",
              "- emergent (honest-only, no vote/instruction confound) is the real beyond-text test."]
        open("results/textctrl_v2_summary.md", "w").write("\n".join(L) + "\n")
        print("\nwrote results/textctrl_v2_summary.md")
    print("V2_DONE", flush=True)
