"""CREATIVE flagship: cross-agent COORDINATION signature (relational, not per-agent).

Every prior probe/baseline asked "is THIS agent a colluder" (per-agent). Collusion is
relational. Test: within a collusion run, are the two colluders' decision-span activations
more ALIGNED with each other than the two honest agents' are - BEYOND their visible-text
similarity? That is a beyond-visible test of coordination itself, which no per-agent probe
can capture.

Per collusion run -> mean pairwise cosine among colluders vs among honest, for ACTIVATIONS
and for TF-IDF TEXT. Report: can alignment separate colluder-pairs from honest-pairs (AUROC),
for activations vs text, and the paired within-run gap with a sign-permutation p-value.
"""
# --- path bootstrap: make sibling src/ subpackages importable as flat modules ---
import os as _os, sys as _sys
_SRCROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('', 'common', 'synthetic', 'narcbench', 'audit', 'exploratory'):
    _pp = _os.path.join(_SRCROOT, _sub)
    if _os.path.isdir(_pp) and _pp not in _sys.path:
        _sys.path.insert(0, _pp)
# --- end bootstrap ---
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import sys; sys.path.insert(0, "src")
import numpy as np
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
import narcbench_loader as nb
ROOT = "/tmp/narc/slice"


def _cos(a, b):
    na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb_)) if na > 0 and nb_ > 0 else 0.0


def _mean_pairwise(vecs):
    if len(vecs) < 2:
        return None
    return float(np.mean([_cos(vecs[i], vecs[j]) for i, j in combinations(range(len(vecs)), 2)]))


def run(model, tier):
    sl = nb.load_slice(ROOT, model, tier)
    rows = nb.agent_table(sl, channel="public", behavioral_filter=False)
    # TF-IDF over all visible texts for the text-alignment control
    texts = [r["visible_text"] for r in rows]
    have_text = sum(len(t) > 5 for t in texts) > 0.5 * len(texts)
    if have_text:
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(texts)
        for r in rows:
            r["_tv"] = tf.transform([r["visible_text"]]).toarray()[0]

    runs = {}
    for r in rows:
        if r["mode"] != "collusion":
            continue
        runs.setdefault(r["run_id"], {"colluder": [], "honest": []}).setdefault(r["role"], []).append(r)

    act_col, act_hon, txt_col, txt_hon = [], [], [], []
    for rid, grp in runs.items():
        c, h = grp.get("colluder", []), grp.get("honest", [])
        if len(c) < 2 or len(h) < 2:
            continue
        ac = _mean_pairwise([r["vec"] for r in c]); ah = _mean_pairwise([r["vec"] for r in h])
        act_col.append(ac); act_hon.append(ah)
        if have_text:
            tc = _mean_pairwise([r["_tv"] for r in c]); th = _mean_pairwise([r["_tv"] for r in h])
            txt_col.append(tc); txt_hon.append(th)

    n = len(act_col)
    if n < 4:
        print(f"RESULT coord {model}/{tier} | too few runs with 2+2 (n={n})"); return
    # AUROC: separate colluder-pair (1) from honest-pair (0) by alignment
    y = np.array([1] * n + [0] * n)
    act_auroc = roc_auc_score(y, np.array(act_col + act_hon))
    # paired within-run gap + sign-permutation p
    gap = np.array(act_col) - np.array(act_hon)
    rng = np.random.default_rng(0)
    null = np.array([np.mean(rng.choice([-1, 1], n) * np.abs(gap)) for _ in range(2000)])
    p = float(np.mean(null >= gap.mean()))
    line = (f"RESULT coord {model}/{tier} | runs={n} act_align_AUROC={act_auroc:.3f} "
            f"mean_gap(col-hon)={gap.mean():+.3f} sign_perm_p={p:.3f}")
    if have_text:
        ty = roc_auc_score(y, np.array(txt_col + txt_hon))
        line += f" | text_align_AUROC={ty:.3f} delta(act-text)={act_auroc - ty:+.3f}"
    print(line)
    print(f"  read: coordination-beyond-text only if act_align_AUROC clearly > text_align_AUROC "
          f"AND > 0.5 with small p (n={n} is small -> treat as exploratory)")


if __name__ == "__main__":
    for m, t in [("gpt_oss_20b", "core"), ("qwen3_32b", "core"),
                 ("gpt_oss_20b", "stego"), ("qwen3_32b", "stego")]:
        try:
            run(m, t)
        except Exception as e:
            print(f"RESULT coord {m}/{t} | ERROR {e}")
    print("DONE")
