"""Driver: linear-vs-nonlinear text-controlled delta on gpt_oss_20b (core, stego).

One-off runner (not imported by anything). For each tier x contrast it prints the
visible baselines, the linear delta (reference), and the MLP/SVC nonlinear deltas,
flagging whether any nonlinear activation probe beats the visible baseline AND its
permutation floor. EXPLORATORY — not load-bearing.
"""
# --- path bootstrap: make sibling src/ subpackages importable as flat modules ---
import os as _os, sys as _sys
_SRCROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('', 'common', 'synthetic', 'narcbench', 'audit', 'exploratory'):
    _pp = _os.path.join(_SRCROOT, _sub)
    if _os.path.isdir(_pp) and _pp not in _sys.path:
        _sys.path.insert(0, _pp)
# --- end bootstrap ---
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import narcbench_loader as nb, stats
import narcbench_textctrl as tc
import narcbench_nonlinear_delta as nl

ROOT = '/tmp/narc/slice'; MODEL = 'gpt_oss_20b'
NPERM = 40


def build(tier, contrast):
    sl = nb.load_slice(ROOT, MODEL, tier)
    rows = nb.agent_table(sl, channel='public', behavioral_filter=False)
    if contrast == 'colluder':
        sub = rows
        y = np.array([1 if r['role'] == 'colluder' else 0 for r in sub]); grp = 'run_id'
    else:  # emergent: honest collusion-vs-control, matched prompt
        sub = [r for r in rows if r['role'] == 'honest']
        y = np.array([1 if r['mode'] == 'collusion' else 0 for r in sub]); grp = 'scenario_id'
    g = np.array([r[grp] for r in sub])
    texts = [r['context_text'] for r in sub]
    A = np.array([r['vec'] for r in sub])
    # behavioral visible baseline features
    beh = []
    for r in sub:
        sl_cfg = sl['configs'].get(r['run_id']) or {}
        tgt = None; fair = None
        # target/fair come from meta rows; pull from any meta row of this run
        beh.append([
            float(r.get('agent_vote') == _target(sl, r['run_id'])),
            float(r.get('agent_vote') == _fair(sl, r['run_id'])),
            float(r.get('n_tokens') or 0.0),
            float(r.get('bp_corr') or 0.0),
        ])
    B = np.array(beh, dtype=float)
    return texts, A, y, g, B


def _target(sl, rid):
    for m in sl['meta']:
        if m.get('run_id') == rid and m.get('target_option') is not None:
            return m.get('target_option')
    return None


def _fair(sl, rid):
    for m in sl['meta']:
        if m.get('run_id') == rid and m.get('fair_option') is not None:
            return m.get('fair_option')
    return None


for tier in ['core', 'stego']:
    for contrast in ['colluder', 'emergent']:
        texts, A, y, g, B = build(tier, contrast)
        n, pos, ng = len(y), int(y.sum()), len(set(g))
        # ---- VISIBLE BASELINES ----
        text_base = stats.text_group_auroc(texts, y, g)
        beh_base = stats.numeric_group_auroc(B, y, g, pca_dims=0)
        print(f"\n##### {tier}/{contrast}  n={n} pos={pos} groups={ng}", flush=True)
        print(f"  VISIBLE: text_base={text_base:.3f}  behavioral_base={beh_base:.3f}", flush=True)
        # ---- LINEAR delta (reference) ----
        rl = tc.delta_test(texts, A, y, g, act_pca=30, n_perm=NPERM)
        print(f"  LINEAR : text={rl['text']:.3f} act={rl['act']:.3f} ta={rl['ta']:.3f} "
              f"delta={rl['delta']:+.3f} floor95={rl['delta_floor95']:.3f} p={rl['delta_p']:.3f}", flush=True)
        # ---- NONLINEAR deltas ----
        for kind in ['mlp', 'svc']:
            rn = nl.delta_test(texts, A, y, g, act_pca=30, kind=kind, n_perm=NPERM)
            beats_vis = (rn['ta'] > max(text_base, beh_base)) and (rn['delta'] > rn['delta_floor95'])
            print(f"  {kind.upper():6s}: text={rn['text']:.3f} act={rn['act']:.3f} ta={rn['ta']:.3f} "
                  f"delta={rn['delta']:+.3f} floor95={rn['delta_floor95']:.3f} p={rn['delta_p']:.3f} "
                  f"| beats_visible+floor={beats_vis}", flush=True)
print("\n=== ALL DONE ===", flush=True)
