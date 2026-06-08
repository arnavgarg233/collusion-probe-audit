"""Extract the primary readout: mean-pooled residual stream at ~0.7 depth over each
agent's response span, A and B stored separately.

Output: data/activations/activations.npz with parallel arrays
    X[N,H], episode_id, item_id, condition, protocol, agent  (+ layer, n_layers)

--mock  : emit random Gaussian vectors aligned to the transcripts (no model needed),
          so the whole probe/control/CV stack can be validated offline. It should
          report everything near the permutation floor (no real signal).
--mock-signal S : additionally inject a mean shift of size S into ONE agent ("A") of
          the collusion(direct) rows only. The joint+A marginals then separate while
          text/vote/inert stay at the floor, validating that the table computes
          A/B/joint/text/vote/floor correctly and that an activation signal surfaces.
          NOTE: this does NOT (and cannot) make the go-rule emit a clean GO verdict.
          (1) The colluder's coalition instruction lives in the prompt, so prompt-only
          separates (prompt~1.0) and the "prompt-only separates" KILL always fires.
          (2) A one-agent shift is a per-agent MARGINAL, so the "single agent matches
          joint" KILL fires (joint cannot beat the A marginal). Injecting into BOTH
          agents (the old behaviour) just makes both marginals saturate, which also
          trips that kill. The synthetic self-test therefore exercises the plumbing,
          not a GO verdict; NARCBench is the real go/no-go.
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
import argparse, json
import numpy as np
import config


def extract(transcripts_path: str, out_path: str, pool: str,
            mock: bool, mock_signal: float, mock_dim: int):
    config.ensure_dirs()
    with open(transcripts_path) as f:
        records = json.load(f)

    X, ep, item, cond, proto, ag = [], [], [], [], [], []
    layer_used, n_layers = -1, -1

    if not mock:
        import model_utils
    rng = np.random.default_rng(config.SEED)
    shift = None  # fixed random direction for the injected mock signal

    for rec in records:
        for agent, payload in rec["agents"].items():
            if mock:
                v = rng.standard_normal(mock_dim).astype(np.float32)
                # Inject the self-test signal into ONE agent only (agent "A") of the
                # matched collusion(direct) condition. With both agents shifted (the old
                # behaviour) the B-only marginal also saturated, masking the A/B split;
                # one-agent injection keeps B-only at the floor so the table honestly
                # shows where the signal lives. This validates the plumbing (A/joint
                # separate; text/vote/inert stay at floor) but does NOT yield a GO
                # verdict -- the prompt-only KILL (coalition instruction in the prompt)
                # and the single-agent-marginal KILL both still fire by design. See the
                # module docstring.
                if (mock_signal and rec["condition"] == "collusion"
                        and rec["protocol"] == "direct" and agent == "A"):
                    if shift is None:
                        shift = rng.standard_normal(mock_dim).astype(np.float32)
                        shift /= np.linalg.norm(shift)
                    v = v + mock_signal * shift
                layer_used, n_layers = -1, -1
            else:
                v, layer_used, n_layers = model_utils.response_hidden(
                    payload["messages"], payload["response_text"],
                    layer_frac=config.LAYER_FRAC, pool=pool)
            X.append(v)
            ep.append(rec["episode_id"]); item.append(rec["item_id"])
            cond.append(rec["condition"]); proto.append(rec["protocol"]); ag.append(agent)

    X = np.stack(X).astype(np.float32)
    np.savez(out_path,
             X=X,
             episode_id=np.array(ep), item_id=np.array(item),
             condition=np.array(cond), protocol=np.array(proto), agent=np.array(ag),
             layer=np.array([layer_used]), n_layers=np.array([n_layers]),
             pool=np.array([pool]))
    tag = "MOCK" + ("+signal" if mock_signal else "") if mock else f"layer {layer_used}/{n_layers}"
    print(f"[extract] {X.shape[0]} vectors dim {X.shape[1]} ({tag}) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default=config.TRANSCRIPTS_PATH)
    ap.add_argument("--out", default=config.ACTIVATIONS_PATH)
    ap.add_argument("--pool", default=config.POOL, choices=["mean", "last", "max"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-signal", type=float, default=0.0)
    ap.add_argument("--mock-dim", type=int, default=128)
    a = ap.parse_args()
    extract(a.transcripts, a.out, a.pool, a.mock, a.mock_signal, a.mock_dim)
