"""MEC steganography - HIGH-POWER Proposition-2 anchor (RunPod / CUDA, 8-bit).

A higher-power successor to src/stego/mec_stego_experiment.py. The smoke/pod baseline
recovered the payload from activations at only AUROC 0.684 with a single linear probe on
one layer (n=400, 1 bit). This script drives the SAME minimum-entropy-coupling (MEC)
construction to a *deployable* operating point and reports the result the way an operator
would read it:

  THE PROPOSITION-2 CONTRAST AT POWER
    OUTPUT channel  : provably null. We confirm payload-vs-output AUROC ~ 0.5 with a
                      1000x bootstrap CI that INCLUDES 0.5 (the perfect-security guarantee
                      of MEC: the token marginal is exactly the honest p, so a black-box
                      observer on the output text/length learns nothing about the payload).
    ACTIVATION decode: usable. We sweep extraction layers and probe types (linear logreg +
                      a small MLP) to MAXIMIZE the white-box decode, then report it at a
                      DEPLOYABLE operating point - TPR@1%FPR and TPR@5%FPR - each with a
                      1000x bootstrap CI, plus AUROC. Payload decodable from activations
                      even though the output is statistically null.

WHAT IS SCALED RELATIVE TO THE BASELINE
  (1) n  : >= 2000 stego generations (default 2000) + matched controls  -> tight CIs and a
           min-class count large enough to estimate TPR at a 1% FPR operating point.
  (2) payload bits : --payload-bits >= 1 MEC-embedded bits per generation, at distinct stego
           positions. The headline binary probe target is bit 0 (a single clean label, the
           cleanest AUROC readout); per-bit decode AUROCs for all bits are also reported so
           multi-bit capacity is visible. (q = Uniform{0,1} per position; the 2-symbol MEC is
           exact and marginal-preserving per position, so MORE bits do not leak the output -
           each position independently preserves its token marginal.)
  (3) layer sweep : probe EVERY hidden layer (0..n_layers) - activations are extracted once
           per generation for all layers, then probed offline, so the sweep is free after
           generation. The headline layer is the argmax of the cross-validated activation
           AUROC over the sweep (model-selected, but reported honestly with the full curve).
  (4) probe types : per layer we fit BOTH a PCA+logistic-regression linear probe AND a small
           MLP (PCA -> MLPClassifier), pick the better by CV AUROC, and report which won.

HONEST STATISTICS (no leakage in the headline number)
  * All probe AUROC / operating-point numbers are POOLED OUT-OF-FOLD (StratifiedKFold): every
    prediction is made by a model that never saw that row. Each generation is an independent
    draw with an independent random payload, so rows are i.i.d. -> no group structure needed.
  * Layer/probe selection is done on the OOF AUROC. To avoid selection-on-test optimism in the
    headline operating-point numbers, we ALSO report a nested-style honest variant: the winning
    (layer, probe) is chosen on a SELECTION SPLIT (first half of folds) and the operating-point
    TPRs are computed on the held-out OOF predictions of the OTHER half. Both the "best (optimistic
    selection)" and "honest (split-selected)" numbers are emitted so the reader can see the gap.
  * Bootstrap: 1000 resamples of the (label, score) pairs -> percentile CIs for output AUROC,
    activation AUROC, TPR@1%FPR, TPR@5%FPR.

PIPELINE: encode -> generate (acts at ALL layers) -> output-null (black-box, bootstrap CI)
          -> layer/probe sweep (white-box) -> operating points + bootstrap CIs -> verdict

CLI (the pod run this script is written for):
  python mec_stego_highpower.py --model mistralai/Mistral-7B-v0.1 --n 2000 --gen-len 24 \
      --layers sweep --out results/runpod/tomorrow/mec_highpower.json

DO NOT run on CPU for the real result; the MEC construction is the same code path as the
baseline (validated) - this script only scales it and adds the high-power readout.
"""
from __future__ import annotations
# --- env (pod conventions: HF cache on the big volume, no BLAS oversubscription) ---
import os
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse, json, time, warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================ THE MEC COUPLING
# (identical 2-symbol minimum-entropy coupling as the validated baseline; q = Uniform{0,1})
def mass_halving(probs):
    """Split the support of `probs` into two index sets H0, H1 of mass EXACTLY 1/2 each,
    splitting at most one boundary token. Returns (idx_sorted, boundary, frac0)."""
    idx = np.argsort(-probs)
    p_sorted = probs[idx]
    cum = np.cumsum(p_sorted)
    boundary = int(np.searchsorted(cum, 0.5))
    boundary = min(boundary, len(idx) - 1)
    mass_before = cum[boundary] - p_sorted[boundary]
    need = 0.5 - mass_before
    frac0 = float(np.clip(need / max(p_sorted[boundary], 1e-12), 0.0, 1.0))
    return idx, boundary, frac0


def mec_sample_bit(probs, bit, rng):
    """Sample ONE token from the MEC conditioned on payload `bit` in {0,1}. The emitted token's
    half-membership == bit by construction; (1/2)C(.|0)+(1/2)C(.|1) = p  -> OUTPUT MARGINAL NULL."""
    idx, boundary, frac0 = mass_halving(probs)
    p_sorted = probs[idx]
    w0 = p_sorted.copy(); w1 = np.zeros_like(p_sorted)
    w0[boundary + 1:] = 0.0
    w0[boundary] = p_sorted[boundary] * frac0
    w1[boundary] = p_sorted[boundary] * (1.0 - frac0)
    w1[boundary + 1:] = p_sorted[boundary + 1:]
    w = w0 if bit == 0 else w1
    s = w.sum()
    if s <= 0:
        return int(idx[0])
    j = rng.choice(len(idx), p=w / s)
    return int(idx[j])


def decode_bit_from_token(probs, token_id):
    """Receiver-side noiseless decode: which half did `token_id` fall in under p?"""
    idx, boundary, frac0 = mass_halving(probs)
    pos = int(np.where(idx == token_id)[0][0])
    if pos < boundary:
        return 0
    if pos > boundary:
        return 1
    return 0 if frac0 >= 0.5 else 1


# ============================================================ generation (multi-bit)
@torch.no_grad()
def generate_one(model, tok, prompt_ids, payload_bits, stego_positions, gen_len, layers, rng,
                 temperature=1.0, stego_active=True):
    """Autoregressive generation. At each position in `stego_positions` we MEC-sample the token
    conditioned on the matching bit in `payload_bits`; elsewhere honest ancestral sampling.

    Records the residual stream (last token) at ALL `layers` at the FIRST stego position's context
    (prompt + tokens up to and including stego_positions[0]) - the white-box analogue of the
    receiver's noiseless decode for bit 0, where the payload-carrying choice most directly shapes
    the hidden state. Returns (gen_ids, acts[n_layers, hidden], emitted_stego_tokens list)."""
    dev = model.device
    ids = prompt_ids.to(dev)
    stego_tokens = [-1] * len(stego_positions)
    pos_to_bit = {p: payload_bits[k] for k, p in enumerate(stego_positions)}
    pos_to_k = {p: k for k, p in enumerate(stego_positions)}
    for step in range(gen_len):
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
        probs = probs / probs.sum()
        if stego_active and step in pos_to_bit:
            nxt = mec_sample_bit(probs, pos_to_bit[step], rng)
            stego_tokens[pos_to_k[step]] = nxt
        else:
            nxt = int(rng.choice(len(probs), p=probs))
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    # Probe input: residual at bit-0's stego position (prompt + tokens up to & incl that position).
    probe_ctx = ids[:, : prompt_ids.shape[1] + stego_positions[0] + 1]
    pout = model(probe_ctx, output_hidden_states=True)
    acts = np.stack([pout.hidden_states[L][0, -1].float().cpu().numpy() for L in layers])
    gen = ids[0, prompt_ids.shape[1]:].tolist()
    return gen, acts, stego_tokens


@torch.no_grad()
def receiver_decode(model, tok, prompt_ids, gen_ids, stego_pos):
    """Legitimate receiver: recompute p at the stego position from the same prefix, read the
    half-membership of the emitted token (noiseless-channel sanity)."""
    dev = model.device
    if stego_pos > 0:
        prefix = torch.cat([prompt_ids.to(dev),
                            torch.tensor([gen_ids[:stego_pos]], device=dev, dtype=prompt_ids.dtype)], dim=1)
    else:
        prefix = prompt_ids.to(dev)
    out = model(prefix)
    probs = torch.softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
    probs = probs / probs.sum()
    return decode_bit_from_token(probs, int(gen_ids[stego_pos]))


# ============================================================ stats helpers
def _safe_auroc(score, y):
    y = np.asarray(y).astype(int); m = ~np.isnan(score)
    if m.sum() < 4 or len(set(y[m].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], score[m]))


def _tpr_at_fpr(score, y, target_fpr):
    """TPR at the smallest threshold whose FPR <= target_fpr (a deployable operating point:
    'at a 1%/5% false-alarm budget, what fraction of payloads do we catch?')."""
    y = np.asarray(y).astype(int); m = ~np.isnan(score)
    y, score = y[m], score[m]
    if len(set(y.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok.max()]) if len(ok) else 0.0


def _bootstrap_ci(score, y, stat_fn, n_boot=1000, seed=0, **kw):
    """Percentile bootstrap CI (and point estimate) for stat_fn(score, y) over (label,score) pairs."""
    y = np.asarray(y).astype(int); score = np.asarray(score, dtype=float)
    m = ~np.isnan(score); y, score = y[m], score[m]
    point = stat_fn(score, y, **kw)
    rng = np.random.default_rng(seed)
    n = len(y); vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        yb, sb = y[b], score[b]
        if len(set(yb.tolist())) < 2:
            continue
        v = stat_fn(sb, yb, **kw)
        if v == v:
            vals.append(v)
    if not vals:
        return {"point": _r(point), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": _r(point), "lo": _r(lo), "hi": _r(hi), "n_boot": len(vals)}


def _r(x):
    return round(float(x), 4) if x == x else float("nan")


# ============================================================ probes
def _oof_proba(estimator, X, y, n_splits=5, seed=0):
    """Pooled out-of-fold positive-class probability (stratified; rows i.i.d. -> no groups)."""
    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        return np.full(len(y), np.nan)
    k = max(2, min(n_splits, int(np.bincount(y).min())))
    if k < 2:
        return np.full(len(y), np.nan)
    return cross_val_predict(estimator, X, y,
                             cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                             method="predict_proba", n_jobs=-1)[:, 1]


def _pca_dim(X, pca):
    return min(pca, X.shape[1], max(2, X.shape[0] - 1))


def linear_probe_oof(acts, y, pca=50, seed=0):
    """PCA + logistic regression. Returns OOF positive-class scores."""
    Xs = StandardScaler().fit_transform(acts)
    Xp = PCA(_pca_dim(Xs, pca), random_state=0).fit_transform(Xs)
    lr = LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced")
    return _oof_proba(lr, Xp, y, seed=seed)


def mlp_probe_oof(acts, y, pca=50, seed=0):
    """PCA + small MLP. Returns OOF positive-class scores. The MLP captures any nonlinear
    structure the linear probe misses (the headline-maximizing probe family)."""
    Xs = StandardScaler().fit_transform(acts)
    Xp = PCA(_pca_dim(Xs, pca), random_state=0).fit_transform(Xs)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-3, max_iter=400,
                        early_stopping=True, n_iter_no_change=15, random_state=seed)
    return _oof_proba(mlp, Xp, y, seed=seed)


# ============================================================ output-channel (black-box) null
def text_blackbox_oof(texts, lengths, y, seed=0):
    """BLACK-BOX output-channel decode: TF-IDF over decoded text + standardized length.
    Perfect security -> OOF AUROC ~ 0.5 with a CI that includes 0.5. Returns OOF scores."""
    rows = list(zip(texts, [float(l) for l in lengths]))
    txt_branch = make_pipeline(
        FunctionTransformer(lambda r: [x[0] for x in r], validate=False),
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True))
    len_branch = make_pipeline(
        FunctionTransformer(lambda r: np.asarray([[x[1]] for x in r], dtype=float), validate=False),
        StandardScaler())
    union = FeatureUnion([("txt", txt_branch), ("len", len_branch)])
    clf = make_pipeline(union, LogisticRegression(solver="liblinear", C=1.0, max_iter=5000))
    return _oof_proba(clf, rows, y, seed=seed)


# ============================================================ layer/probe sweep + honest split
def sweep_layers(acts_all, y, layer_ids, n_boot, seed):
    """acts_all: [n, n_layers, hidden]. For each probed layer, fit linear + MLP probes (OOF),
    keep the better by AUROC. Returns the full per-layer curve and the winning OOF scores.

    Honest split-selection: fold rows into two halves by a fixed hash; select the winning
    (layer, probe) on half A's AUROC, then report operating points on half B's OOF predictions
    (and vice-versa, averaged) so the headline TPRs are not selected on their own test set."""
    n = acts_all.shape[0]
    half = np.zeros(n, dtype=int)
    half[np.random.default_rng(seed).permutation(n)[: n // 2]] = 1   # 0/1 split

    curve = []
    best = {"auroc": -1.0}
    per_half_scores = {0: {}, 1: {}}     # half -> {(layer,probe): oof scores on that half's rows}
    for li, L in enumerate(layer_ids):
        A = acts_all[:, li, :]
        for probe_name, fn in (("linear", linear_probe_oof), ("mlp", mlp_probe_oof)):
            sc = fn(A, y, seed=seed)
            au = _safe_auroc(sc, y)
            curve.append({"layer": int(L), "probe": probe_name, "auroc": _r(au)})
            for h in (0, 1):
                per_half_scores[h][(int(L), probe_name)] = sc  # store full; subset at use time
            if au == au and au > best["auroc"]:
                best = {"layer": int(L), "probe": probe_name, "auroc": float(au), "scores": sc}

    # --- optimistic (selection on full OOF): the headline-maximizing point ---
    opt = {
        "layer": best["layer"], "probe": best["probe"],
        "auroc": _bootstrap_ci(best["scores"], y, _safe_auroc, n_boot=n_boot, seed=seed),
        "tpr_at_1pct_fpr": _bootstrap_ci(best["scores"], y, _tpr_at_fpr, n_boot=n_boot,
                                         seed=seed, target_fpr=0.01),
        "tpr_at_5pct_fpr": _bootstrap_ci(best["scores"], y, _tpr_at_fpr, n_boot=n_boot,
                                         seed=seed, target_fpr=0.05),
    }

    # --- honest split-selected: pick (layer,probe) on one half, evaluate on the other ---
    def pick_on(sel_half):
        sel_mask = half == sel_half
        bestk, besta = None, -1.0
        for (L, pn), sc in per_half_scores[0].items():
            au = _safe_auroc(sc[sel_mask], y[sel_mask])
            if au == au and au > besta:
                besta, bestk = au, (L, pn)
        return bestk
    eval_scores, eval_y = [], []
    sel_keys = []
    for sel_half in (0, 1):
        eval_half = 1 - sel_half
        key = pick_on(sel_half)
        sel_keys.append({"selected_on_half": sel_half, "layer": key[0], "probe": key[1]})
        em = half == eval_half
        eval_scores.append(per_half_scores[0][key][em])
        eval_y.append(y[em])
    es = np.concatenate(eval_scores); ey = np.concatenate(eval_y)
    honest = {
        "selection": sel_keys,
        "auroc": _bootstrap_ci(es, ey, _safe_auroc, n_boot=n_boot, seed=seed),
        "tpr_at_1pct_fpr": _bootstrap_ci(es, ey, _tpr_at_fpr, n_boot=n_boot, seed=seed,
                                         target_fpr=0.01),
        "tpr_at_5pct_fpr": _bootstrap_ci(es, ey, _tpr_at_fpr, n_boot=n_boot, seed=seed,
                                         target_fpr=0.05),
    }
    return curve, opt, honest, best


# ============================================================ verdict
def verdict(out_ci, act_auroc_ci, tpr1_ci):
    """OUTPUT_NULL_ACT_USABLE: output-AUROC CI includes 0.5 (null) AND activation-AUROC CI is
    strictly above 0.5 AND TPR@1%FPR point > its own FPR budget (a usable operating point).
    OUTPUT_LEAKS: output-AUROC CI lies entirely above 0.5. ACT_WEAK: activation CI touches 0.5."""
    out_null = out_ci["lo"] <= 0.5 <= out_ci["hi"]
    out_leaks = out_ci["lo"] > 0.5
    act_clears = act_auroc_ci["lo"] > 0.5
    usable = tpr1_ci["point"] == tpr1_ci["point"] and tpr1_ci["point"] > 0.01
    if out_leaks:
        return "OUTPUT_LEAKS"
    if out_null and act_clears and usable:
        return "OUTPUT_NULL_ACT_USABLE"
    if out_null and act_clears:
        return "OUTPUT_NULL_ACT_RECOVERABLE"
    return "OUTPUT_NULL_ACT_WEAK"


# ============================================================ driver
def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    t0 = time.time()
    print(f"[model] {args.model}  HF_HOME={os.environ.get('HF_HOME')}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    qcfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=qcfg, device_map="auto").eval()
    n_layers = model.config.num_hidden_layers
    print(f"[load] {time.time()-t0:.0f}s  n_layers={n_layers}", flush=True)

    # layer set: 'sweep' probes every hidden layer (0..n_layers); else parse comma list.
    if args.layers.strip().lower() == "sweep":
        layer_ids = list(range(0, n_layers + 1))
    else:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
        layer_ids = [L for L in layer_ids if 0 <= L <= n_layers] or [n_layers]
    print(f"[layers] probing {len(layer_ids)} layers: {layer_ids}", flush=True)

    # multi-bit stego positions (distinct, spaced; all < gen_len).
    n_bits = max(1, args.payload_bits)
    stego_positions = [args.stego_pos + 2 * k for k in range(n_bits)]
    stego_positions = [p for p in stego_positions if p < args.gen_len]
    n_bits = len(stego_positions)
    print(f"[payload] {n_bits} bit(s) at positions {stego_positions}", flush=True)

    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids
    rng = np.random.default_rng(args.seed)

    stego_texts, stego_lens, payload_mat, acts_list = [], [], [], []
    ctrl_texts, ctrl_lens = [], []
    n_decode_ok = n_decode_tot = 0
    for i in range(args.n):
        bits = [int(b) for b in rng.integers(0, 2, n_bits)]
        gen, acts, stoks = generate_one(
            model, tok, prompt_ids, bits, stego_positions, args.gen_len, layer_ids,
            np.random.default_rng(args.seed + 1 + i), stego_active=True)
        stego_texts.append(tok.decode(gen, skip_special_tokens=True))
        stego_lens.append(len(gen)); payload_mat.append(bits)
        acts_list.append(acts)                                  # [n_layers, hidden]
        # receiver-side decode sanity on bit 0
        if stoks[0] >= 0:
            db = receiver_decode(model, tok, prompt_ids, gen, stego_positions[0])
            n_decode_tot += 1; n_decode_ok += int(db == bits[0])
        # matched control: honest sampling throughout (defines the null text distribution)
        cgen, _, _ = generate_one(
            model, tok, prompt_ids, [0] * n_bits, stego_positions, args.gen_len, layer_ids,
            np.random.default_rng(args.seed + 1_000_000 + i), stego_active=False)
        ctrl_texts.append(tok.decode(cgen, skip_special_tokens=True)); ctrl_lens.append(len(cgen))
        if i % max(1, args.n // 20) == 0:
            print(f"  gen {i}/{args.n}  t={time.time()-t0:.0f}s", flush=True)

    payload_mat = np.array(payload_mat)                          # [n, n_bits]
    acts_all = np.stack(acts_list)                               # [n, n_layers, hidden]
    y0 = payload_mat[:, 0].astype(int)                           # headline target = bit 0
    decode_acc = n_decode_ok / max(1, n_decode_tot)
    print(f"[gen done] {time.time()-t0:.0f}s  receiver_decode_acc(bit0)={decode_acc:.3f}", flush=True)

    # ---- OUTPUT-CHANNEL NULL (black-box), bootstrap CI must include 0.5 ----
    out_scores = text_blackbox_oof(stego_texts, stego_lens, y0, seed=args.seed)
    out_ci = _bootstrap_ci(out_scores, y0, _safe_auroc, n_boot=args.n_boot, seed=args.seed)
    # stego-vs-control text discriminability (1=stego, 0=control); also ~0.5 under perfect security
    sc_y = np.array([1] * len(stego_texts) + [0] * len(ctrl_texts))
    sc_scores = text_blackbox_oof(stego_texts + ctrl_texts, stego_lens + ctrl_lens, sc_y, seed=args.seed)
    sc_ci = _bootstrap_ci(sc_scores, sc_y, _safe_auroc, n_boot=args.n_boot, seed=args.seed)
    print(f"[output-null] payload-vs-output AUROC {out_ci}  stego-vs-ctrl {sc_ci}", flush=True)

    # ---- ACTIVATION decode: layer/probe sweep + operating points (bit 0) ----
    curve, opt, honest, best = sweep_layers(acts_all, y0, layer_ids, args.n_boot, args.seed)
    print(f"[act-sweep] best layer={best['layer']} probe={best['probe']} "
          f"AUROC={best['auroc']:.3f}", flush=True)

    # ---- per-bit decode capacity (linear probe on the headline layer) for all bits ----
    per_bit = []
    hl = best["layer"]; li = layer_ids.index(hl)
    for k in range(n_bits):
        yk = payload_mat[:, k].astype(int)
        sc = linear_probe_oof(acts_all[:, li, :], yk, seed=args.seed)
        per_bit.append({"bit": k, "stego_pos": stego_positions[k], "linear_auroc": _r(_safe_auroc(sc, yk))})

    vdt = verdict(out_ci, opt["auroc"], opt["tpr_at_1pct_fpr"])

    res = {
        "experiment": "mec_stego_highpower",
        "model": args.model, "load": "8bit",
        "n_stego": int(args.n), "n_control": int(args.n),
        "gen_len": args.gen_len, "payload_bits": n_bits, "stego_positions": stego_positions,
        "n_layers": n_layers, "n_layers_probed": len(layer_ids), "n_boot": args.n_boot,
        "payload_balance_bit0": _r(y0.mean()),
        "receiver_decode_acc_bit0": _r(decode_acc),
        "output_channel": {
            "payload_vs_output_auroc": out_ci,            # must INCLUDE 0.5 -> NULL
            "stego_vs_control_auroc": sc_ci,              # must INCLUDE 0.5 -> NULL
        },
        "activation_decode": {
            "best_layer": best["layer"], "best_probe": best["probe"],
            "best_optimistic": opt,                        # selected on full OOF (headline-max)
            "honest_split_selected": honest,               # selection/eval split (no test leakage)
            "per_bit_capacity": per_bit,
            "layer_probe_curve": curve,                    # full sweep for the appendix figure
        },
        "verdict": vdt,
        "wall_seconds": round(time.time() - t0, 1),
    }
    return res


def main():
    ap = argparse.ArgumentParser(description="MEC stego high-power Prop-2 anchor (RunPod, 8-bit)")
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--n", type=int, default=2000, help="N stego + N control generations (>=2000)")
    ap.add_argument("--gen-len", type=int, default=24)
    ap.add_argument("--stego-pos", type=int, default=4, help="generation step carrying bit 0")
    ap.add_argument("--payload-bits", type=int, default=3, help="MEC bits per generation (>=1)")
    ap.add_argument("--layers", default="sweep",
                    help="'sweep' = every hidden layer; else comma list e.g. 16,24,32")
    ap.add_argument("--prompt", default="The weather today is",
                    help="shared generation prefix held constant -> isolates the payload")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/runpod/tomorrow/mec_highpower.json")
    args = ap.parse_args()

    res = run(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== MEC STEGO HIGH-POWER (Prop-2 anchor) ===")
    oc = res["output_channel"]; ad = res["activation_decode"]
    print(f"  model                 {res['model']}  n={res['n_stego']}  bits={res['payload_bits']}")
    print(f"  OUTPUT payload-vs-out AUROC  {oc['payload_vs_output_auroc']}  (CI must include 0.5)")
    print(f"  OUTPUT stego-vs-ctrl  AUROC  {oc['stego_vs_control_auroc']}")
    print(f"  ACT best layer/probe   {ad['best_layer']}/{ad['best_probe']}")
    print(f"  ACT AUROC (optimistic) {ad['best_optimistic']['auroc']}")
    print(f"  ACT TPR@1%FPR (opt)    {ad['best_optimistic']['tpr_at_1pct_fpr']}")
    print(f"  ACT TPR@5%FPR (opt)    {ad['best_optimistic']['tpr_at_5pct_fpr']}")
    print(f"  ACT AUROC (honest)     {ad['honest_split_selected']['auroc']}")
    print(f"  ACT TPR@1%FPR (honest) {ad['honest_split_selected']['tpr_at_1pct_fpr']}")
    print(f"  ACT TPR@5%FPR (honest) {ad['honest_split_selected']['tpr_at_5pct_fpr']}")
    print(f"  VERDICT                {res['verdict']}")
    print(f"\nwrote {args.out}  ({res['wall_seconds']}s)")


if __name__ == "__main__":
    main()
