"""Black-to-White Boosts - confound audit (RunPod, self-contained, paste-and-go).

The "Benchmarking Deception Probes via Black-to-White Performance Boosts" paper
(Parrack, Attubato & Heimersheim, arXiv:2507.12691, repo
github.com/carlo-leonardo-attubato/black_to_white_boosts) reports white-box vs black-box
numbers DIRECTLY: a token-level deception probe (logistic regression on the residual
stream at layer 22 of Llama-3.3-70B-Instruct) is compared against an LLM monitor that
only sees the visible transcript. The "boost" is how much white-box (probe) beats
black-box (text). This dataset is the LIGHTEST lift in the breadth suite because the
probe infra is RELEASED: every shipped sample already carries the per-token probe
scores, so NO 70B forward pass is needed on the box.

The six datasets ship as Git-LFS JSONL files in results/:
  ai_liar, ai_liar_deception_dilution_100, insider_trading_full,
  sandbagging_wmdp_mmlu, apps_backdoors, alpaca (control, all-honest).
Each line is one DeceptionDetectionSample:
  {"metadata": {"dataset", "sample_index", "label": "honest"|"deceptive", ...},
   "conversations": {"main": {"tokens":[...], "messages":[{role,content},...],
                              "available_probes":[...],
                              "probe_scores": {<probe>: [float per token], ...},
                              "assistant_indices":[...], "public_indices":[...], ...},
                     <follow-up branches...>},
   "monitor_results": [ ... shipped black/white monitor scores ... ]}

Our LAW: deception here is EXECUTED in the visible response (a lie / a concealed
trade / a sandbagged answer / a backdoored block of code), so we predict CONFOUNDED:
a cheap text baseline that sees the SAME public transcript should match the probe, and
the text-controlled delta should fall within its dataset-grouped permutation null. We
REPRODUCE the boost (white-box probe AUROC vs black-box text AUROC) AND add the rigorous
controlled-delta + group-aware permutation floor on top.

Two-step:
  python b2w_audit.py --inspect       # clone (+ git lfs pull) and print the data schema (run FIRST)
  python b2w_audit.py --run           # parse shipped probe scores + audit (NO GPU needed)

The probe-score field names below are LOCKED to the released schema (metadata.label,
conversations.main.probe_scores, public_indices). The only spots you might touch on the
box are marked >>> ADJUST (which branch counts as the assistant response, and which
precomputed probe to treat as THE white-box feature).
"""
import argparse, glob, json, os, subprocess
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Real repo (verified): default branch is `master`; data lives in results/*.jsonl via Git LFS.
REPO = "https://github.com/carlo-leonardo-attubato/black_to_white_boosts"
WORK = "/workspace/b2w"
RESULTS = os.path.join(WORK, "results")
# The six released datasets (file stems under results/). alpaca is the all-honest control.
DATASETS = [
    "ai_liar",
    "ai_liar_deception_dilution_100",
    "insider_trading_full",
    "sandbagging_wmdp_mmlu",
    "apps_backdoors",
    "alpaca",
]
# Real HF id for the assistant whose layer-22 residual the probe reads (probe scores are shipped;
# this is only needed if you choose to RE-EXTRACT instead of using the released scores).
MODEL_HF = "meta-llama/Llama-3.3-70B-Instruct"
PROBE_LAYER = 22


def sh(c):
    print("+", c, flush=True); subprocess.run(c, shell=True)


# ------------------------------------------------------------------ STEP A: inspect
def inspect():
    if not os.path.isdir(WORK):
        sh(f"git clone --depth 1 {REPO} {WORK}")
    # Data files are Git-LFS pointers; pull the real bytes. Harmless if lfs is absent / already pulled.
    sh(f"git -C {WORK} lfs install || true")
    sh(f"git -C {WORK} lfs pull || true")

    print("\n=== results/ files ===")
    files = sorted(glob.glob(f"{RESULTS}/*.jsonl"))
    for fp in files:
        sz = os.path.getsize(fp)
        tag = "  (LFS POINTER - run `git lfs pull`)" if sz < 1000 else ""
        print(f"  {os.path.basename(fp)}  {sz} bytes{tag}")
    if not files:
        print("  (none found - check repo layout under", RESULTS, ")")
        return

    # Pick a real (non-pointer) file to dump.
    real = next((fp for fp in files if os.path.getsize(fp) > 1000), files[0])
    print(f"\n=== sample line: {os.path.basename(real)} ===")
    try:
        line = open(real).readline()
        d = json.loads(line)
    except Exception as e:
        print("read err (likely an un-pulled LFS pointer):", e)
        print("first 200 bytes:", open(real).read(200))
        return

    print("TOP KEYS:", list(d.keys()))
    print("metadata:", json.dumps(d.get("metadata", {}), indent=2)[:800])
    convs = d.get("conversations", {})
    print("conversation branches:", list(convs.keys()))
    bid = "main" if "main" in convs else (list(convs)[0] if convs else None)
    if bid:
        b = convs[bid]
        print(f"\nbranch '{bid}' keys:", list(b.keys()))
        print("  available_probes:", b.get("available_probes"))
        ps = b.get("probe_scores", {})
        print("  probe_scores probes:", list(ps.keys()))
        for k, v in ps.items():
            print(f"    {k}: len={len(v)}  head={ [round(float(x),3) for x in v[:5]] }")
        print("  n_tokens:", len(b.get("tokens", [])))
        print("  n_assistant_idx:", len(b.get("assistant_indices", [])),
              " n_public_idx:", len(b.get("public_indices", [])))
        print("  messages[:2]:", json.dumps(b.get("messages", [])[:2])[:600])
    mr = d.get("monitor_results", [])
    print("\nshipped monitor_results:", len(mr),
          "->", [r.get("monitor", {}).get("name") for r in mr[:8]])

    # Label balance per file (cheap, streams).
    print("\n=== label balance per dataset ===")
    for fp in files:
        if os.path.getsize(fp) < 1000:
            print(f"  {os.path.basename(fp)}: (LFS pointer, skipped)")
            continue
        n = pos = 0
        for ln in open(fp):
            if not ln.strip():
                continue
            try:
                lab = json.loads(ln)["metadata"].get("label")
            except Exception:
                continue
            n += 1; pos += int(lab == "deceptive")
        print(f"  {os.path.basename(fp)}: n={n}  deceptive={pos}  honest={n-pos}")
    print("\n>>> After --inspect, confirm: (a) probe_scores keys (the white-box probe names), "
          "(b) that metadata.label is honest/deceptive, (c) which branch is the response "
          "('main'). Then --run uses the shipped scores directly (no GPU).")


# ------------------------------------------------------------------ parsing the shipped samples
def _response_branch(sample):
    """Return the conversation branch holding the assistant response. >>> ADJUST if you
    want a follow-up branch instead of the primary transcript."""
    convs = sample.get("conversations", {})
    if "main" in convs:
        return convs["main"]
    return convs[next(iter(convs))] if convs else None


def _public_assistant_indices(branch):
    """Indices of public assistant tokens (the visible response the probe/monitor scores)."""
    a = set(branch.get("assistant_indices", []))
    pub = branch.get("public_indices")
    if pub is not None:
        a &= set(pub)
    if not a:                                   # fallback: any public token, else all tokens
        a = set(branch.get("public_indices", [])) or set(range(len(branch.get("tokens", []))))
    return sorted(a)


def _public_text(branch):
    """Visible transcript text the BLACK-BOX monitor would read (access-fairness: the text
    baseline gets exactly the public content the probe is scored over)."""
    msgs = branch.get("messages", [])
    if msgs:
        return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in msgs)
    # fallback: reconstruct from public tokens
    toks = branch.get("tokens", [])
    idx = branch.get("public_indices") or range(len(toks))
    return "".join(toks[i].replace("Ġ", " ").replace("Ċ", "\n") for i in idx)


def _probe_feats(branch, probe_names):
    """Fixed-length WHITE-BOX feature vector built from the SHIPPED per-token probe scores
    over public assistant tokens. For each probe we take the same summary stats the paper's
    numeric monitors use (mean / peak) plus a few robust extras (std, min, last, q25/q75).
    This is the probe's information, model-free (scores are precomputed at layer 22)."""
    idx = _public_assistant_indices(branch)
    ps = branch.get("probe_scores", {})
    feats = []
    for name in probe_names:
        v = np.asarray(ps.get(name, []), dtype=float)
        s = v[idx] if (len(v) and max(idx, default=-1) < len(v)) else v
        s = s[np.isfinite(s)] if len(s) else s
        if len(s) == 0:
            feats += [0.0] * 7
            continue
        feats += [float(np.mean(s)), float(np.max(s)), float(np.std(s)),
                  float(np.min(s)), float(s[-1]),
                  float(np.percentile(s, 25)), float(np.percentile(s, 75))]
    return np.asarray(feats, dtype=float)


def parse_dataset(stem):
    """Load one results/<stem>.jsonl. Returns (X_probe, texts, y, groups) where:
       X_probe = per-sample white-box feature vector from shipped probe scores,
       texts   = visible public transcript per sample (black-box baseline input),
       y       = 1 if metadata.label == 'deceptive' else 0,
       groups  = sample_index (permutation/CV unit = one rollout)."""
    fp = os.path.join(RESULTS, f"{stem}.jsonl")
    if not os.path.isfile(fp):
        return None
    if os.path.getsize(fp) < 1000:
        print(f"  !! {stem}.jsonl looks like an un-pulled LFS pointer ({os.path.getsize(fp)}B) "
              f"- run `git -C {WORK} lfs pull`")
        return None

    rows = []
    probe_names = None
    for ln in open(fp):
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
        except Exception:
            continue
        b = _response_branch(d)
        if b is None:
            continue
        # Lock the probe set from the first sample (the released white-box probes).
        if probe_names is None:
            probe_names = sorted((b.get("probe_scores") or {}).keys())
            if not probe_names:
                print(f"  !! {stem}: no probe_scores in branch - run --inspect")
                return None
            # >>> ADJUST: to pin a SINGLE probe (e.g. only the 'pairs'/'roleplaying' probe)
            # instead of all shipped probes, filter probe_names here.
        lab = d.get("metadata", {}).get("label")
        y = 1 if lab == "deceptive" else 0
        gid = d.get("metadata", {}).get("sample_index", len(rows))
        rows.append((
            _probe_feats(b, probe_names),
            _public_text(b),
            y,
            f"{stem}:{gid}",
        ))
    if not rows:
        return None
    X = np.vstack([r[0] for r in rows])
    texts = [r[1] for r in rows]
    y = np.array([r[2] for r in rows], dtype=int)
    g = np.array([r[3] for r in rows])
    print(f"  {stem}: n={len(y)} deceptive={int(y.sum())} probes={probe_names} "
          f"feat_dim={X.shape[1]}")
    return X, texts, y, g


# ------------------------------------------------------------------ OPTIONAL re-extraction (NOT needed)
def extract_b2w(texts, model_name=MODEL_HF, layer=PROBE_LAYER, max_len=2048):
    """OPTIONAL: re-derive a layer-22 residual feature on the box instead of using the
    shipped probe scores. The released data already contains the probe scores, so this is
    a fallback only - and Llama-3.3-70B needs an H200-class GPU, so DO NOT run it here.
    >>> ADJUST: the released probe is a trained linear direction at layer 22; to truly
    reproduce the probe score you would load that probe vector and dot it with the residual.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=__import__("transformers").BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=__import__("torch").float16), device_map="auto").eval()
    out = []
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states[layer][0, -1]
        out.append(hs.float().cpu().numpy())
        if i % 50 == 0:
            print(f"  extract {i}/{len(texts)}", flush=True)
    return np.stack(out)


# ------------------------------------------------------------------ the audit (access-fairness controlled)
def _oof(est, X, y, g):
    y = np.asarray(y).astype(int); g = np.asarray(g)
    if len(set(y)) < 2 or len(set(g)) < 2:
        return np.full(len(y), np.nan)
    k = max(2, min(5, len(set(g))))
    return cross_val_predict(est, X, y, cv=GroupKFold(k), groups=g, method="predict_proba")[:, 1]


def _auroc(p, y):
    y = np.asarray(y).astype(int); m = ~np.isnan(p)
    return roc_auc_score(y[m], p[m]) if (m.sum() > 4 and len(set(y[m])) == 2) else float("nan")


def _block_perm(y, g, rng):
    g = np.asarray(g); y = np.asarray(y)
    bg = {gg: y[g == gg] for gg in set(g)}
    if all(len(set(v)) == 1 for v in bg.values()):       # label constant within group -> block permute
        ks = list(bg); perm = rng.permutation([bg[k][0] for k in ks]); mp = dict(zip(ks, perm))
        return np.array([mp[gg] for gg in g])
    return rng.permutation(y)


def audit(X_act, texts, y, groups, n_perm=80, seed=0):
    """White-box (shipped probe) vs black-box (visible text) with access-fairness:
       the text baseline reads the SAME public transcript the probe is scored over.
       GO iff text-controlled delta > group-permutation floor95 AND delta >= 0.10."""
    y = np.asarray(y).astype(int)
    text_clf = lambda: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True),
        LogisticRegression(solver="liblinear", C=1.0, max_iter=5000))
    lr = lambda: LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
    Xs = StandardScaler().fit_transform(X_act)
    Apca = PCA(min(30, Xs.shape[1], max(2, Xs.shape[0] - 1))).fit_transform(Xs)
    tpred = _oof(text_clf(), list(texts), y, groups)
    act_auroc = _auroc(_oof(lr(), Apca, y, groups), y)         # white-box probe alone (the "boost" numerator)
    text_auroc = _auroc(tpred, y)                               # black-box visible-text monitor
    full = np.column_stack([tpred.reshape(-1, 1), Apca])
    full_auroc = _auroc(_oof(lr(), full, y, groups), y)         # text + probe
    delta = full_auroc - text_auroc                            # controlled white-box contribution
    boost = act_auroc - text_auroc                             # naive black-to-white boost (paper-style)
    rng = np.random.default_rng(seed); floors = []
    for _ in range(n_perm):
        yp = _block_perm(y, groups, rng)
        v = _auroc(_oof(text_clf(), list(texts), yp, groups), yp)
        f = _auroc(_oof(lr(), np.column_stack(
            [_oof(text_clf(), list(texts), yp, groups).reshape(-1, 1), Apca]), yp, groups), yp)
        if v == v and f == f:
            floors.append(f - v)
    floor95 = float(np.percentile(floors, 95)) if floors else float("nan")
    p = float(np.mean([fl >= delta for fl in floors])) if floors else float("nan")
    verdict = ("WHITE_BOX_WINS" if (delta > floor95 and delta >= 0.10) else
               "CONFOUNDED" if delta <= floor95 else "PARTIAL")
    return dict(n=int(len(y)), pos=int(y.sum()),
                act=round(act_auroc, 3), text=round(text_auroc, 3), full=round(full_auroc, 3),
                boost=round(boost, 3), delta=round(delta, 3), floor95=round(floor95, 3),
                p=round(p, 3), verdict=verdict)


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--datasets", default=",".join(d for d in DATASETS if d != "alpaca"),
                    help="comma-separated dataset stems to audit (alpaca is all-honest -> excluded by default)")
    ap.add_argument("--model", default=MODEL_HF, help="only used by the optional --reextract fallback")
    ap.add_argument("--layer", type=int, default=PROBE_LAYER)
    ap.add_argument("--n_perm", type=int, default=80)
    ap.add_argument("--out", default="/workspace/b2w_result.json")
    a = ap.parse_args()

    if a.inspect or not a.run:
        inspect(); return

    stems = [s.strip() for s in a.datasets.split(",") if s.strip()]
    all_res = {}
    for stem in stems:
        print(f"\n--- parsing {stem} ---")
        parsed = parse_dataset(stem)
        if parsed is None:
            print(f"  skipped {stem} (missing / LFS pointer / no probe scores)")
            continue
        X, texts, y, g = parsed
        if len(set(y)) < 2:
            print(f"  {stem}: single-class (n={len(y)} pos={int(y.sum())}) - control set, "
                  f"no AUROC; skipping audit")
            continue
        res = audit(X, texts, y, g, n_perm=a.n_perm)
        all_res[stem] = res
        print(f"  {stem}: boost(act-text)={res['boost']} | controlled delta={res['delta']} "
              f"floor95={res['floor95']} p={res['p']} -> {res['verdict']}")

    print("\n=== BLACK-TO-WHITE BOOSTS AUDIT ===")
    print(json.dumps(all_res, indent=2))
    print("\nExpected: CONFOUNDED on every set (deception is executed in the visible "
          "response; a text baseline with the SAME public transcript matches the probe, "
          "so the controlled delta falls inside the dataset-grouped permutation floor). "
          "A large naive `boost` with a near-zero `delta` is exactly the confound the "
          "paper's black-vs-white comparison does not control for.")
    json.dump(all_res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
