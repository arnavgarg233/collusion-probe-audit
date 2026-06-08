"""Apollo strategic-deception suite - confound audit (RunPod, self-contained, paste-and-go).

The Apollo paper (Goldowsky-Dill et al., "Detecting Strategic Deception with Linear Probes",
ICML 2025; repo github.com/ApolloResearch/deception-detection) trains a linear probe on the
~layer-22 residual of Llama-3.3-70B-Instruct (also Gemma-2-9B) and reports 0.96-0.999 AUROC
separating *deceptive* from *honest* model responses across ai_liar / insider_trading /
sandbagging_v2 / how_to_catch_an_ai_liar / roleplaying / werewolf.

In every one of those datasets the deception is EXECUTED in the visible response text the probe
reads (the lie, the cover-story, the wrong answer, the false vote). So our law predicts
CONFOUNDED per dataset: a cheap text baseline over the SAME transcript the activation sees should
match the probe, and the text-controlled delta should fall inside its scenario-grouped permutation
null. We run the audit PER DATASET (each rollout file is its own MIXED honest/deceptive
calibration set) and emit a per-dataset verdict.

Two-step:
  python apollo_audit.py --inspect        # clone + print the rollout schema (run FIRST)
  python apollo_audit.py --run --model meta-llama/Llama-3.3-70B-Instruct --layer 22

If the repo ships cached probe activations aligned to labels, --run reuses them (no extraction).
Otherwise it re-extracts the layer-22 last-token residual from the model on each labeled response.
The PARSE_* field names below are locked to the schema seen on main (input_messages / output_strs /
graded_deception / expected_label); if the schema has drifted, --inspect prints it and the one-spot
edit is marked >>> ADJUST.
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

REPO = "https://github.com/ApolloResearch/deception-detection"
WORK = "/workspace/apollo"
ROLLOUTS = f"{WORK}/data/rollouts"

# The MIXED per-dataset calibration files (each holds both honest and deceptive outputs).
# We default to the llama-70b-3.3 rollouts (the paper's headline model). werewolf /
# how_to_catch_an_ai_liar ship only 9b reasoning rollouts in the repo, parsed separately below.
# >>> ADJUST after --inspect if filenames differ on your checkout.
DATASET_FILES = {
    "ai_liar":              "ai_liar__original_without_answers__llama-70b-3.3.json",
    "insider_trading":      "insider_trading__upscale__llama-70b-3.3.json",
    "insider_trading_dd":   "insider_trading_doubledown__upscale__llama-70b-3.3.json",
    "sandbagging_v2":       "sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json",
    "roleplaying":          "roleplaying__plain__llama-70b-3.3.json",
}
# how_to_catch_an_ai_liar and werewolf live outside data/rollouts as graded rollout json:
EXTRA_FILES = {
    "how_to_catch_an_ai_liar": f"{WORK}/data/how_to_catch_an_ai_liar/improved_rollouts_9b.json",
    "werewolf":                f"{WORK}/data/werewolf/reasoning_rollouts_9b.json",
}


def sh(c):
    print("+", c, flush=True); subprocess.run(c, shell=True)


# ------------------------------------------------------------------ STEP A: inspect
def inspect():
    if not os.path.isdir(WORK):
        sh(f"git clone --depth 1 {REPO} {WORK}")
    print("\n=== repo tree (depth<=2) ===")
    for root, dirs, files in os.walk(WORK):
        d = root[len(WORK):].count(os.sep)
        if d <= 2:
            print(" " * d, root.replace(WORK, "."), "->", files[:12])
        if d >= 2:
            dirs[:] = []
    print("\n=== data/rollouts/ ===")
    for f in sorted(glob.glob(f"{ROLLOUTS}/*.json")):
        print("  ", os.path.basename(f))
    # show one rollout file's schema (top-level keys + first rollout element)
    cand = [os.path.join(ROLLOUTS, v) for v in DATASET_FILES.values()] + list(EXTRA_FILES.values())
    f0 = next((c for c in cand if os.path.isfile(c)), None)
    if f0 is None:
        f0 = next(iter(sorted(glob.glob(f"{ROLLOUTS}/*.json"))), None)
    if f0:
        print("\n=== sample:", f0.replace(WORK, "."), "===")
        try:
            d = json.load(open(f0))
            if isinstance(d, dict):
                print("top-level keys:", list(d.keys()))
                rolls = d.get("rollouts", d.get("data", []))
            else:
                rolls = d
            print("n rollouts:", len(rolls))
            if rolls:
                r0 = rolls[0]
                print("rollout[0] keys:", list(r0.keys()) if isinstance(r0, dict) else type(r0))
                # trim long fields for readability
                show = {}
                for k, v in (r0.items() if isinstance(r0, dict) else []):
                    if isinstance(v, str) and len(v) > 300:
                        show[k] = v[:300] + "...<trim>"
                    elif isinstance(v, list) and v and isinstance(v[0], str) and len(v[0]) > 200:
                        show[k] = [v[0][:200] + "...<trim>"] + (["..."] if len(v) > 1 else [])
                    else:
                        show[k] = v
                print(json.dumps(show, indent=2, default=str)[:3000])
        except Exception as e:
            print("read err:", e)
    print("\n>>> Paste this output back if the schema differs from "
          "(input_messages / output_messages / output_strs / graded_deception / expected_label). "
          "If you see cached probe activations + labels, we skip extraction entirely.")


# ------------------------------------------------------------------ parsing (>>> ADJUST after inspect)
def _msgs_to_text(messages):
    """Render a list of {role, content} chat messages into a flat transcript string."""
    parts = []
    for m in messages or []:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        elif isinstance(m, str):
            parts.append(m)
    return "\n".join(parts)


def parse_dataset(fp):
    """Parse ONE Apollo rollout file -> list of {text, label(1=deceptive), scenario_id}.

    Schema (locked to repo main):
      file = {"dataset_class","base_name","variant","model","rollouts":[ ... ]}
      rollout = {
        "input_messages":[{role,content,detect}],  # shared prompt/context
        "output_messages":[{role,content,detect}], # one per sample
        "output_strs":[str, ...],                  # the model's response text (what probe reads)
        "graded_deception":["honest"|"deceptive", ...],  # per-sample label
        "expected_label":"honest"|"deceptive",     # design-intended label (fallback)
      }
    The VISIBLE transcript = input prompt text + the sampled response (full access fairness:
    the text baseline sees exactly the context the layer-22 last token sees).
    """
    try:
        d = json.load(open(fp))
    except Exception as e:
        print("  parse err", fp, e); return []
    rolls = d.get("rollouts", d.get("data", [])) if isinstance(d, dict) else d
    rows = []
    for si, r in enumerate(rolls):
        if not isinstance(r, dict):
            continue
        prompt_text = _msgs_to_text(r.get("input_messages") or r.get("input_str_messages") or [])
        if not prompt_text:
            prompt_text = str(r.get("input_str", "") or "")
        outs = r.get("output_strs")
        if outs is None:
            outs = [_msgs_to_text([m]) for m in (r.get("output_messages") or [])]
        outs = outs or []
        # >>> ADJUST: per-sample deception label. Primary = graded_deception list; fall back to
        # expected_label (string) broadcast over samples, or a numeric deception_percentage.
        grades = r.get("graded_deception")
        exp = r.get("expected_label")
        pct = r.get("deception_percentage")
        for oi, out in enumerate(outs):
            if grades is not None and oi < len(grades) and grades[oi] is not None:
                g = grades[oi]
                lab = 1 if (str(g).lower().startswith("dec") or g in (1, True)) else 0
            elif exp is not None:
                lab = 1 if str(exp).lower().startswith("dec") else 0
            elif pct is not None:
                lab = 1 if float(pct) >= 0.5 else 0
            else:
                continue
            text = (prompt_text + "\n" + str(out)).strip() if out else prompt_text.strip()
            if text:
                rows.append({"text": text, "label": int(lab), "scenario_id": f"{si}"})
    return rows


def discover_datasets():
    """Map dataset-name -> existing file path on this checkout."""
    found = {}
    for name, fn in DATASET_FILES.items():
        p = os.path.join(ROLLOUTS, fn)
        if os.path.isfile(p):
            found[name] = p
    for name, p in EXTRA_FILES.items():
        if os.path.isfile(p):
            found[name] = p
    return found


def load_cached_activations(name):
    """If the repo ships per-response activations aligned to labels for this dataset, load them.

    Returns (X, y, groups) or None. The Apollo repo mostly stores acts as .pt/.pkl under
    example_results/<run>/; if a per-example .npz with X/y is present we reuse it.
    """
    pats = [f"{WORK}/**/*{name}*activation*.npz", f"{WORK}/**/*{name}*acts*.npz",
            f"{WORK}/**/*activation*{name}*.npz"]
    for pat in pats:
        for fp in glob.glob(pat, recursive=True):
            try:
                d = np.load(fp, allow_pickle=True)
                keys = set(d.files)
                xk = next((k for k in ["act", "X", "activations", "residual", "acts"] if k in keys), None)
                yk = next((k for k in ["y", "label", "labels", "deceptive"] if k in keys), None)
                gk = next((k for k in ["scenario_id", "group", "groups", "scenario"] if k in keys), None)
                if xk and yk:
                    X = np.asarray(d[xk]); y = np.asarray(d[yk]).astype(int)
                    g = np.asarray(d[gk]) if gk else np.arange(len(y))
                    print(f"  using cached activations {fp.replace(WORK,'.')}: X{X.shape}")
                    return X, y, g
            except Exception:
                continue
    return None


# ------------------------------------------------------------------ extraction (Llama-3.3-70B layer-22)
def extract_resid(texts, model_name, layer, max_len=2048):
    """Last-token residual-stream activation at `layer` for each transcript (the probe's input)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=__import__("transformers").BitsAndBytesConfig(load_in_8bit=True), device_map="auto").eval()
    out = []
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states[layer][0, -1]
        out.append(hs.float().cpu().numpy())
        if i % 100 == 0:
            print(f"    extract {i}/{len(texts)}", flush=True)
    return np.stack(out)


# ------------------------------------------------------------------ the audit (same protocol as among_us)
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
    y = np.asarray(y).astype(int)
    text_clf = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True),
                                     LogisticRegression(solver="liblinear", C=1.0, max_iter=5000))
    lr = lambda: LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
    Xs = StandardScaler().fit_transform(X_act)
    Apca = PCA(min(30, Xs.shape[1], max(2, Xs.shape[0] - 1))).fit_transform(Xs)
    tpred = _oof(text_clf(), list(texts), y, groups)
    act_auroc = _auroc(_oof(lr(), Apca, y, groups), y)
    text_auroc = _auroc(tpred, y)
    full = np.column_stack([tpred.reshape(-1, 1), Apca])
    full_auroc = _auroc(_oof(lr(), full, y, groups), y)
    delta = full_auroc - text_auroc
    rng = np.random.default_rng(seed); floors = []
    for _ in range(n_perm):
        yp = _block_perm(y, groups, rng)
        v = _auroc(_oof(text_clf(), list(texts), yp, groups), yp)
        f = _auroc(_oof(lr(), np.column_stack([_oof(text_clf(), list(texts), yp, groups).reshape(-1, 1), Apca]), yp, groups), yp)
        if v == v and f == f:
            floors.append(f - v)
    floor95 = float(np.percentile(floors, 95)) if floors else float("nan")
    p = float(np.mean([fl >= delta for fl in floors])) if floors else float("nan")
    verdict = ("WHITE_BOX_WINS" if (delta > floor95 and delta >= 0.10) else
               "CONFOUNDED" if delta <= floor95 else "PARTIAL")
    return dict(n=int(len(y)), pos=int(y.sum()), act=round(act_auroc, 3), text=round(text_auroc, 3),
                full=round(full_auroc, 3), delta=round(delta, 3), floor95=round(floor95, 3),
                p=round(p, 3), verdict=verdict)


def run_one(name, fp, model, layer):
    print(f"\n--- dataset: {name}  ({os.path.basename(fp)}) ---")
    cached = load_cached_activations(name)
    if cached is not None:
        X, y, g = cached
        msgs = parse_dataset(fp)
        texts = [m["text"] for m in msgs][:len(y)] if len(msgs) >= len(y) else [""] * len(y)
    else:
        msgs = parse_dataset(fp)
        pos = sum(m["label"] for m in msgs)
        print(f"  parsed {len(msgs)} responses, {pos} deceptive / {len(msgs)-pos} honest")
        if len(msgs) < 40 or pos == 0 or pos == len(msgs):
            print("  SKIP: too few responses or single-class (run --inspect and adjust PARSE_* fields)")
            return dict(n=len(msgs), pos=pos, verdict="SKIP_UNPARSED_OR_SINGLECLASS")
        texts = [m["text"] for m in msgs]
        y = np.array([m["label"] for m in msgs])
        g = np.array([m["scenario_id"] for m in msgs])
        X = extract_resid(texts, model, layer)
    res = audit(X, texts, y, g)
    print(f"  act={res['act']} text={res['text']} full={res['full']} "
          f"delta={res['delta']} floor95={res['floor95']} -> {res['verdict']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--datasets", default="all",
                    help="comma-list of dataset names, or 'all'")
    ap.add_argument("--out", default="/workspace/apollo_result.json")
    a = ap.parse_args()
    if a.inspect or not a.run:
        inspect(); return
    if not os.path.isdir(WORK):
        sh(f"git clone --depth 1 {REPO} {WORK}")
    found = discover_datasets()
    if not found:
        print("No rollout files found - run --inspect first and adjust DATASET_FILES.")
        return
    want = set(found) if a.datasets == "all" else set(s.strip() for s in a.datasets.split(","))
    results = {}
    for name in sorted(found):
        if name not in want:
            continue
        try:
            results[name] = run_one(name, found[name], a.model, a.layer)
        except Exception as e:
            print(f"  ERROR on {name}: {e}")
            results[name] = dict(verdict="ERROR", error=str(e))
    print("\n=== APOLLO DECEPTION AUDIT (per dataset) ===")
    print(f"{'dataset':<26} {'n':>5} {'act':>6} {'text':>6} {'delta':>7} {'fl95':>7}  verdict")
    for name, r in results.items():
        if "act" in r:
            print(f"{name:<26} {r['n']:>5} {r['act']:>6} {r['text']:>6} "
                  f"{r['delta']:>7} {r['floor95']:>7}  {r['verdict']}")
        else:
            print(f"{name:<26} {r.get('n','-'):>5} {'-':>6} {'-':>6} {'-':>7} {'-':>7}  {r['verdict']}")
    print("\nExpected: CONFOUNDED on every dataset (deception is executed in the visible response "
          "the layer-22 probe reads; a TF-IDF baseline over the same transcript should match it, "
          "and the text-controlled delta should sit inside the scenario-grouped null).")
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
