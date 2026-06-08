"""Among Us deception - confound audit (RunPod, self-contained, paste-and-go).

The Among Us paper (Golechha & Garriga-Alonso, arXiv:2504.04072, repo github.com/7vik/AmongUs)
claims a layer-20 linear probe on Phi-4 detects deception at 0.94-0.99 AUROC. Deception there is
EXECUTED in visible chat (lies/accusations/votes), so our law predicts CONFOUNDED: a cheap
text baseline over the same messages should match the probe, and the text-controlled delta should
fall within its game-grouped permutation null.

Two-step:
  python among_us_audit.py --inspect        # clone + print the data schema (run FIRST)
  python among_us_audit.py --run --model microsoft/phi-4 --layer 20   # parse + extract + audit

If the repo already ships cached activations + labels, --run reuses them (no extraction).
Otherwise it re-extracts the layer-20 residual from Phi-4 on each labeled message.
The PARSE_* field names below are best-effort; after --inspect, adjust them to the real schema
(it's a one-spot edit, marked >>> ADJUST).
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

REPO = "https://github.com/7vik/AmongUs"
WORK = "/workspace/amongus"


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
            print(" " * d, root.replace(WORK, "."), "->", files[:10])
        if d >= 2:
            dirs[:] = []
    jsons = glob.glob(f"{WORK}/**/*.json", recursive=True) + glob.glob(f"{WORK}/**/*.jsonl", recursive=True)
    acts = (glob.glob(f"{WORK}/**/*.npy", recursive=True) + glob.glob(f"{WORK}/**/*.npz", recursive=True)
            + glob.glob(f"{WORK}/**/*.pt", recursive=True) + glob.glob(f"{WORK}/**/*.safetensors", recursive=True))
    print(f"\n{len(jsons)} json/jsonl, {len(acts)} activation/weight files")
    for a in acts[:10]:
        print("  act/weight:", a.replace(WORK, "."))
    if jsons:
        f0 = jsons[0]
        print("\n=== sample:", f0.replace(WORK, "."), "===")
        try:
            if f0.endswith(".jsonl"):
                d = json.loads(open(f0).readline())
            else:
                d = json.load(open(f0))
            sample = d[0] if isinstance(d, list) else d
            print(json.dumps(sample, indent=2)[:2500])
        except Exception as e:
            print("read err:", e)
    print("\n>>> Paste this output back and I'll lock the parser to the real schema. "
          "If you see cached activations + a labels file, we skip extraction entirely.")


# ------------------------------------------------------------------ parsing (>>> ADJUST after inspect)
def parse_messages():
    """Return list of dicts: {text, label(1=deceptive), game_id}. Best-effort across likely schemas."""
    rows = []
    files = sorted(glob.glob(f"{WORK}/**/*.json", recursive=True) + glob.glob(f"{WORK}/**/*.jsonl", recursive=True))
    for fp in files:
        gid = os.path.basename(fp)
        try:
            games = [json.loads(l) for l in open(fp)] if fp.endswith(".jsonl") else json.load(open(fp))
        except Exception:
            continue
        games = games if isinstance(games, list) else [games]
        for g in games:
            gameid = g.get("game_id", g.get("id", gid)) if isinstance(g, dict) else gid
            msgs = (g.get("messages") or g.get("discussion") or g.get("transcript") or g.get("turns") or []) if isinstance(g, dict) else []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                text = m.get("text") or m.get("content") or m.get("message") or m.get("utterance") or ""
                # >>> ADJUST: deception label. Common encodings: explicit 'deceptive'/'is_lie',
                # or derive from speaker role 'impostor' + a 'lie' flag.
                lab = m.get("deceptive", m.get("is_lie", m.get("lie", None)))
                if lab is None:
                    role = (m.get("role") or m.get("speaker_role") or "").lower()
                    lab = 1 if ("impostor" in role or m.get("is_impostor")) else 0
                if text and lab is not None:
                    rows.append({"text": str(text), "label": int(bool(lab)), "game_id": str(gameid)})
    return rows


def load_cached_activations():
    """If the repo ships per-message activations aligned to labels, load them. Returns (X, y, groups) or None."""
    npz = glob.glob(f"{WORK}/**/*activation*.npz", recursive=True) + glob.glob(f"{WORK}/**/*acts*.npz", recursive=True)
    for fp in npz:
        try:
            d = np.load(fp, allow_pickle=True)
            keys = set(d.files)
            xk = next((k for k in ["act", "X", "activations", "residual"] if k in keys), None)
            yk = next((k for k in ["y", "label", "labels", "deceptive"] if k in keys), None)
            gk = next((k for k in ["game_id", "group", "groups", "game"] if k in keys), None)
            if xk and yk:
                X = np.asarray(d[xk]); y = np.asarray(d[yk]).astype(int)
                g = np.asarray(d[gk]) if gk else np.arange(len(y))
                print(f"using cached activations {fp.replace(WORK,'.')}: X{X.shape}")
                return X, y, g
        except Exception:
            continue
    return None


# ------------------------------------------------------------------ extraction (Phi-4 layer-20)
def extract_phi4(texts, model_name, layer, max_len=512):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").eval()
    out = []
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            hs = model(**ids, output_hidden_states=True).hidden_states[layer][0, -1]
        out.append(hs.float().cpu().numpy())
        if i % 200 == 0:
            print(f"  extract {i}/{len(texts)}", flush=True)
    return np.stack(out)


# ------------------------------------------------------------------ the audit (same protocol as the paper)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="microsoft/phi-4")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--out", default="/workspace/amongus_result.json")
    a = ap.parse_args()
    if a.inspect or not a.run:
        inspect(); return
    cached = load_cached_activations()
    if cached is not None:
        X, y, g = cached
        # try to attach texts if available; else text baseline is skipped (NaN)
        msgs = parse_messages()
        texts = [m["text"] for m in msgs][:len(y)] if len(msgs) >= len(y) else [""] * len(y)
    else:
        msgs = parse_messages()
        print(f"parsed {len(msgs)} messages, {sum(m['label'] for m in msgs)} deceptive")
        assert len(msgs) > 50, "parser found too few messages - run --inspect and adjust PARSE_* fields"
        texts = [m["text"] for m in msgs]; y = np.array([m["label"] for m in msgs])
        g = np.array([m["game_id"] for m in msgs])
        X = extract_phi4(texts, a.model, a.layer)
    res = audit(X, texts, y, g)
    print("\n=== AMONG US AUDIT ===")
    print(json.dumps(res, indent=2))
    print(f"\nExpected: CONFOUNDED (deception executes in visible chat). "
          f"act={res['act']} text={res['text']} delta={res['delta']} floor95={res['floor95']} -> {res['verdict']}")
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
