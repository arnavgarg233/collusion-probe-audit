"""Recompute the sandbagging numbers on CPU from the stored activations.

One extraction (sandbag_quantization_transfer_rawacts.npz: 370 WMDP questions, password-locked
Mistral-7B at fp16, int8 and nf4, layers 16/24/28/30) and one decoder (per-fold StandardScaler,
PCA(40), logistic regression with C=0.5, GroupKFold(5) by subject) give every decode number in
the paper's sandbagging section:

- decode_one_run     -> sandbag_one_run.json / .csv  (Table 9, Fig. 6, quantization paragraph)
- worstcase_subspace -> worstcase_subspace.csv       (worst-case subspace paragraph)
- access_fair        -> sandbag_access_fair.json     (question-text baseline, controlled delta)

access_fair needs the WMDP prompt file, which is fetched from its public source and checked
against the SHA-256 in data/manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from . import decoders as dec
from .paths import DATA, EVIDENCE, EXTERNAL, SANDBAG_ACTS, rel

OUT = EVIDENCE / "sandbagging"
N_BOOT = 1000
N_PERM = 200


def _load():
    z = np.load(SANDBAG_ACTS, allow_pickle=True)
    return z, [int(x) for x in z["layers"]], np.asarray(z["correct"]), np.asarray(z["subject"])


def _mask(z, q, correct):
    nop = np.asarray(z[f"nopwd_{q}"])
    pwd = np.asarray(z[f"pwd_{q}"])
    return nop, pwd, (nop != correct) & (pwd == correct)


def decode_one_run() -> dict:
    z, layers, correct, subject = _load()
    out = {
        "source": rel(SANDBAG_ACTS),
        "decoder": "per-fold StandardScaler+PCA(40)+LogisticRegression(C=0.5), "
        "GroupKFold(5) by subject",
        "n_boot": N_BOOT,
        "per_quant": {},
    }
    for q in ["fp16", "int8", "nf4"]:
        acts = z[f"acts_{q}"]
        nop, pwd, m = _mask(z, q, correct)
        yc = correct[m]
        gc = subject[m]
        _, counts = np.unique(yc, return_counts=True)
        Q = {
            "n": int(len(correct)),
            "no_pwd_acc": round(float((nop == correct).mean()), 3),
            "pwd_acc": round(float((pwd == correct).mean()), 3),
            "n_sandbagged": int(m.sum()),
            "visible_acc_on_sandbagged": round(float((nop[m] == yc).mean()), 3),
            "majority_letter_acc": round(float(counts.max() / counts.sum()), 3),
            "layers": {},
        }
        for li, L in enumerate(layers):
            X = acts[m, li, :].astype(np.float64)
            proba, classes, hard = dec.binarized_oof_proba(X, yc, gc, 40)
            rows = np.where(~np.isnan(proba).any(1))[0]
            hit = (hard[rows] == yc[rows]).astype(float)
            rng = np.random.default_rng(0)
            accs = [hit[rng.integers(0, len(rows), len(rows))].mean() for _ in range(N_BOOT)]
            au = dec.ovr_macro_auroc(proba, classes, yc, n_boot=N_BOOT, seed=0)
            ps = np.concatenate([proba[rows, j] for j in range(len(classes))])
            py = np.concatenate([(yc[rows] == c).astype(int) for c in classes])
            Q["layers"][str(L)] = {
                "acc": round(float(hit.mean()), 3),
                "acc_ci": [
                    round(float(np.percentile(accs, 2.5)), 3),
                    round(float(np.percentile(accs, 97.5)), 3),
                ],
                "auroc": au["point"],
                "auroc_ci": [au["lo"], au["hi"]],
                "tpr1": dec.bootstrap_ci(ps, py, "tpr", 0.01, N_BOOT),
                "tpr5": dec.bootstrap_ci(ps, py, "tpr", 0.05, N_BOOT),
            }
        out["per_quant"][q] = Q
    return out


def one_run_csv(out: dict) -> str:
    lines = ["quant,layer,n_sandbagged,acc,acc_lo,acc_hi,auroc,auroc_lo,auroc_hi,tpr1,tpr5"]
    for q, Q in out["per_quant"].items():
        for L, r in Q["layers"].items():
            lines.append(
                f"{q},{L},{Q['n_sandbagged']},{r['acc']},{r['acc_ci'][0]},{r['acc_ci'][1]},"
                f"{r['auroc']},{r['auroc_ci'][0]},{r['auroc_ci'][1]},{r['tpr1']['point']},"
                f"{r['tpr5']['point']}"
            )
    return "\n".join(lines) + "\n"


def worstcase_subspace() -> str:
    z, layers, correct, subject = _load()
    _, _, m = _mask(z, "fp16", correct)
    A = z["acts_fp16"][m, layers.index(24), :].astype(np.float64)
    rows = dec.pca_ablation_curve(A, correct[m], subject[m], [0, 3, 6, 10, 20], n_boot=200, n_perm=20)
    lines = [
        "# fp16 run of sandbag_one_run.csv (n=205, layer 24). Top-k PCA directions fit on the "
        "training fold and projected out; 4-way decode accuracy (chance 0.25), GroupKFold by subject.",
        "subspace_removed_topk,decode_acc,chance",
    ]
    lines += [f"{r['k']},{r['decode_acc']},0.25" for r in rows]
    return "\n".join(lines) + "\n"


def wmdp_csv() -> Path:
    """The WMDP prompt file used for extraction, fetched once and checked by SHA-256."""
    src = next(
        s for s in json.loads((DATA / "manifest.json").read_text())["sources"] if s["id"] == "wmdp_prompts"
    )
    path = EXTERNAL / "wmdp_test.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(src["url"], path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != src["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest} does not match the manifest")
    return path


def access_fair() -> dict:
    z, layers, correct, subject = _load()
    no, _, ans, sub = dec.parse_rows(wmdp_csv(), 370)
    if list(ans) != list(correct) or list(sub) != list(subject):
        raise SystemExit("WMDP rows do not match the activation file")
    _, _, m = _mask(z, "fp16", correct)
    yc = correct[m]
    gc = subject[m]
    prompts = np.asarray(no, dtype=object)[m]

    def text_proba(y):
        classes = sorted(set(y))
        n = len(y)
        proba = np.full((n, len(classes)), np.nan)
        k = max(2, min(5, len(set(gc))))
        for tr, te in GroupKFold(k).split(np.zeros(n), y, gc):
            clf = dec.text_clf().fit(prompts[tr], y[tr])
            pr = clf.predict_proba(prompts[te])
            for j, c in enumerate(clf.classes_):
                proba[te, classes.index(c)] = pr[:, j]
        return proba, classes

    def macro(proba, classes, y):
        return dec.ovr_macro_auroc(proba, classes, y)["point"]

    tp, tc = text_proba(yc)
    text_acc = float((np.asarray(tc)[np.nanargmax(tp, 1)] == yc).mean())
    text_au = macro(tp, tc, yc)
    out = {"n_sandbagged": int(m.sum()), "text_acc": round(text_acc, 3), "text_auroc": text_au,
           "layers": {}}
    for li, L in enumerate(layers):
        X = z["acts_fp16"][m, li, :].astype(np.float64)
        p, c, _ = dec.binarized_oof_proba(X, yc, gc, 40)
        delta = macro(p, c, yc) - text_au
        rng = np.random.default_rng(0)
        null = []
        for _ in range(N_PERM):
            yp = dec.block_perm_labels(yc, gc, rng)
            pp, cc, _ = dec.binarized_oof_proba(X, yp, gc, 40)
            tpp, tcc = text_proba(yp)
            null.append(macro(pp, cc, yp) - macro(tpp, tcc, yp))
        fl = float(np.percentile(null, 95))
        pv = float(np.mean([d >= delta for d in null]))
        out["layers"][str(L)] = {"delta": round(delta, 3), "floor95": round(fl, 3), "p": round(pv, 3)}
    return out


def _compare_access_fair(new: dict, old: dict) -> list[str]:
    """Delta, text baseline and n must match exactly. The permutation floor depends on the
    iteration order of a set of subject names (string hashing; PYTHONHASHSEED=0 reproduces the
    stored floors exactly), so it is compared to 0.01 and
    the p-value only on which side of 0.005 it falls for the layers the paper reports."""
    errs = []
    for k in ("n_sandbagged", "text_acc", "text_auroc"):
        if new[k] != old[k]:
            errs.append(f"access_fair {k}: {new[k]} != {old[k]}")
    for L, r in old["layers"].items():
        n = new["layers"][L]
        if n["delta"] != r["delta"]:
            errs.append(f"access_fair L{L} delta: {n['delta']} != {r['delta']}")
        if abs(n["floor95"] - r["floor95"]) > 0.01:
            errs.append(f"access_fair L{L} floor95: {n['floor95']} vs {r['floor95']}")
        if L != "16" and (n["p"] < 0.005) != (r["p"] < 0.005):
            errs.append(f"access_fair L{L} p: {n['p']} vs {r['p']}")
    return errs


TOL = 0.01


def _numbers(x) -> list:
    if isinstance(x, dict):
        return [v for k in x for v in _numbers(x[k])]
    if isinstance(x, list):
        return [v for i in x for v in _numbers(i)]
    return [x]


def _max_diff(stored: str, new: str, is_json: bool) -> float:
    """Largest absolute difference between two records with the same layout. Apple's Accelerate
    library differs between macOS releases, which moves the near-chance layer 16 by up to one
    question (0.005) and its bootstrap interval by up to 0.01; every other value is identical."""
    if is_json:
        a, b = _numbers(json.loads(stored)), _numbers(json.loads(new))
    else:
        a = [c for line in stored.splitlines() for c in line.split(",")]
        b = [c for line in new.splitlines() for c in line.split(",")]
    if len(a) != len(b):
        return float("inf")
    d = 0.0
    for x, y in zip(a, b):
        try:
            d = max(d, abs(float(x) - float(y)))
        except (TypeError, ValueError):
            if x != y:
                return float("inf")
    return round(d, 9)


def run(check: bool, full: bool, out_dir: Path | None = None) -> int:
    warnings.filterwarnings("ignore")  # scikit-learn convergence and solver deprecation notices
    out_dir = Path(out_dir) if out_dir else OUT
    errs: list[str] = []
    one = decode_one_run()
    one_json = json.dumps(one, indent=1)
    one_csv = one_run_csv(one)
    worst = worstcase_subspace()
    products = {
        "sandbag_one_run.json": one_json,
        "sandbag_one_run.csv": one_csv,
        "worstcase_subspace.csv": worst,
    }
    for name, text in products.items():
        if check:
            stored = (OUT / name).read_text()
            d = _max_diff(stored, text, name.endswith(".json"))
            print(f"{'ok  ' if d <= TOL else 'DIFF'} {rel(OUT / name)} (max difference {d:g})")
            if d > TOL:
                errs.append(name)
        else:
            (out_dir / name).write_text(text)
    if full:
        fair = access_fair()
        if check:
            old = json.loads((OUT / "sandbag_access_fair.json").read_text())
            e = _compare_access_fair(fair, old)
            print(f"{'ok  ' if not e else 'DIFF'} {rel(OUT / 'sandbag_access_fair.json')}"
                  f" (floor to 0.01): {json.dumps(fair['layers'])}")
            errs += e
        else:
            (out_dir / "sandbag_access_fair.json").write_text(json.dumps(fair, indent=1))
    for e in errs:
        print("mismatch:", e)
    return 1 if errs else 0
