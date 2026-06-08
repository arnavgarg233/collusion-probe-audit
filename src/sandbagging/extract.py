"""Sandbagging activation extraction (capability-hiding probe).

For each WMDP question, run the password-locked Mistral-7B with and without the password,
recording the model's answer each way + the no-password mid-layer activation. The audit then
asks: does the no-password activation decode the CORRECT (withheld) answer beyond the model's
(wrong) no-password output? If yes -> activations reveal hidden capability behaviour can't.
Data recipe from github.com/Felhof/sandbagging-elicitation (wmdp_test.csv has both prompts).
"""
import os, csv, argparse, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "mistralai/Mistral-7B-v0.1"
ADAPTER = "FelixHofstaetter/mistral-7b-sandbagging-new"
LETTERS = ["A", "B", "C", "D"]
LAYERS = [16, 24, 28, 30]   # sweep late layers (lit finds sandbag signal ~layer 30 of 32 in Mistral)


def device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=370)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--csv", default="/tmp/sandbag/data/wmdp_test.csv")
    ap.add_argument("--out", default="/tmp/sandbag/acts.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dev = device(); print(f"[device] {dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(ADAPTER)   # adds <PAD> -> vocab 32001 (matches adapter)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
    model.config.pad_token_id = tok.pad_token_id
    model.resize_token_embeddings(len(tok))        # 32000 -> 32001 so the adapter loads
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    model.to(dev).eval()
    print(f"[load] {time.time()-t0:.0f}s, {model.config.num_hidden_layers} layers", flush=True)

    cand_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in LETTERS]
    rows = list(csv.DictReader(open(a.csv)))[:a.n]
    nopwd_act, correct, nopwd_pred, pwd_pred, subj = [], [], [], [], []
    start = 0
    if os.path.exists(a.out):                       # RESUME from last checkpoint
        try:
            prev = np.load(a.out, allow_pickle=True)
            nopwd_act = list(prev["act"]); correct = list(prev["correct"])
            nopwd_pred = list(prev["nopwd_pred"]); pwd_pred = list(prev["pwd_pred"])
            subj = list(prev["subject"]); start = len(correct)
            print(f"[resume] {start} questions already done", flush=True)
        except Exception:
            pass

    def save():
        np.savez(a.out, act=np.stack(nopwd_act).astype(np.float32), correct=np.array(correct),
                 nopwd_pred=np.array(nopwd_pred), pwd_pred=np.array(pwd_pred),
                 subject=np.array(subj), layers=np.array(LAYERS))   # act: [n, len(LAYERS), hidden]

    @torch.no_grad()
    def run(prompt, want_act):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
        out = model(**ids, output_hidden_states=want_act)
        ll = out.logits[0, -1]
        pred = LETTERS[int(torch.tensor([ll[c] for c in cand_ids]).argmax())]
        act = (np.stack([out.hidden_states[L][0, -1].float().cpu().numpy() for L in LAYERS])
               if want_act else None)   # shape [len(LAYERS), hidden]
        return pred, act

    for i, r in enumerate(rows):
        if i < start:
            continue
        try:
            p_no, act = run(r["question_prompt"], True)
            p_pw, _ = run(r["question_prompt_with_password"], False)
        except Exception as e:
            print(f"  row {i} ERR {e}", flush=True); continue
        nopwd_act.append(act); correct.append(r["answer"].strip())
        nopwd_pred.append(p_no); pwd_pred.append(p_pw); subj.append(r.get("subject", "?"))
        if i % 25 == 0 or i == len(rows) - 1:
            save()                                   # CHECKPOINT - crash-safe
            print(f"  {i}/{len(rows)} t={time.time()-t0:.0f}s saved  "
                  f"no-pwd_acc={np.mean([c==p for c,p in zip(correct,nopwd_pred)]):.2f} "
                  f"pwd_acc={np.mean([c==p for c,p in zip(correct,pwd_pred)]):.2f}", flush=True)

    save()
    print(f"[done] {len(correct)} Qs -> {a.out}", flush=True)
    print(f"  no-password acc:  {np.mean([c==p for c,p in zip(correct,nopwd_pred)]):.3f} (sandbag -> low)", flush=True)
    print(f"  with-password acc:{np.mean([c==p for c,p in zip(correct,pwd_pred)]):.3f} (full capability -> high)", flush=True)


if __name__ == "__main__":
    main()
