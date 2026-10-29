import numpy as np, torch, csv, urllib.request, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
BASE="mistralai/Mistral-7B-v0.1"; ADP="FelixHofstaetter/mistral-7b-sandbagging-new"
LAYER=24; LETTERS=["A","B","C","D"]
tok=AutoTokenizer.from_pretrained(ADP)
m=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float16,device_map="cuda")
m.config.pad_token_id=tok.pad_token_id; m.resize_token_embeddings(len(tok),mean_resizing=False)
m=PeftModel.from_pretrained(m,ADP).merge_and_unload().eval()
print("[loaded] sandbagging Mistral-7B",flush=True)
cand=[tok(" "+L,add_special_tokens=False).input_ids[-1] for L in LETTERS]
os.makedirs("/workspace/sb",exist_ok=True); fp="/workspace/sb/wmdp.csv"
if not os.path.exists(fp): urllib.request.urlretrieve("https://raw.githubusercontent.com/Felhof/sandbagging-elicitation/main/data/mcqa/wmdp_test.csv",fp)
rows=list(csv.DictReader(open(fp)))[:300]
STEER={"vec":None,"a":0.0}; cap={}
def hook(mod,inp,out):
    h=out[0] if isinstance(out,tuple) else out
    cap["h"]=h.detach()[:,-1,:].float().cpu().numpy()
    if STEER["vec"] is not None and STEER["a"]!=0.0:
        h[:,-1,:]=h[:,-1,:]+STEER["a"]*STEER["vec"].to(h.dtype).to(h.device)
    return out
m.model.layers[LAYER].register_forward_hook(hook)
@torch.no_grad()
def fwd(p):
    ids=tok(p,return_tensors="pt",truncation=True,max_length=512).to(m.device)
    o=m(**ids); pred=LETTERS[int(torch.tensor([o.logits[0,-1][c] for c in cand]).argmax())]
    return pred, cap["h"][0].copy()
STEER["vec"]=None
na,pa,cor,npp,pwp=[],[],[],[],[]
for r in rows:
    pn,an=fwd(r["question_prompt"]); pp,ap=fwd(r["question_prompt_with_password"])
    na.append(an);pa.append(ap);cor.append(r["answer"].strip());npp.append(pn);pwp.append(pp)
na,pa,cor,npp,pwp=map(np.array,(na,pa,cor,npp,pwp))
sb=(npp!=cor)&(pwp==cor)
print("no-pwd acc %.3f | pwd acc %.3f | sandbagged %d"%((cor==npp).mean(),(cor==pwp).mean(),sb.sum()),flush=True)
draw=(pa[sb]-na[sb]).mean(0)  # raw capability shift the password induces
dvt=torch.tensor(draw)
rng=np.random.default_rng(0); rnd=rng.standard_normal(draw.shape); rnd=rnd/np.linalg.norm(rnd)*np.linalg.norm(draw); rndt=torch.tensor(rnd)
sbrows=[r for r,s in zip(rows,sb) if s]
def acc(vec,a):
    STEER["vec"]=vec;STEER["a"]=a; c=sum(fwd(r["question_prompt"])[0]==r["answer"].strip() for r in sbrows); STEER["vec"]=None;STEER["a"]=0.0
    return c/len(sbrows)
print("STEER a=0 baseline no-pwd acc on sandbagged = %.3f (pwd ceiling on these = 1.000)"%acc(None,0),flush=True)
for a in [0.5,1,2,4]:
    print("STEER cap-dir a=%.1f acc=%.3f | random-dir acc=%.3f | opposite(-a) acc=%.3f"%(a,acc(dvt,a),acc(rndt,a),acc(dvt,-a)),flush=True)
print("STEER_DONE",flush=True)
