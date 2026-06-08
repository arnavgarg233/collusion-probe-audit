import numpy as np, torch, csv, urllib.request, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
BASE="mistralai/Mistral-7B-v0.1"; ADP="FelixHofstaetter/mistral-7b-sandbagging-new"; LETTERS=["A","B","C","D"]
tok=AutoTokenizer.from_pretrained(ADP)
m=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float16,device_map="cuda")
m.config.pad_token_id=tok.pad_token_id; m.resize_token_embeddings(len(tok),mean_resizing=False)
m=PeftModel.from_pretrained(m,ADP).merge_and_unload().eval(); print("[loaded]",flush=True)
cand=[tok(" "+L,add_special_tokens=False).input_ids[-1] for L in LETTERS]
fp="/workspace/sb/wmdp.csv"; os.makedirs("/workspace/sb",exist_ok=True)
if not os.path.exists(fp): urllib.request.urlretrieve("https://raw.githubusercontent.com/Felhof/sandbagging-elicitation/main/data/mcqa/wmdp_test.csv",fp)
rows=list(csv.DictReader(open(fp)))[:300]
LAYERS=[16,20,24,28]; STEER={"vec":None,"a":0.0,"L":None}; cap={}
def mk(L):
    def hook(mod,inp,out):
        h=out[0] if isinstance(out,tuple) else out
        cap[L]=h.detach()[:,-1,:].float().cpu().numpy()
        if STEER["vec"] is not None and STEER["a"]!=0.0 and STEER["L"]==L:
            h[:, :, :]=h[:, :, :]+STEER["a"]*STEER["vec"].to(h.dtype).to(h.device)  # ALL positions
        return out
    return hook
for L in LAYERS: m.model.layers[L].register_forward_hook(mk(L))
@torch.no_grad()
def fwd(p):
    ids=tok(p,return_tensors="pt",truncation=True,max_length=512).to(m.device)
    o=m(**ids); pred=LETTERS[int(torch.tensor([o.logits[0,-1][c] for c in cand]).argmax())]
    return pred, {L:cap[L][0].copy() for L in LAYERS}
STEER["vec"]=None; na={L:[] for L in LAYERS}; pa={L:[] for L in LAYERS}; cor=[];npp=[];pwp=[]
for r in rows:
    pn,an=fwd(r["question_prompt"]); pp,ap=fwd(r["question_prompt_with_password"])
    for L in LAYERS: na[L].append(an[L]); pa[L].append(ap[L])
    cor.append(r["answer"].strip()); npp.append(pn); pwp.append(pp)
cor=np.array(cor);npp=np.array(npp);pwp=np.array(pwp); sb=(npp!=cor)&(pwp==cor)
print("no-pwd %.3f pwd %.3f sandbagged %d"%((cor==npp).mean(),(cor==pwp).mean(),sb.sum()),flush=True)
sbrows=[r for r,s in zip(rows,sb) if s]; rng=np.random.default_rng(0)
def acc(vec,a,L):
    STEER["vec"]=vec;STEER["a"]=a;STEER["L"]=L; c=sum(fwd(r["question_prompt"])[0]==r["answer"].strip() for r in sbrows); STEER["vec"]=None;STEER["a"]=0.0;STEER["L"]=None
    return c/len(sbrows)
for L in LAYERS:
    draw=(np.array(pa[L])[sb]-np.array(na[L])[sb]).mean(0); dvt=torch.tensor(draw)
    rnd=rng.standard_normal(draw.shape); rnd=rnd/np.linalg.norm(rnd)*np.linalg.norm(draw); rndt=torch.tensor(rnd)
    print("LAYER %d base=%.3f (all-position steering)"%(L,acc(None,0,L)),flush=True)
    for a in [1,2,4]:
        print("  L%d a=%.1f cap=%.3f rand=%.3f opp=%.3f"%(L,a,acc(dvt,a,L),acc(rndt,a,L),acc(dvt,-a,L)),flush=True)
print("STEER2_DONE",flush=True)
