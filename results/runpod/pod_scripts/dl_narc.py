import json,os,urllib.request
from concurrent.futures import ThreadPoolExecutor as T
t=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/aaronrose227/narcbench/tree/main?recursive=1'))
B='https://huggingface.co/datasets/aaronrose227/narcbench/resolve/main';R='/tmp/narc/slice'
j=[x['path'] for x in t if x['type']=='file' and (x['path'].startswith('activations/') or x['path'].startswith('scenarios/'))]
[os.makedirs(os.path.dirname(os.path.join(R,p)),exist_ok=True) for p in j]
ok=0
def dl(p):
 global ok
 try: urllib.request.urlretrieve(B+'/'+p,os.path.join(R,p)); return 1
 except: return 0
import sys
r=list(T(16).map(dl,j)); print('NARC_DL_DONE',sum(r),'/',len(j))
