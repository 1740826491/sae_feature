#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans

def load_docs(inp):
    docs=[]
    for p in sorted(Path(inp).glob('doc_features_rank*.npy')):
        docs.extend(np.load(p,allow_pickle=True).tolist())
    return [np.unique(np.asarray(x,dtype=np.int32)) for x in docs]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--depth-purity',required=True);ap.add_argument('--out',required=True);ap.add_argument('--experts',type=int,nargs='+',default=[2,4,8]);ap.add_argument('--route-k',type=int,default=32);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);docs=load_docs(a.input)
    depth=defaultdict(list)
    with open(a.depth_purity) as f:
        for r in csv.DictReader(f):
            try: freq=float(r.get('frequency','') or 0.0)
            except ValueError: freq=0.0
            depth[int(r['depth'])].append((int(r['feature']),freq))
    summaries=[]
    for d,items in sorted(depth.items()):
        items=sorted(items,key=lambda x:x[1],reverse=True); route=np.array([x[0] for x in items[:a.route_k]],dtype=np.int32); held=np.array([x[0] for x in items[a.route_k:]],dtype=np.int32)
        if len(route)<2 or len(held)<2: continue
        rid={int(x):i for i,x in enumerate(route)}; X=np.zeros((len(docs),len(route)),dtype=np.float32); H=np.zeros((len(docs),len(held)),dtype=np.float32); hid={int(x):i for i,x in enumerate(held)}
        for n,ds in enumerate(docs):
            for x in ds:
                if int(x) in rid:X[n,rid[int(x)]]=1
                if int(x) in hid:H[n,hid[int(x)]]=1
        for E in a.experts:
            if len(docs)<E: continue
            labels=MiniBatchKMeans(n_clusters=E,random_state=0,n_init=3,batch_size=1024,max_iter=100).fit_predict(X)
            loads=np.bincount(labels,minlength=E); rates=[]
            for e in range(E): rates.append(H[labels==e].mean(0) if loads[e] else np.zeros(H.shape[1]))
            vals=[]
            for i in range(E):
                for j in range(i+1,E):
                    mn=np.minimum(rates[i],rates[j]).sum();mx=np.maximum(rates[i],rates[j]).sum();vals.append(float(mn/mx) if mx else 0.0)
            summaries.append({'depth':d,'experts':E,'route_features':len(route),'heldout_features':len(held),'documents':len(docs),'mean_overlap':float(np.mean(vals)) if vals else 0.0,'loads':loads.tolist(),'coverage':float(np.mean(H.sum(1)>0))})
    (out/'routing_overlap_summary.json').write_text(json.dumps({'method':'depth-selected document-level offline KMeans routing','results':summaries},indent=2))
    with (out/'routing_overlap_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['depth','experts','route_features','heldout_features','documents','mean_overlap','coverage','loads']);w.writeheader();w.writerows(summaries)
    print(json.dumps({'rows':len(summaries),'out':str(out)}))
if __name__=='__main__':main()
