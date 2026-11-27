#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np

def train(X,Y,groups,E,steps=1200,lr=.08):
    D=X.shape[1]; O=Y.shape[1]; W=np.zeros((E,D,O),dtype=np.float64); losses=[]
    for s in range(steps):
        pred=np.einsum('nd,ndo->no',X,W[groups]); err=pred-Y; losses.append(float(np.mean(err*err)))
        for e in range(E):
            m=groups==e
            if m.any(): W[e]-=lr*(X[m].T@err[m])/max(1,m.sum())
    return float(losses[-1]), next((i for i,x in enumerate(losses) if x <= losses[-1]*1.10),steps), losses[::max(1,steps//20)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--overlap-csv',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=13);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed);rows=[]
 with open(a.overlap_csv) as f: rows=list(csv.DictReader(f))
 res=[]
 for r in rows:
  E=int(r['experts']);ov=float(r['mean_overlap']);D=64;O=16;n=5000;shared=max(1,min(D-2,round(ov*D)));teacher=rng.standard_normal((D,O))/np.sqrt(D);X=np.zeros((n,D));Y=np.zeros((n,O));G=rng.integers(E,size=n)
  for e in range(E):
   m=G==e; private=np.arange(shared+e*max(1,(D-shared)//E),min(D,shared+(e+1)*max(1,(D-shared)//E))); ids=np.concatenate([np.arange(shared),private]); ids=ids[:max(2,min(16,len(ids)))]; X[m[:,None] if False else m[:,],:][:,ids]=rng.normal(size=(m.sum(),len(ids))) if False else X[m][:,ids]
   if m.any():
    X[m[:,],:][:,ids]=rng.normal(size=(m.sum(),len(ids)))
  Y=X@teacher; dense,ds,curve=train(X,Y,np.zeros(n,dtype=int),1,steps=1200); expert,es,ecurve=train(X,Y,G,E,steps=1200)
  res.append({'depth':r.get('depth',''),'experts':E,'empirical_overlap':ov,'dense_final_mse':dense,'expert_final_mse':expert,'dense_steps':ds,'expert_steps':es,'dense_params':D*O,'expert_params':E*D*O,'shared_features':shared})
 with (out/'fixed_assignment_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=res[0].keys());w.writeheader();w.writerows(res)
 (out/'complete.json').write_text(json.dumps({'method':'fixed assignment synthetic gradient training','rows':len(res),'steps':1200},indent=2));print(json.dumps({'rows':len(res)}))
if __name__=='__main__':main()
