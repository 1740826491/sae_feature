#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np

def ridge(X,Y,l): return np.linalg.solve(X.T@X+l*np.eye(X.shape[1]),X.T@Y)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--overlap-csv',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=11);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed);res=[]
 with open(a.overlap_csv) as f: rows=list(csv.DictReader(f))
 for r in rows:
  ov=float(r['empirical_overlap']) if 'empirical_overlap' in r else float(r['mean_overlap']);E=int(r['experts']);D=256;rank=4;Ydim=64;n=6000;nt=2000;shared=max(1,min(D-2,round(ov*D)));teacher=rng.standard_normal((D,Ydim))/np.sqrt(D)
  groups=[np.arange(shared+e*(D-shared)//E,shared+(e+1)*(D-shared)//E) for e in range(E)];X=[];Y=[];R=[]
  for _ in range(n+nt):
   e=int(rng.integers(E)); ids=np.concatenate([rng.choice(shared,min(12,shared),replace=False),rng.choice(groups[e],min(12,len(groups[e])),replace=False)]);x=np.zeros(D);x[ids]=rng.normal(size=len(ids));X.append(x);Y.append(x@teacher);R.append(e)
  X=np.asarray(X);Y=np.asarray(Y);R=np.asarray(R);tr=np.arange(n);te=np.arange(n,n+nt);wd=ridge(X[tr],Y[tr],1e-2);dense=float(np.mean((X[te]@wd-Y[te])**2));ems=[]
  for e in range(E):
   m=R[tr]==e; mt=R[te]==e
   if m.sum() and mt.sum():
    # rank-constrained expert: low-dimensional bottleneck projection
    U=rng.standard_normal((D,rank))/np.sqrt(D);A=ridge(X[tr][m]@U,Y[tr][m],1e-2);ems.extend(((X[te][mt]@U@A-Y[te][mt])**2).ravel().tolist())
  res.append({'depth':r.get('depth',''),'experts':E,'overlap':ov,'dense_mse':dense,'lowrank_expert_mse':float(np.mean(ems)) if ems else None,'dense_params':D*Ydim,'expert_rank_params':E*(D*rank+rank*Ydim),'shared_features':shared})
 with (out/'bottleneck_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=res[0].keys());w.writeheader();w.writerows(res)
 (out/'complete.json').write_text(json.dumps({'method':'overlap-matched low-rank bottleneck capacity','rows':len(res)},indent=2));print(json.dumps({'rows':len(res)}))
if __name__=='__main__':main()
