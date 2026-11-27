#!/usr/bin/env python3
import argparse,csv,json,time
from pathlib import Path
import numpy as np

def fit_ridge(X,Y,l=1e-3):
    A=X.T@X+l*np.eye(X.shape[1]); return np.linalg.solve(A,X.T@Y)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--overlap-csv',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=20000);ap.add_argument('--seed',type=int,default=7);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed);rows=[]
    with open(a.overlap_csv) as f: rows=list(csv.DictReader(f))
    results=[]
    for r in rows:
        ov=float(r['mean_overlap']); E=int(r['experts']); D=128; outdim=32; n=8000; test=2000
        shared=min(D-2,max(0,round(ov*D))); groups=[]
        for e in range(E): groups.append(np.arange(shared+e*(D-shared)//E,shared+(e+1)*(D-shared)//E))
        teacher=rng.standard_normal((D,outdim))/np.sqrt(D); train=[]; target=[]; route=[]
        for i in range(n+test):
            e=int(rng.integers(E)); ids=np.concatenate([rng.choice(shared,size=min(8,shared),replace=False) if shared else np.array([],int),rng.choice(groups[e],size=min(8,len(groups[e])),replace=False)])
            x=np.zeros(D);x[ids]=rng.normal(size=len(ids)); train.append(x);target.append(x@teacher);route.append(e)
        X=np.asarray(train);Y=np.asarray(target);route=np.asarray(route); tr=np.arange(n);te=np.arange(n,n+test)
        wd=fit_ridge(X[tr],Y[tr]); pred=X[te]@wd; dense_mse=float(np.mean((pred-Y[te])**2))
        expert_mse=[]
        for e in range(E):
            mask=route[tr]==e; w=fit_ridge(X[tr][mask],Y[tr][mask]); testmask=route[te]==e; expert_mse.extend(((X[te][testmask]@w-Y[te][testmask])**2).ravel().tolist())
        results.append({'depth':r.get('depth',''),'experts':E,'empirical_overlap':ov,'dense_mse':dense_mse,'expert_mse':float(np.mean(expert_mse)) if expert_mse else None,'dense_params':D*outdim,'expert_params':E*D*outdim,'shared_features':shared})
    with (out/'capacity_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys() if results else ['status']);w.writeheader();w.writerows(results)
    (out/'complete.json').write_text(json.dumps({'method':'empirical-overlap-matched synthetic ridge capacity','rows':len(results)},indent=2))
    print(json.dumps({'rows':len(results),'out':str(out)}))
if __name__=='__main__':main()
