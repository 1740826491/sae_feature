#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
def load_parts(root):
 rows=[]
 for f in sorted((Path(root)/'token_activations').glob('*.npz')):
  z=np.load(f); rows.append((z['feature_ids'],z['activation_values']))
 if not rows: raise FileNotFoundError('no token activation parts')
 return np.concatenate([x[0] for x in rows]),np.concatenate([x[1] for x in rows])
def jac(a,b): return len(a&b)/len(a|b) if a or b else 0.0
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--activation',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed)
 ids,vals=load_parts(a.activation);n=ids.shape[0];active=ids[vals>0];freq=np.bincount(active[active>=0],minlength=131072);usable=np.flatnonzero(freq>=40);usable=usable[np.argsort(freq[usable])[::-1]];pool=usable[:min(1000,len(usable))];pos={int(f):i for i,f in enumerate(pool)};X=np.zeros((n,len(pool)),np.float32)
 for i in range(n):
  for j,f in enumerate(ids[i]):
   p=pos.get(int(f))
   if p is not None and vals[i,j]>0:X[i,p]=vals[i,j]
 order=rng.permutation(n);tr=order[:int(.7*n)];te=order[int(.85*n):];rows=[]
 for name,lo,hi in [('high',0,333),('mid',333,666),('low',666,1000)]:
  Z=X[:,lo:min(hi,X.shape[1])]
  if Z.shape[1]<2:continue
  for d in [min(8,Z.shape[1]),min(32,Z.shape[1]),min(64,Z.shape[1])]:
   keep=np.argsort(Z[tr].var(0))[::-1][:d];R=Z[:,keep]
   for K in [2,4,8,16]:
    km=MiniBatchKMeans(n_clusters=K,random_state=a.seed,n_init=3,batch_size=2048,max_iter=80).fit(R[tr]);lab=km.predict(R[te]);sets=[set(np.flatnonzero(lab==k).tolist()) for k in range(K)];r=rng.integers(0,K,len(te));rs=[set(np.flatnonzero(r==k).tolist()) for k in range(K)];pair=lambda ss:float(np.mean([jac(ss[i],ss[j]) for i in range(K) for j in range(i+1,K)])) if K>1 else 0.0;rows.append({'stage':'E1','group':name,'dim':int(d),'K':K,'assignment_jaccard':pair(sets),'random_jaccard':pair(rs),'loads':np.bincount(lab,minlength=K).tolist()})
   for q in [min(8,Z.shape[1]),min(32,Z.shape[1]),min(64,Z.shape[1])]:
    p=PCA(n_components=q,random_state=a.seed).fit(Z[tr]);rows.append({'stage':'E3','group':name,'dim':int(q),'explained_variance':float(p.explained_variance_ratio_.sum())})
   km=MiniBatchKMeans(n_clusters=2,random_state=a.seed,n_init=3,batch_size=2048,max_iter=80).fit(R[tr]);root=km.predict(R[te]);rows += [{'stage':'E2','group':name,'dim':int(d),'depth':1,'leaf_loads':np.bincount(root,minlength=2).tolist()},{'stage':'E2','group':name,'dim':int(d),'depth':2,'leaf_count':4,'parent_loads':np.bincount(root,minlength=2).tolist()}]
 supports={int(f):set(np.flatnonzero(np.any(ids==f,axis=1)).tolist()) for f in pool[:200]};fs=list(supports);v=[]
 for i,f in enumerate(fs):
  for g in fs[i+1:]:v.append(jac(supports[f],supports[g]))
 for q in [0,.25,.5,.75,1]:
  s=[x for x in v if x>=q];rows.append({'stage':'E4','threshold':q,'pair_count':len(s),'mean_support_jaccard':float(np.mean(s)) if s else 0.0})
 base=np.argmax(X[te],axis=1)
 with (out/'E6_corruption.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['corruption','changed_assignment_fraction','marginal_l1'])
  for alpha in [0,.1,.25,.5,.75,1]:
   c=base.copy();m=rng.random(len(c))<alpha;c[m]=rng.permutation(c[m]);w.writerow([alpha,float(np.mean(c!=base)),float(np.abs(np.bincount(c,minlength=X.shape[1])-np.bincount(base,minlength=X.shape[1])).sum())])
 (out/'E1_E4_results.json').write_text(json.dumps(rows,indent=2));(out/'complete.json').write_text(json.dumps({'stages':['E1','E2','E3','E4','E6'],'tokens':int(n),'pool_features':int(len(pool)),'train_tokens':int(len(tr)),'test_tokens':int(len(te))},indent=2));print(json.dumps({'tokens':int(n),'pool_features':int(len(pool))}))
if __name__=='__main__':main()
