#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def main():
    p=argparse.ArgumentParser();p.add_argument('--activation',type=Path,required=True);p.add_argument('--depth',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--experts',nargs='+',type=int,default=[2,4,8]);p.add_argument('--route-features',type=int,default=32);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    docs=[]
    for f in sorted(a.activation.glob('doc_features_rank*.npy')): docs.extend(np.load(f,allow_pickle=True).tolist())
    docsets=[set(map(int,d)) for d in docs]; depth={}
    with a.depth.open() as f:
        for r in csv.DictReader(f): depth[int(r['feature'])]=int(r['depth'])
    allf=sorted({x for d in docsets for x in d if x in depth}); freq={x:sum(x in d for d in docsets) for x in allf}; rows=[]
    for dep in sorted(set(depth.values())):
        pool=sorted([x for x in allf if depth[x]==dep],key=lambda x:freq[x],reverse=True)
        if len(pool)<3: continue
        n=min(a.route_features,len(pool)-1); rng=np.random.default_rng(123+dep)
        methods={'frequency':pool[:n],'random':list(rng.choice(pool,size=n,replace=False))}
        for method,route_ids in methods.items():
            held=[x for x in pool if x not in set(route_ids)]; X=np.asarray([[int(x in d) for x in route_ids] for d in docsets],dtype=np.float32)
            for k in a.experts:
                if len(docsets)<k: continue
                lab=MiniBatchKMeans(n_clusters=k,random_state=0,n_init=3,batch_size=1024).fit_predict(X); M=np.zeros((k,len(held)))
                for e in range(k):
                    sel=np.flatnonzero(lab==e); M[e]=np.asarray([[int(x in docsets[i]) for x in held] for i in sel]).mean(0) if len(sel) else 0
                ovs=[]
                for i in range(k):
                    for j in range(i+1,k):
                        den=np.maximum(M[i],M[j]).sum(); ovs.append(float(np.minimum(M[i],M[j]).sum()/den) if den else 0.)
                rows.append({'method':method,'depth':dep,'experts':k,'route_features':len(route_ids),'heldout_features':len(held),'mean_overlap':float(np.mean(ovs)) if ovs else 0.,'coverage':float(np.mean(X.sum(1)>0)),'min_load':int(np.bincount(lab,minlength=k).min()),'max_load':int(np.bincount(lab,minlength=k).max())}); np.save(a.out/f'assign_{method}_depth{dep}_k{k}.npy',lab)
    fields=list(rows[0]) if rows else ['method','depth']
    with (a.out/'overlap_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (a.out/'complete.json').write_text(json.dumps({'rows':len(rows),'method':'held-out document activation overlap'},indent=2))
if __name__=='__main__': main()
