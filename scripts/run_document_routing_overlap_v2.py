#!/usr/bin/env python3
"""Document-level SAE routing overlap: feature routing and domain grouping.

Train documents select routing/evaluation features and fit clusters.  Test
documents alone produce overlap metrics.  Domain labels are a second grouping
view, never an input to the feature router.
"""

from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             completeness_score, homogeneity_score,
                             normalized_mutual_info_score, v_measure_score)


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--doc-features',required=True); p.add_argument('--domain-files',required=True)
    p.add_argument('--feature-depth',required=True); p.add_argument('--out',required=True)
    p.add_argument('--depths',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--experts',type=int,nargs='+',default=[4,8,16])
    p.add_argument('--routing-dims',type=int,nargs='+',default=[8,32,64])
    p.add_argument('--feature-split-seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--random-repeats',type=int,default=20)
    p.add_argument('--min-train-docs',type=int,default=10)
    p.add_argument('--train-frac',type=float,default=.7); p.add_argument('--val-frac',type=float,default=.15)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--smoke',action='store_true')
    return p.parse_args()


def load_inputs(a):
    domain_paths=sorted(Path(a.domain_files).glob('*.txt'))
    domains=[p.stem.split('_',1)[-1] for p in domain_paths]
    arrays=sorted(Path(a.doc_features).glob('doc_features_rank*.npy'),key=lambda p:int(p.stem.split('rank')[-1]))
    if len(arrays)!=len(domains): raise ValueError('rank/domain count mismatch')
    docs=[]; labels=[]
    for y,p in enumerate(arrays):
        x=np.load(p,allow_pickle=True)
        docs.extend(np.unique(np.asarray(v,dtype=np.int32)) for v in x); labels.extend([y]*len(x))
    depth={}; by_depth=defaultdict(list)
    with open(a.feature_depth) as f:
        for r in csv.DictReader(f):
            fid,d=int(r['feature']),int(r['depth']); depth[fid]=d; by_depth[d].append(fid)
    return docs,np.asarray(labels,dtype=np.int16),domains,depth,{d:np.asarray(sorted(v),dtype=np.int32) for d,v in by_depth.items()}


def split_docs(labels,a,rng):
    split=np.empty(len(labels),dtype=np.int8)
    for y in np.unique(labels):
        ix=np.flatnonzero(labels==y); rng.shuffle(ix); nt=int(len(ix)*a.train_frac); nv=int(len(ix)*a.val_frac)
        split[ix[:nt]]=0; split[ix[nt:nt+nv]]=1; split[ix[nt+nv:]]=2
    return split


def depth_matrix(docs,features,depth_map,depth):
    pos={int(f):j for j,f in enumerate(features)}; rr=[];cc=[]
    for i,doc in enumerate(docs):
        for f in doc:
            if depth_map.get(int(f))==depth: rr.append(i);cc.append(pos[int(f)])
    return csr_matrix((np.ones(len(rr),dtype=np.float32),(rr,cc)),shape=(len(docs),len(features)))


def feature_split(X,train,seed,min_docs):
    freq=np.asarray(X[train].sum(0)).ravel(); eligible=np.flatnonzero(freq>=min_docs)
    if len(eligible)<10: return np.array([],int),np.array([],int),freq
    # Frequency-decile stratification keeps routing and evaluation pools comparable.
    order=eligible[np.argsort(freq[eligible],kind='stable')]; bins=np.array_split(order,min(10,len(order)))
    rng=np.random.default_rng(seed); route=[]; evaluate=[]
    for b in bins:
        b=b.copy();rng.shuffle(b);n=max(1,int(round(.2*len(b))));route.extend(b[:n]);evaluate.extend(b[n:])
    return np.asarray(route),np.asarray(evaluate),freq


def route(X,train,test,route_pool,freq,d,k,seed):
    variance=np.asarray(X[train][:,route_pool].mean(0)).ravel(); variance=variance*(1-variance)
    chosen=route_pool[np.argsort(-variance,kind='stable')[:min(d,len(route_pool))]]
    rms=np.sqrt(np.maximum(np.asarray(X[train][:,chosen].mean(0)).ravel(),1e-8))
    Xt=X[train][:,chosen].toarray()/rms; Xq=X[test][:,chosen].toarray()/rms
    informative_train=Xt.sum(1)>0; informative_test=Xq.sum(1)>0
    km=MiniBatchKMeans(n_clusters=k,n_init=5,max_iter=100,batch_size=2048,random_state=seed)
    km.fit(Xt[informative_train]); lab=np.full(len(test),-1,dtype=np.int16)
    lab[informative_test]=km.predict(Xq[informative_test])
    return lab,chosen,informative_test,km


def profiles(Xtest,assignment,features,groups):
    P=np.zeros((groups,len(features)),dtype=np.float64); loads=np.zeros(groups,dtype=np.int64)
    for e in range(groups):
        ix=np.flatnonzero(assignment==e);loads[e]=len(ix)
        if len(ix):P[e]=np.asarray(Xtest[ix][:,features].mean(0)).ravel()
    return P,loads


def pair_metrics(P,loads):
    rows=[]
    for i in range(len(P)):
        for j in range(i+1,len(P)):
            a,b=P[i],P[j]; mn=np.minimum(a,b).sum();mx=np.maximum(a,b).sum()
            cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else math.nan
            support_a=a>0;support_b=b>0;union=(support_a|support_b).sum()
            # Minimal feature sets carrying 90% of profile mass.
            def topmass(v):
                if v.sum()==0:return np.zeros(len(v),bool)
                o=np.argsort(-v); n=np.searchsorted(np.cumsum(v[o]),.9*v.sum())+1;m=np.zeros(len(v),bool);m[o[:n]]=1;return m
            ta,tb=topmass(a),topmass(b);tu=(ta|tb).sum()
            rows.append({'expert_i':i,'expert_j':j,'load_i':int(loads[i]),'load_j':int(loads[j]),
                         'weighted_jaccard':float(mn/mx) if mx else math.nan,'cosine':cos,
                         'support_jaccard':float((support_a&support_b).sum()/union) if union else math.nan,
                         'top90_jaccard':float((ta&tb).sum()/tu) if tu else math.nan,
                         'l1_distance':float(np.abs(a-b).sum()),'pair_weight':int(loads[i]*loads[j])})
    return rows


def aggregate_pairs(rows):
    out={}
    for m in ['weighted_jaccard','cosine','support_jaccard','top90_jaccard','l1_distance']:
        valid=[r for r in rows if not math.isnan(r[m])]; w=np.asarray([r['pair_weight'] for r in valid],float)
        out[m+'_macro']=float(np.mean([r[m] for r in valid])) if valid else math.nan
        out[m+'_weighted']=float(np.average([r[m] for r in valid],weights=w)) if valid and w.sum() else math.nan
    return out


def feature_metrics(P,loads):
    M=P*loads[:,None];den=M.sum(0);valid=den>0;q=np.zeros_like(M);q[:,valid]=M[:,valid]/den[valid]
    purity=q[:,valid].max(0);eff=1/np.maximum((q[:,valid]**2).sum(0),1e-12)
    leakage=1-M[:,valid].max(0).sum()/M[:,valid].sum() if valid.any() else math.nan
    return {'evaluated_features':int(valid.sum()),'feature_purity_macro':float(purity.mean()) if len(purity) else math.nan,
            'effective_experts_macro':float(eff.mean()) if len(eff) else math.nan,'event_leakage':float(leakage)}


def load_metrics(loads):
    p=loads/loads.sum() if loads.sum() else np.zeros_like(loads,dtype=float)
    return {'min_load':int(loads.min()) if len(loads) else 0,'max_load':int(loads.max()) if len(loads) else 0,
            'load_imbalance_max_over_mean':float(loads.max()/loads.mean()) if len(loads) and loads.mean() else math.nan,
            'effective_loaded_experts':float(1/(p@p)) if p.sum() else 0.0}


def alignment(cluster,domain):
    mask=cluster>=0;c=cluster[mask];y=domain[mask]
    if not len(c):return {}
    tab=np.zeros((int(c.max())+1,int(y.max())+1),dtype=int)
    np.add.at(tab,(c,y),1);ri,ci=linear_sum_assignment(-tab)
    return {'nmi':float(normalized_mutual_info_score(y,c)),'ami':float(adjusted_mutual_info_score(y,c)),
            'ari':float(adjusted_rand_score(y,c)),'homogeneity':float(homogeneity_score(y,c)),
            'completeness':float(completeness_score(y,c)),'v_measure':float(v_measure_score(y,c)),
            'hungarian_accuracy':float(tab[ri,ci].sum()/len(c)),'contingency':tab.tolist()}


def write_csv(path,rows):
    if not rows:return
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with open(path,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    a=args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed)
    docs,labels,domains,depth_map,features=load_inputs(a);split=split_docs(labels,a,rng)
    train=np.flatnonzero(split==0);test=np.flatnonzero(split==2)
    matrices={d:depth_matrix(docs,features[d],depth_map,d) for d in a.depths}
    if a.smoke:a.experts=[8];a.routing_dims=[32];a.feature_split_seeds=[42];a.random_repeats=2
    summary=[];pairs=[];alignments=[];assignments=[];feature_splits=[];comparisons=[]
    # Domain-label grouping is evaluated once per evaluation depth.
    for ed in a.depths:
        ep=np.flatnonzero(np.asarray(matrices[ed][train].sum(0)).ravel()>=a.min_train_docs)
        P,L=profiles(matrices[ed][test],labels[test],ep,len(domains));pr=pair_metrics(P,L);agg=aggregate_pairs(pr)
        row={'method':'domain','assignment_variant':'all_documents','routing_depth':'domain','evaluation_depth':ed,'experts':len(domains),'routing_dim':0,'split_seed':a.seed,**agg,**feature_metrics(P,L),**load_metrics(L)}
        summary.append(row)
        for x in pr:pairs.append({**{k:row[k] for k in ['method','routing_depth','evaluation_depth','experts','routing_dim','split_seed']},**x})
    for fs in a.feature_split_seeds:
      split_info={d:feature_split(matrices[d],train,fs,a.min_train_docs) for d in a.depths}
      for d,(rp,ep,freq) in split_info.items():
        for j in rp:feature_splits.append({'split_seed':fs,'depth':d,'feature':int(features[d][j]),'role':'routing','train_docs':int(freq[j])})
        for j in ep:feature_splits.append({'split_seed':fs,'depth':d,'feature':int(features[d][j]),'role':'evaluation','train_docs':int(freq[j])})
      for rd in a.depths:
       rp,_,freq=split_info[rd]
       for k in a.experts:
        for dim in a.routing_dims:
         if len(rp)<dim:continue
         lab,chosen,covered,_=route(matrices[rd],train,test,rp,freq,dim,k,a.seed+fs+rd+k+dim)
         variants={'informative_only':lab}
         lab_hash=lab.copy(); zero=np.flatnonzero(lab_hash<0)
         lab_hash[zero]=((test[zero].astype(np.int64)*2654435761+a.seed)%k).astype(np.int16)
         variants['hash_zero_all_documents']=lab_hash
         for variant,current_lab in variants.items():
          al=alignment(current_lab,labels[test]);alignments.append({'split_seed':fs,'routing_depth':rd,'experts':k,'routing_dim':dim,'assignment_variant':variant,'coverage':float(covered.mean()),**al})
          for ix,e in zip(test,current_lab):assignments.append({'split_seed':fs,'routing_depth':rd,'experts':k,'routing_dim':dim,'assignment_variant':variant,'doc_index':int(ix),'domain':domains[int(labels[ix])],'expert':int(e)})
          for ed in a.depths:
           ep=split_info[ed][1];P,L=profiles(matrices[ed][test],current_lab,ep,k);pr=pair_metrics(P,L);agg=aggregate_pairs(pr)
           subset=current_lab>=0;DP,DL=profiles(matrices[ed][test][subset],labels[test][subset],ep,len(domains));dpr=pair_metrics(DP,DL);dagg=aggregate_pairs(dpr)
           comparisons.append({'split_seed':fs,'routing_depth':rd,'evaluation_depth':ed,'experts':k,'routing_dim':dim,'assignment_variant':variant,
                               'documents':int(subset.sum()),'feature_weighted_jaccard':agg['weighted_jaccard_macro'],
                               'domain_weighted_jaccard_same_documents':dagg['weighted_jaccard_macro'],
                               'feature_minus_domain_jaccard':agg['weighted_jaccard_macro']-dagg['weighted_jaccard_macro']})
           base=[];valid=np.flatnonzero(current_lab>=0)
           for repeat in range(a.random_repeats):
            perm=current_lab.copy();perm[valid]=rng.permutation(perm[valid]);RP,RL=profiles(matrices[ed][test],perm,ep,k);base.append(aggregate_pairs(pair_metrics(RP,RL))['weighted_jaccard_macro'])
           row={'method':'feature','assignment_variant':variant,'routing_depth':rd,'evaluation_depth':ed,'experts':k,'routing_dim':dim,'split_seed':fs,
                'coverage':float(covered.mean()),'routing_features':len(chosen),'evaluation_features':len(ep),**agg,**feature_metrics(P,L),**load_metrics(L),
                'random_weighted_jaccard_macro':float(np.mean(base)),
                'jaccard_absolute_reduction':float(np.mean(base)-agg['weighted_jaccard_macro']),
                'jaccard_relative_reduction':float(1-agg['weighted_jaccard_macro']/np.mean(base)) if np.mean(base)>0 else math.nan}
           summary.append(row)
           for x in pr:pairs.append({**{z:row[z] for z in ['method','assignment_variant','routing_depth','evaluation_depth','experts','routing_dim','split_seed']},**x})
    write_csv(out/'summary.csv',summary);write_csv(out/'pairwise_metrics.csv',pairs);write_csv(out/'feature_splits.csv',feature_splits);write_csv(out/'assignments.csv',assignments);write_csv(out/'feature_vs_domain.csv',comparisons)
    (out/'alignment.json').write_text(json.dumps(alignments,indent=2));(out/'document_split.csv').write_text('doc_index,domain,split\n'+'\n'.join(f'{i},{domains[int(y)]},{["train","validation","test"][int(s)]}' for i,(y,s) in enumerate(zip(labels,split))))
    manifest={'status':'complete','documents':len(docs),'domains':domains,'depths':a.depths,'rows':len(summary),'config':vars(a),
              'definitions':{'feature_routing':'train-fit clusters from 20% routing features; test overlap on held-out 80%',
                             'domain_grouping':'eight observed SFT domain labels; no clustering','unit':'document-level binary feature presence'}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str));print(json.dumps(manifest,indent=2,default=str))
if __name__=='__main__':main()
