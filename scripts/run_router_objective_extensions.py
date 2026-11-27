#!/usr/bin/env python3
"""Compare token/document clustering, feature-graph routing, direct overlap routing,
and overlapping feature ownership on held-out document-level SAE features."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans,SpectralClustering

def parse():
 p=argparse.ArgumentParser();p.add_argument('--doc-features',required=True);p.add_argument('--domain-files',required=True);p.add_argument('--feature-depth',required=True);p.add_argument('--out',required=True)
 p.add_argument('--depths',type=int,nargs='+',default=[0,1,2,3]);p.add_argument('--experts',type=int,default=8);p.add_argument('--routing-dim',type=int,default=128);p.add_argument('--feature-seed',type=int,default=42);p.add_argument('--random-repeats',type=int,default=20);p.add_argument('--direct-iters',type=int,default=20);p.add_argument('--min-train-docs',type=int,default=10);p.add_argument('--smoke',action='store_true');return p.parse_args()
def load(a):
 names=[p.stem.split('_',1)[-1] for p in sorted(Path(a.domain_files).glob('*.txt'))];docs=[];y=[]
 for i,p in enumerate(sorted(Path(a.doc_features).glob('doc_features_rank*.npy'),key=lambda p:int(p.stem.split('rank')[-1]))):
  z=np.load(p,allow_pickle=True);docs.extend(np.unique(np.asarray(v,dtype=np.int32)) for v in z);y.extend([i]*len(z))
 dep={};by=defaultdict(list)
 with open(a.feature_depth) as f:
  for r in csv.DictReader(f):q,d=int(r['feature']),int(r['depth']);dep[q]=d;by[d].append(q)
 return docs,np.asarray(y),names,dep,{d:np.asarray(sorted(v),dtype=np.int32) for d,v in by.items()}
def matrix(docs,fs,dep,d):
 pos={int(f):i for i,f in enumerate(fs)};rr=[];cc=[]
 for i,x in enumerate(docs):
  for f in x:
   if dep.get(int(f))==d:rr.append(i);cc.append(pos[int(f)])
 return csr_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(len(docs),len(fs)))
def split_docs(y,rng):
 s=np.empty(len(y),np.int8)
 for q in np.unique(y):
  ix=np.flatnonzero(y==q);rng.shuffle(ix);a=int(.7*len(ix));b=int(.85*len(ix));s[ix[:a]]=0;s[ix[a:b]]=1;s[ix[b:]]=2
 return s
def feature_split(X,tr,seed,mn):
 f=np.asarray(X[tr].sum(0)).ravel();ok=np.flatnonzero(f>=mn);o=ok[np.argsort(f[ok])];rng=np.random.default_rng(seed);R=[];B=[]
 for z in np.array_split(o,min(10,len(o))):
  z=z.copy();rng.shuffle(z);n=max(1,round(.2*len(z)));R.extend(z[:n]);B.extend(z[n:])
 return np.asarray(R),np.asarray(B),f
def choose(X,tr,R,d):
 p=np.asarray(X[tr][:,R].mean(0)).ravel();v=p*(1-p);return R[np.argsort(-v)[:min(d,len(R))]]
def balanced(scores,k):
 n=len(scores);cap=np.full(k,n//k);cap[:n%k]+=1;lab=np.full(n,-1,np.int16)
 for flat in np.argsort(scores.ravel())[::-1]:
  i,e=divmod(int(flat),k)
  if lab[i]<0 and cap[e]>0:lab[i]=e;cap[e]-=1
  if (lab>=0).all():break
 return lab
def kmeans_route(X,tr,te,C,k,seed):
 rms=np.sqrt(np.maximum(np.asarray(X[tr][:,C].mean(0)).ravel(),1e-8));A=X[tr][:,C].toarray()/rms;Q=X[te][:,C].toarray()/rms;it=A.sum(1)>0;iq=Q.sum(1)>0
 km=MiniBatchKMeans(k,n_init=5,max_iter=100,batch_size=2048,random_state=seed).fit(A[it]);lt=np.full(len(tr),-1,np.int16);lq=np.full(len(te),-1,np.int16);lt[it]=km.predict(A[it]);lq[iq]=km.predict(Q[iq]);return lt,lq,it.mean(),iq.mean()
def graph_route(X,tr,te,C,k,seed):
 A=X[tr][:,C].astype(float);freq=np.asarray(A.sum(0)).ravel();co=(A.T@A).toarray();aff=co/np.sqrt(np.maximum(freq[:,None]*freq[None,:],1));np.fill_diagonal(aff,1)
 fg=SpectralClustering(k,affinity='precomputed',assign_labels='kmeans',random_state=seed).fit_predict(aff)
 scale=np.sqrt(np.maximum(freq/len(tr),1e-8));
 def assign(rows):
  Z=X[rows][:,C].toarray()/scale;score=np.stack([Z[:,fg==e].sum(1)/math.sqrt(max(1,(fg==e).sum())) for e in range(k)],1);active=score.max(1)>0;lab=np.full(len(rows),-1,np.int16);lab[active]=score[active].argmax(1);return lab,active.mean()
 return *assign(tr),*assign(te),fg
def direct_route(X,tr,te,C,k,seed,iters):
 A=X[tr][:,C].toarray();Q=X[te][:,C].toarray();active=A.sum(1)>0;aq=Q.sum(1)>0;km=MiniBatchKMeans(k,n_init=3,random_state=seed).fit(A[active]);lab=balanced(-((A[active,None,:]-km.cluster_centers_[None,:,:])**2).sum(2),k);last=None;hist=[]
 for _ in range(iters):
  M=np.stack([A[active][lab==e].sum(0) for e in range(k)]);own=M.argmax(0);score=np.stack([A[active][:,own==e].sum(1) for e in range(k)],1);new=balanced(score,k);leak=1-M.max(0).sum()/max(1,M.sum());hist.append(float(leak))
  if last is not None and np.array_equal(new,lab):break
  last=lab;lab=new
 lt=np.full(len(tr),-1,np.int16);lt[active]=lab;scoreq=np.stack([Q[:,own==e].sum(1) for e in range(k)],1);lq=np.full(len(te),-1,np.int16);lq[aq]=balanced(scoreq[aq],k)
 return lt,lq,active.mean(),aq.mean(),own,hist
def prof(X,rows,lab,B,k):
 P=np.zeros((k,len(B)));L=np.zeros(k,int)
 for e in range(k):
  ix=np.flatnonzero(lab==e);L[e]=len(ix)
  if len(ix):P[e]=np.asarray(X[rows[ix]][:,B].mean(0)).ravel()
 return P,L
def metrics(P,L):
 js=[];cs=[]
 for i in range(len(P)):
  for j in range(i+1,len(P)):
   a,b=P[i],P[j];den=np.maximum(a,b).sum();js.append(np.minimum(a,b).sum()/den if den else np.nan);q=np.linalg.norm(a)*np.linalg.norm(b);cs.append(a@b/q if q else np.nan)
 M=P*L[:,None];den=M.sum(0);v=den>0;q=np.zeros_like(M);q[:,v]=M[:,v]/den[v];pur=q[:,v].max(0);eff=1/np.maximum((q[:,v]**2).sum(0),1e-12)
 return {'jaccard':float(np.nanmean(js)),'cosine':float(np.nanmean(cs)),'purity':float(pur.mean()),'effective_experts':float(eff.mean()),'leakage':float(1-M[:,v].max(0).sum()/M[:,v].sum()),'min_load':int(L.min()),'max_load':int(L.max())}
def ownership(X,tr,te,labtr,labte,B,k):
 M=np.stack([np.asarray(X[tr[labtr==e]][:,B].sum(0)).ravel() for e in range(k)]);order=np.argsort(-M,axis=0);rows=[]
 T=X[te][:,B].tocoo();valid=(labte[T.row]>=0);rr=T.row[valid];cc=T.col[valid];ee=labte[rr];total=len(rr)
 for b in [1,2,3,4,8]:
  b=min(b,k);hit=np.asarray([(order[:b,c]==e).any() for c,e in zip(cc,ee)]);rows.append({'ownership_rule':f'top_{b}','mean_owners':b,'test_event_coverage':float(hit.mean()) if total else np.nan})
 mass=M.sum(0);cum=np.take_along_axis(M,order,axis=0).cumsum(0);validf=mass>0
 for q in [.5,.8,.9,.95,.99]:
  need=np.ones(len(B),int)
  for j in np.flatnonzero(validf):need[j]=int(np.searchsorted(cum[:,j],q*mass[j])+1)
  hit=np.asarray([(order[:need[c],c]==e).any() for c,e in zip(cc,ee)]);rows.append({'ownership_rule':f'train_mass_{q}','mean_owners':float(need[validf].mean()),'test_event_coverage':float(hit.mean()) if total else np.nan})
 return rows
def write(path,rows):
 if not rows:return
 fs=[]
 for r in rows:
  for k in r:
   if k not in fs:fs.append(k)
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,fs);w.writeheader();w.writerows(rows)
def main():
 a=parse();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(42);docs,y,names,dep,fs=load(a);s=split_docs(y,rng);tr=np.flatnonzero(s==0);te=np.flatnonzero(s==2);summary=[];ownrows=[]
 if a.smoke:a.depths=[1];a.random_repeats=2;a.direct_iters=5
 for d in a.depths:
  X=matrix(docs,fs[d],dep,d);R,B,f=feature_split(X,tr,a.feature_seed,a.min_train_docs);C=choose(X,tr,R,a.routing_dim);methods={}
  lt,lq,ct,cq=kmeans_route(X,tr,te,C,a.experts,42+d);methods['kmeans_binary_rms']=(lt,lq,ct,cq,{})
  lt,ct,lq,cq,fg=graph_route(X,tr,te,C,a.experts,42+d);methods['feature_graph']=(lt,lq,ct,cq,{'feature_groups':fg.tolist()})
  lt,lq,ct,cq,owner,hist=direct_route(X,tr,te,C,a.experts,42+d,a.direct_iters);methods['direct_overlap']=(lt,lq,ct,cq,{'train_routing_leakage_history':hist})
  for name,(labtr,labte,ct,cq,extra) in methods.items():
   P,L=prof(X,te,labte,B,a.experts);m=metrics(P,L);valid=np.flatnonzero(labte>=0);null=[]
   for z in range(a.random_repeats):
    q=labte.copy();q[valid]=rng.permutation(q[valid]);null.append(metrics(*prof(X,te,q,B,a.experts))['jaccard'])
   row={'depth':d,'method':name,'routing_features':len(C),'evaluation_features':len(B),'train_coverage':ct,'test_coverage':cq,**m,'random_jaccard':float(np.mean(null)),'relative_reduction':float(1-m['jaccard']/np.mean(null)),**extra};summary.append(row)
   for z in ownership(X,tr,te,labtr,labte,B,a.experts):ownrows.append({'depth':d,'method':name,**z})
 write(out/'routing_summary.csv',summary);write(out/'ownership_coverage.csv',ownrows);(out/'manifest.json').write_text(json.dumps({'status':'complete','documents':len(docs),'config':vars(a),'methods':['kmeans_binary_rms','feature_graph','direct_overlap']},indent=2));print(json.dumps({'rows':len(summary),'ownership_rows':len(ownrows),'out':str(out)}))
if __name__=='__main__':main()
