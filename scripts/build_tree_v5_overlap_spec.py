#!/usr/bin/env python3
"""Build the canonical 1-2-8 Tree-V5 spec with conservative overlap."""
import argparse,json
from pathlib import Path
import numpy as np

SETS={"ideal":[0,0],"real_max":[0.4057347404,0.4057347404],
      "random_assignment":[0.827406,0.827406]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--sae',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=int,default=128);p.add_argument('--seed',type=int,default=20261127);a=p.parse_args()
 W=np.load(a.sae)['W_dec'];rng=np.random.default_rng(a.seed)
 for name,qs in SETS.items():
  nodes=[{'name':'R','level':0,'parent':None}];front=[0]
  for l,b in enumerate([2,4],1):
   nxt=[]
   for parent in front:
    for j in range(b):nodes.append({'name':f"{nodes[parent]['name']}.{j}",'level':l,'parent':parent});nxt.append(len(nodes)-1)
   front=nxt
  bylevel=[[i for i,x in enumerate(nodes) if x['level']==l] for l in range(3)]
  need=a.n
  for l,q in enumerate(qs,1):
   c=int(round(2*q*a.n/(1+q)));need+=c+len(bylevel[l])*(a.n-c)
  ids=rng.choice(W.shape[0],W.shape[0],replace=False);cur=0
  root=ids[cur:cur+a.n];cur+=a.n;nodes[0].update(features=root.tolist(),freq=[1.]*a.n,activity=8.)
  for l,q in enumerate(qs,1):
   c=int(round(2*q*a.n/(1+q)));common=ids[cur:cur+c];cur+=c
   for ni in bylevel[l]:
    private=ids[cur:cur+a.n-c];cur+=a.n;pool=np.concatenate([common,private]).astype(np.int64,copy=False)
    nodes[ni].update(features=pool.tolist(),freq=[1.]*a.n,activity=8.)
  paths=[]
  for leaf in front:
   x=[];v=leaf
   while v is not None:x.append(v);v=nodes[v]['parent']
   paths.append(x[::-1])
  unique=np.unique(np.concatenate([np.asarray(x['features'],dtype=np.int64) for x in nodes])).astype(np.int64);U=W[unique].astype(np.float32);U/=np.linalg.norm(U,axis=1,keepdims=True)
  out=a.out/name;out.mkdir(parents=True,exist_ok=True);np.save(out/'U.npy',U)
  json.dump({'levels':3,'dim':int(W.shape[1]),'nodes':nodes,'paths':paths,'path_probs':[1/len(paths)]*len(paths),'U_feature_ids':unique.tolist(),'overlap_name':name,'target_overlaps':qs},(out/'tree.json').open('w'))
  print(name,'nodes',len(nodes),'paths',len(paths),'unique',len(unique))
if __name__=='__main__':main()
