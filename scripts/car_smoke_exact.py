import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--minimum-count',type=int,default=16);p.add_argument('--containment',type=float,default=.85);p.add_argument('--reverse',type=float,default=.85);p.add_argument('--frequency-ratio',type=float,default=1.10);p.add_argument('--joint-containment',type=float,default=.75);p.add_argument('--minimum-constituents',type=int,default=2);p.add_argument('--maximum-constituents',type=int,default=8);p.add_argument('--max-base-candidates',type=int,default=4096);a=p.parse_args();t=time.time();out=Path(a.out)
 docs=[]
 for f in sorted(out.glob('doc_features_rank*.npy')): docs.extend(np.load(f,allow_pickle=True).tolist())
 nf=131072; rows=[];cols=[]
 for i,fs in enumerate(docs):
  u=np.unique(np.asarray(fs,dtype=np.int32));rows.extend([i]*len(u));cols.extend(u.tolist())
 x=sp.csr_matrix((np.ones(len(rows),dtype=np.uint8),(rows,cols)),shape=(len(docs),nf)); x.eliminate_zeros(); x.data[:]=1; freq=np.asarray(x.sum(0)).ravel().astype(np.int64)
 order=np.lexsort((np.arange(nf),-freq));pool=order[freq[order]>0][:a.max_base_candidates];base=x[:,pool];xc=x.tocsc();postings=[xc.indices[xc.indptr[f]:xc.indptr[f+1]] for f in range(nf)];parents={};evidence=[]
 for lo in range(0,nf,128):
  co=(x[:,lo:lo+128].T @ base).tocsr()
  for local in range(co.shape[0]):
   f=lo+local
   if freq[f]<a.minimum_count: continue
   start,end=co.indptr[local:local+2];cand=[]
   for pos,joint in zip(co.indices[start:end],co.data[start:end]):
    g=int(pool[pos]);containment=joint/freq[f];reverse=joint/freq[g];ratio=freq[g]/freq[f]
    if g!=f and containment>=a.containment and reverse<a.reverse and ratio>=a.frequency_ratio:cand.append((containment,ratio,g))
   cand.sort(key=lambda v:(-v[0],-v[1],v[2]));selected=[];together=postings[f]
   for _,_,g in cand:
    if len(selected)>=a.maximum_constituents:break
    proposed=np.intersect1d(together,postings[g],assume_unique=True)
    if len(proposed)/freq[f]>=a.joint_containment:selected.append(g);together=proposed
   if len(selected)>=a.minimum_constituents:parents[f]=selected;evidence.append((f,selected,int(freq[f]),len(together)/freq[f]))
 depth=np.zeros(nf,dtype=np.int32)
 for f in order:
  if int(f) in parents: depth[f]=1+max(depth[g] for g in parents[int(f)])
 with (out/'car_exact_edges.csv').open('w') as h:
  h.write('parent,child,child_frequency,joint_containment\n')
  for f,gs,fr,jc in evidence:
   for g in gs:h.write(f'{g},{f},{fr},{jc}\n')
 with (out/'car_exact_labels.csv').open('w') as h:
  h.write('feature,frequency,depth,label\n');h.writelines(f'{f},{freq[f]},{depth[f]},{"composite" if depth[f]>0 else "leaf"}\n' for f in np.flatnonzero(freq>0))
 (out/'car_exact_complete.json').write_text(json.dumps({'documents':len(docs),'active_features':int((freq>0).sum()),'accepted_composites':len(evidence),'edges':int(sum(len(x[1]) for x in evidence)),'max_depth':int(depth.max(initial=0)),'minimum_count':a.minimum_count,'legacy_parameters':vars(a),'elapsed_s':time.time()-t},indent=2))
 print(json.dumps({'documents':len(docs),'active_features':int((freq>0).sum()),'accepted_composites':len(evidence),'edges':int(sum(len(x[1]) for x in evidence)),'max_depth':int(depth.max(initial=0)),'elapsed_s':time.time()-t}),flush=True)
if __name__=='__main__':main()
