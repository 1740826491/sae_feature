#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--activation',required=True);p.add_argument('--hierarchy',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 parts=sorted((Path(a.activation)/'token_activations').glob('*.npz'));counts=np.zeros((8,131072),np.int64);tokens=np.zeros(8,np.int64);names={}
 for q in parts:
  z=np.load(q);rank=int(z['domain_id']);names[rank]=str(z['domain']);ids=z['feature_ids'];vals=z['activation_values'];tokens[rank]+=ids.shape[0]
  for row in range(len(ids)):
   active=np.unique(ids[row][np.isfinite(vals[row])&(vals[row]>0)]);counts[rank,active]+=1
 depth={}
 with open(Path(a.hierarchy)/'feature_depth.csv') as f:
  for r in csv.DictReader(f):depth[int(r['feature'])]=int(r['depth'])
 rows=[]
 for fid in np.flatnonzero(counts.sum(0)):
  c=counts[:,fid].astype(float);pdom=c/c.sum();ent=-sum(x*math.log(x) for x in pdom if x>0);best=int(pdom.argmax());rows.append({'feature':int(fid),'depth':depth.get(int(fid),''),'activation_tokens':int(c.sum()),'domain_entropy':ent,'specificity':1-ent/math.log(8),'dominant_domain':names.get(best,str(best)),'dominant_prob':float(pdom[best])})
 with open(out/'token_domain_purity.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 by={}
 for r in rows:
  if r['depth']!='':by.setdefault(int(r['depth']),[]).append(r['specificity'])
 summary=[{'depth':d,'features':len(v),'mean_specificity':float(np.mean(v)),'median_specificity':float(np.median(v))} for d,v in sorted(by.items())]
 with open(out/'depth_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (out/'complete.json').write_text(json.dumps({'status':'complete','parts':len(parts),'tokens_by_domain':tokens.tolist(),'features':len(rows)},indent=2));print(json.dumps({'features':len(rows),'parts':len(parts)}))
if __name__=='__main__':main()
