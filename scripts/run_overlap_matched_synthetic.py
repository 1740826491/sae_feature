#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--overlap',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader((a.overlap/'overlap_summary.csv').open()));out=[]
 for r in rows:
  ov=float(r['mean_overlap']); k=int(r['experts']);
  for seed in range(3):
   rng=np.random.default_rng(seed); n=20000; d=64; common=max(1,int(ov*d)); A=rng.standard_normal((n,d)); B=rng.standard_normal((n,d)); B[:,:common]=A[:,:common];
   mse=float(np.mean((A-B)**2)); out.append({'depth':int(r['depth']),'experts':k,'target_overlap':ov,'seed':seed,'synthetic_mse':mse})
 with (a.out/'synthetic_overlap_capacity.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=out[0].keys() if out else ['depth']);w.writeheader();w.writerows(out)
 (a.out/'complete.json').write_text(json.dumps({'rows':len(out),'method':'synthetic overlap matched by shared coordinates'},indent=2))
if __name__=='__main__': main()
