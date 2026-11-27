#!/usr/bin/env python3
import argparse,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();inp=Path(a.input)/'token_activations';out=Path(a.output);out.mkdir(parents=True,exist_ok=True);freq=np.zeros(131072,dtype=np.int64);pairs=defaultdict(int)
 files=sorted(inp.glob('*.npz'))
 for p in files:
  z=np.load(p);ids=z['feature_ids'];vals=z['activation_values']
  for row in range(ids.shape[0]):
   active=np.unique(ids[row][vals[row]>0]);active=active[active>=0];freq[active]+=1
   for i in range(len(active)):
    for j in range(i+1,len(active)):pairs[(int(active[i]),int(active[j]))]+=1
 np.save(out/'frequency_rank0.npy',freq)
 with (out/'coactivation_rank0.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['a','b','coactivation']);w.writerows((i,j,c) for (i,j),c in pairs.items())
 print({'parts':len(files),'features':int((freq>0).sum()),'pairs':len(pairs)})
if __name__=='__main__':main()
