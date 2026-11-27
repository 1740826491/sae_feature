#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
rows=[json.loads(x.read_text()) for x in a.input.glob('*/w*_*/result.json')]
if not rows: raise SystemExit('no result.json files')
a.out.mkdir(parents=True,exist_ok=True)
fields=sorted({k for r in rows for k in r if k!='hits'})+['hit_0.2_steps','hit_0.2_samples','hit_0.1_steps','hit_0.1_samples']
flat=[]
for r in rows:
    z={k:v for k,v in r.items() if k!='hits'}
    for t in ('0.2','0.1'):
        hit=r['hits'].get(t); z[f'hit_{t}_steps']=hit['step'] if hit else ''; z[f'hit_{t}_samples']=hit['samples'] if hit else ''
    flat.append(z)
with (a.out/'all_cells.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
groups={}
for r in flat:
    key=(r['overlap_name'],r['target_overlap'],r['width'],r['mode'])
    groups.setdefault(key,[]).append(r)
summary=[]
for (name,q,width,mode),items in groups.items():
    row={'overlap_name':name,'target_overlap':q,'width':width,'mode':mode,'seeds':len(items)}
    for metric in ('best_val_nmse','test_nmse','total_params','active_params','hit_0.2_steps','hit_0.1_steps'):
        vals=[float(x[metric]) for x in items if x.get(metric,'')!='']
        row[metric+'_mean']=float(np.mean(vals)) if vals else ''
        row[metric+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else ''
        if metric.startswith('hit_'): row[metric+'_success_fraction']=len(vals)/len(items)
    summary.append(row)
with (a.out/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(a.out/'complete.json').write_text(json.dumps({'cells':len(rows),'summary_rows':len(summary)},indent=2))
