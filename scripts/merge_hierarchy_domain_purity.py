#!/usr/bin/env python3
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--hierarchy',type=Path,required=True); p.add_argument('--purity',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    depths={}
    with a.hierarchy.open() as f:
        for r in csv.DictReader(f): depths[int(r['feature'])]=int(r['depth'])
    rows=[]
    with a.purity.open() as f:
        for r in csv.DictReader(f):
            fid=int(r.get('feature',r.get('feature_id'))); depth=depths.get(fid)
            if depth is None: continue
            purity=float(r.get('specificity',r.get('domain_specificity',r.get('purity',0.0))))
            rows.append((fid,depth,purity))
    with (a.out/'feature_depth_purity.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['feature','depth','domain_purity']); w.writerows(rows)
    groups=defaultdict(list)
    for _,d,s in rows: groups[d].append(s)
    summary=[]
    for d in sorted(groups):
        vals=groups[d]; summary.append({'depth':d,'n_features':len(vals),'mean_purity':sum(vals)/len(vals),'min_purity':min(vals),'max_purity':max(vals)})
    (a.out/'depth_purity_summary.json').write_text(json.dumps({'rows':len(rows),'summary':summary},indent=2))
    print(json.dumps({'rows':len(rows),'summary':summary}))
if __name__=='__main__': main()
