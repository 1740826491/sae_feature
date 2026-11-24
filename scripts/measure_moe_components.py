"""Probe one-component targets on held-out draws with the saved best model."""
import json
from pathlib import Path
import numpy as np
import torch
from run_moe_sweep import Data, Student, SAE

torch.set_num_threads(1)
with np.load(SAE) as z:
    decoder=z['W_dec']
cache={}
for root in ('formal','wide2','matched1','matched2'):
    for f in sorted(Path(root).glob('*/result.json')):
        out=f.parent/'components.json'
        if out.exists():
            continue
        r=json.loads(f.read_text())
        k,w,mode=r['k'],r['width'],r['mode']
        if (k,w) not in cache:
            cache.clear()
            cache[k,w]=Data(k,w,decoder)
        data=cache[k,w]
        model=Student(w,k,mode).cuda().eval()
        model.load_state_dict(torch.load(f.parent/'best.pt',weights_only=True))
        scores={}
        with torch.no_grad():
            for side,name in enumerate(('common','leaf')):
                rng=np.random.default_rng(8102)
                err=torch.zeros(2*k,device='cuda')
                power=torch.zeros_like(err)
                for _ in range(128):
                    x,y,g,m,p=data.batch(rng,component=side)
                    pred,_=model(x,g,m)
                    err.index_add_(0,p,(pred-y).square().sum(1))
                    power.index_add_(0,p,y.square().sum(1))
                scores[name]=(err/power).tolist()
        tmp=out.with_suffix('.tmp')
        tmp.write_text(json.dumps(scores))
        tmp.replace(out)
