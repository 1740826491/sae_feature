"""K=2 linear teachers; fixed input views and low-rank student updates."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from run_moe_sweep import Data, SAE, basis, tensor

P, D, B = 597, 3584, 384


class Task:
    def __init__(self, decoder):
        self.data = Data(2, D, decoder)
        rng = np.random.default_rng(177)
        projection = basis(rng,D,D//2)[:,:P]*math.sqrt(D/P)
        self.small = self.data.inputs @ tensor(projection)
        rng = np.random.default_rng(2718)
        self.W = tensor(basis(rng,P,P))
        self.base = self.small @ self.W
        self.delta = torch.empty_like(self.base)
        for group in self.data.pools:
            lo,hi = group[0],group[-1]+1
            u,v = tensor(basis(rng,P,8)),tensor(basis(rng,P,8))
            diff = (self.small[lo:hi] @ u) @ v.T
            self.delta[lo:hi] = diff * (self.base[lo:hi].norm()/diff.norm())
        self.ref_error = float((self.base-self.small@self.W).abs().max())

    def batch(self,rng,dim,lam,side=None,export=None):
        codes=np.tile(np.arange(16),24)
        rng.shuffle(codes)
        path,alt=codes//4,codes%4
        route=np.stack((path%2,2+path),1)
        ids=np.empty_like(route)
        for i in range(2):
            for g in np.unique(route[:,i]):
                mask=route[:,i]==g
                ids[mask,i]=rng.choice(self.data.pools[g],mask.sum())
        ids=tensor(ids)
        amp=tensor(rng.uniform(.5,1.5,(B,2,1)).astype('float32'))
        if side is not None:
            amp[:,1-side]=0
        x=((self.small if dim==P else self.data.inputs)[ids]*amp).sum(1)
        shared=(self.base[ids]*amp).sum(1)
        delta=(self.delta[ids]*amp).sum(1)*lam
        if export is not None:
            parts=(self.base[ids]+lam*self.delta[ids])*amp
            np.savez_compressed(export, feature_ids=self.data.feature_ids[ids.cpu().numpy()],
                                expert_ids=route, strength=amp.squeeze(-1).cpu().numpy(),
                                x=x.cpu().numpy(), y=(shared+delta).cpu().numpy(),
                                y_common=parts[:,0].cpu().numpy(), y_leaf=parts[:,1].cpu().numpy())
        return x,shared+delta,delta,tensor(route),tensor(np.stack((alt%2,2+alt),1))


class Model(nn.Module):
    def __init__(self,d,r,mode,out_dim=P):
        super().__init__()
        self.mode=mode
        self.out_dim=out_dim
        self.maps=nn.ModuleList()
        for _ in range(1 if mode=='dense' else 6):
            if r:
                layer=nn.Sequential(nn.Linear(d,r,bias=False),nn.Linear(r,out_dim,bias=False))
                nn.init.zeros_(layer[1].weight)
            else:
                layer=nn.Linear(d,out_dim,bias=False)
            self.maps.append(layer)
        self.gate=nn.Linear(d,6) if mode=='learned' else None

    def forward(self,x,g,m):
        if self.mode=='dense':
            return self.maps[0](x)
        route=m if self.mode=='mixed' else g
        if self.gate is not None:
            logits=self.gate(x)
            route=logits.topk(2,1).indices
            prob=logits.softmax(1)
        out=x.new_zeros(len(x),self.out_dim)
        for i,layer in enumerate(self.maps):
            ix=(route==i).any(1).nonzero().flatten()
            if ix.numel():
                z=nn.functional.pad(x[ix],(0,0,0,(-ix.numel())%32))
                y=layer(z)[:ix.numel()]
                if self.gate is not None:
                    w=prob[ix,i:i+1]
                    y=y*(1+2*(w-w.detach()))
                out=out.index_add(0,ix,y)
        return out


@torch.no_grad()
def evaluate(task,model,d,lam,seed,n):
    scores={}
    for side,name in ((None,'mix'),(0,'common'),(1,'leaf')):
        rng=np.random.default_rng(seed)
        sums=torch.zeros(3,device='cuda')
        for _ in range(n):
            x,y,delta,g,m=task.batch(rng,d,lam,side)
            err=(model(x,g,m)-y).square().sum()
            sums+=torch.stack((err,y.square().sum(),delta.square().sum()))
        e,p,q=sums.tolist()
        scores[name]=e/p
        if lam:
            scores[name+'_vs_shared_only']=e/q
    return scores


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--square-output',action='store_true')
    ap.add_argument('--capacity-curve',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--optimizer',choices=('adamw','sgd'),default='adamw')
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    with np.load(SAE) as z:
        task=Task(z['W_dec'])
    assert task.ref_error < 1e-5
    modes=('dense','specified','learned','mixed')
    cells=[(0.,d,0,'dense',11) for d in (P,D)]
    cells += [(.25,d,0,'dense',seed) for seed in (11,22,33) for d in (P,D)]
    cells += [(.25,d,r,mode,seed) for seed in (11,22,33) for r in (300,400,500) for d in (P,D) for mode in modes]
    if a.capacity_curve:
        cells=[(.25,D,r,m,seed) for seed in (11,22,33)
               for r in (100,200,597,0) for m in ('specified','mixed','dense')
               if r or m!='dense']  # Existing full Dense is reused.
    if a.smoke:
        cells=[(.25,D,300,m,11) for m in modes]+[(0.,P,0,'dense',11)]
    if a.square_output:
        lift=tensor(basis(np.random.default_rng(3584),D,P).T)
        task.base=task.base@lift
        task.delta=task.delta@lift
        cells=[(.25,D,0 if m=='dense' else 300,m,seed)
               for seed in ((11,) if a.smoke else (11,22,33)) for m in modes]
        if a.worker==0:
            a.out.mkdir(parents=True,exist_ok=True)
            task.batch(np.random.default_rng(8102),D,.25,export=a.out/'analysis_sample.npz')
    for lam,d,r,mode,seed in cells[a.worker::a.workers]:
        folder=a.out/f'l{lam}_d{d}_r{r}_{mode}_{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'result.json').exists():continue
        rng=np.random.default_rng(seed)
        torch.manual_seed(seed)
        model=Model(d,r,mode,D if a.square_output else P).cuda()
        if a.optimizer == 'sgd':
            opt=torch.optim.SGD(model.parameters(),lr=a.lr)
        else:
            opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0)
        steps=2 if a.smoke else 6000
        with (folder/'curve.jsonl').open('w',buffering=1) as log:
            for step in range(1,steps+1):
                x,y,_,g,m=task.batch(rng,d,lam)
                opt.zero_grad(set_to_none=True)
                loss=(model(x,g,m)-y).square().mean()
                loss.backward()
                opt.step()
                if step==1 or step%100==0 or step==steps:
                    scores=evaluate(task,model,d,lam,8101,1 if a.smoke else 8)
                    assert all(map(math.isfinite,scores.values()))
                    print(json.dumps(dict(step=step,**scores)),file=log)
        scores=evaluate(task,model,d,lam,8102,1 if a.smoke else 128)
        torch.save(model.state_dict(),folder/'final.pt')
        tmp=folder/'result.tmp'
        tmp.write_text(json.dumps(dict(step=steps,params=sum(p.numel() for p in model.parameters()),optimizer=a.optimizer,lr=a.lr,**scores)))
        tmp.replace(folder/'result.json')


if __name__=='__main__':main()
