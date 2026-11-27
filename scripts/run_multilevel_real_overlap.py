#!/usr/bin/env python3
"""Three-level MoE experiment parameterized by measured held-out overlaps.

Representation dimension is always the SAE decoder width (3584).  ``width`` is
only the processing/bottleneck dimension of a dense block or expert.  Each
sample chooses one of K experts independently at each of three levels and is a
sum of features from all three selected pools.
"""
from __future__ import annotations

import argparse, hashlib, json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn


OVERLAPS = {
    "ideal": (0.0, 0.0, 0.0),
    "real": (0.4057347404, 0.2298932808, 0.1508672708),
    "random_assignment": (0.827406, 0.616472, 0.409054),
}
METHODS = ("dense", "dense_matched", "tree_oracle", "tree_random",
           "tree_learned", "flat_top3")


def stable_seed(*x):
    return int.from_bytes(hashlib.sha256("|".join(map(str, x)).encode()).digest()[:4], "little")


@dataclass(frozen=True)
class Cell:
    overlap: str
    width: int
    method: str
    seed: int


class Data:
    def __init__(self, decoder, overlaps, experts, pool, active, seed, device):
        self.K, self.L, self.pool, self.active = experts, 3, pool, active
        rng = np.random.default_rng(seed)
        pools, ids_by_level, realized = [], [], []
        cursor = 0
        # Each level has a shared core plus disjoint expert-private features:
        # J=C/(2P-C), so C=2qP/(1+q).
        for q in overlaps:
            common = int(round(2*q*pool/(1+q))); private = pool-common
            total = common + experts*private
            ids = rng.choice(len(decoder), total, replace=False)
            ids_by_level.append(ids)
            level_pools=[]
            for e in range(experts):
                level_pools.append(np.r_[np.arange(cursor,cursor+common),
                                         np.arange(cursor+common+e*private,cursor+common+(e+1)*private)])
            pools.append(level_pools); cursor += total
            realized.append(common/max(1,2*pool-common))
        all_ids=np.concatenate(ids_by_level)
        U=decoder[all_ids].astype(np.float32,copy=True)
        U/=np.maximum(np.linalg.norm(U,axis=1,keepdims=True),1e-8)
        # One canonical target per feature, irrespective of how many pools own it.
        trng=np.random.default_rng(271828)
        T=trng.standard_normal((len(U),64)).astype(np.float32)/8
        self.U=torch.as_tensor(U,device=device); self.T=torch.as_tensor(T,device=device)
        self.pools=pools; self.realized=realized; self.device=device

    def batch(self, seed, step, batch):
        rng=np.random.default_rng(stable_seed(seed,step,"batch"))
        routes=rng.integers(self.K,size=(batch,self.L),dtype=np.int64)
        chosen=np.empty((batch,self.L,self.active),np.int64)
        for l in range(self.L):
            for e in range(self.K):
                ix=np.flatnonzero(routes[:,l]==e)
                if len(ix): chosen[ix,l]=rng.choice(self.pools[l][e],(len(ix),self.active),replace=True)
        amp=rng.uniform(.5,1.5,(batch,self.L,self.active,1)).astype(np.float32)
        ix=torch.as_tensor(chosen,device=self.device); a=torch.as_tensor(amp,device=self.device)
        x=(self.U[ix]*a).sum((1,2))/math.sqrt(self.L*self.active)
        y=(self.T[ix]*a).sum((1,2))/math.sqrt(self.L*self.active)
        return x,y,torch.as_tensor(routes,device=self.device)


class Block(nn.Module):
    def __init__(self,din,width,dout=64):
        super().__init__(); self.a=nn.Linear(din,width,bias=False);self.b=nn.Linear(width,dout,bias=False)
        nn.init.normal_(self.a.weight,std=.02);nn.init.zeros_(self.b.weight)
    def forward(self,x): return self.b(self.a(x))


class Model(nn.Module):
    def __init__(self,din,width,K,method):
        super().__init__();self.method=method;self.K=K
        if method.startswith("dense"):
            self.blocks=nn.ModuleList([Block(din,width*(3 if method=="dense_matched" else 1))])
        elif method=="flat_top3":
            self.blocks=nn.ModuleList(Block(din,width) for _ in range(3*K));self.gate=nn.Linear(din,3*K)
        else:
            self.blocks=nn.ModuleList(Block(din,width) for _ in range(3*K))
            if method=="tree_learned": self.gates=nn.ModuleList(nn.Linear(din,K) for _ in range(3))

    def forward(self,x,routes,random_routes=None):
        if self.method.startswith("dense"): return self.blocks[0](x),x.new_zeros(())
        if self.method=="flat_top3":
            p=self.gate(x).softmax(-1); top=p.topk(3,-1); out=x.new_zeros((len(x),64))
            for e,b in enumerate(self.blocks):
                hit=(top.indices==e); rows=hit.any(1).nonzero().flatten()
                if len(rows): out.index_add_(0,rows,b(x[rows])*(top.values*hit).sum(1)[rows,None])
            mean=p.mean(0);aux=(mean*mean).sum()*len(self.blocks)
            return out,aux
        if self.method=="tree_learned":
            out=x.new_zeros((len(x),64));aux=x.new_zeros(())
            for l,g in enumerate(self.gates):
                p=g(x).softmax(-1);aux=aux+(p.mean(0).pow(2).sum()*self.K)
                # Soft mixture keeps task gradients to all gates and is deterministic.
                for e in range(self.K): out=out+self.blocks[l*self.K+e](x)*p[:,e,None]
            return out,aux/3
        rr=routes if self.method=="tree_oracle" else random_routes
        out=x.new_zeros((len(x),64))
        for l in range(3):
            for e in range(self.K):
                rows=(rr[:,l]==e).nonzero().flatten()
                if len(rows): out.index_add_(0,rows,self.blocks[l*self.K+e](x[rows]))
        return out,x.new_zeros(())


def eval_nmse(model,data,seed,batches,batch):
    model.eval();se=sp=0.;acc=[]
    with torch.no_grad():
        for j in range(batches):
            x,y,r=data.batch(seed,10_000_000+j,batch)
            gen=torch.Generator(device=x.device).manual_seed(seed+j)
            rr=torch.randint(data.K,r.shape,device=x.device,generator=gen)
            z,_=model(x,r,rr);se+=float((z-y).pow(2).sum());sp+=float(y.pow(2).sum())
            if model.method=="tree_learned": acc.append(float(torch.stack([(g(x).argmax(-1)==r[:,l]).float().mean() for l,g in enumerate(model.gates)]).mean()))
    model.train();return se/max(sp,1e-12),(float(np.mean(acc)) if acc else None)


def run(cell,args,decoder,device):
    folder=args.out/cell.overlap/f"d{cell.width}_{cell.method}_s{cell.seed}";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"result.json").exists():return
    torch.manual_seed(cell.seed)
    data=Data(decoder,OVERLAPS[cell.overlap],args.experts,args.pool_size,args.active_features,args.data_seed,device)
    model=Model(3584,cell.width,args.experts,cell.method).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0)
    hits={str(t):None for t in args.thresholds};curve=[];t0=time.time();gate_grad=None
    for step in range(1,args.steps+1):
        x,y,r=data.batch(cell.seed,step,args.batch_size)
        gen=torch.Generator(device=device).manual_seed(stable_seed(cell.seed,step,"route"))
        rr=torch.randint(args.experts,r.shape,device=device,generator=gen)
        z,aux=model(x,r,rr);loss=(z-y).pow(2).mean()+args.aux_coef*aux
        opt.zero_grad(set_to_none=True);loss.backward()
        if step==1 and cell.method in ("tree_learned","flat_top3"):
            ps=list(model.gates.parameters()) if cell.method=="tree_learned" else list(model.gate.parameters())
            gate_grad=float(sum((p.grad.pow(2).sum() for p in ps if p.grad is not None),torch.zeros((),device=device)).sqrt())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            nmse,acc=eval_nmse(model,data,cell.seed+9000,args.eval_batches,args.batch_size)
            row={"step":step,"samples":step*args.batch_size,"nmse":nmse,"route_accuracy":acc};curve.append(row)
            for t in args.thresholds:
                if hits[str(t)] is None and nmse<=t:hits[str(t)]=row
    test,acc=eval_nmse(model,data,cell.seed+19000,args.test_batches,args.batch_size)
    total=sum(p.numel() for p in model.parameters());active=total if cell.method.startswith("dense") else sum(p.numel() for p in model.blocks[0].parameters())*3
    result={**asdict(cell),"target_overlaps":OVERLAPS[cell.overlap],"realized_overlaps":data.realized,
            "representation_dim":3584,"processing_dim":cell.width,"experts_per_level":args.experts,
            "levels":3,"hits":hits,"curve":curve,"test_nmse":test,"test_route_accuracy":acc,
            "gate_grad_step1":gate_grad,"total_params":total,"active_params":active,"seconds":time.time()-t0}
    (folder/"result.json").write_text(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--sae",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--overlaps",nargs="+",default=list(OVERLAPS));p.add_argument("--widths",nargs="+",type=int,default=[64,128,256,358])
    p.add_argument("--methods",nargs="+",default=list(METHODS));p.add_argument("--seeds",nargs="+",type=int,default=[11,22,33])
    p.add_argument("--experts",type=int,default=8);p.add_argument("--pool-size",type=int,default=128);p.add_argument("--active-features",type=int,default=8)
    p.add_argument("--steps",type=int,default=3000);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--eval-batches",type=int,default=8);p.add_argument("--test-batches",type=int,default=32);p.add_argument("--thresholds",nargs="+",type=float,default=[.2,.1])
    p.add_argument("--lr",type=float,default=1e-3);p.add_argument("--aux-coef",type=float,default=.01);p.add_argument("--data-seed",type=int,default=20261127)
    p.add_argument("--worker",type=int,default=0);p.add_argument("--workers",type=int,default=1);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if any(x not in OVERLAPS for x in a.overlaps) or any(x not in METHODS for x in a.methods):raise ValueError("unknown overlap/method")
    cells=[Cell(o,d,m,s) for o in a.overlaps for d in a.widths for m in a.methods for s in a.seeds]
    manifest={"status":"running","contract":{"question":"learning efficiency under three simultaneously measured overlaps","representation_dim":3584,"processing_dims":a.widths,"overlaps":OVERLAPS,"methods":a.methods},"cells":len(cells),"config":vars(a)}
    (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str))
    with np.load(a.sae) as z: decoder=z["W_dec"]
    if decoder.shape[1]!=3584:raise ValueError(decoder.shape)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i,c in enumerate(cells):
        if i%a.workers==a.worker:run(c,a,decoder,device)
    (a.out/f"worker_{a.worker}_complete.json").write_text(json.dumps({"status":"complete"}))

if __name__=="__main__":main()
