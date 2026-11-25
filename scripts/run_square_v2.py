"""Square v2: D->internal->D teacher, D->rank->D students.

Runs Dense, oracle/specified, learned top-2, and deliberately mixed routing
for the available K=2 and K=4 cluster labels.  This is intentionally separate
from the legacy run_moe_sweep.py experiment.
"""
import argparse, csv, itertools, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

D, B, DEVICE = 3584, 384, 'cuda'
ROOT = Path('/mnt/workspace/Hrj/gemma_hierarchy')
LABELS = ROOT/'results/dclm_1m_9b_residpost_top150_car_A/decoder_similarity_clusters'
SAE = ROOT/'models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz'

def t(x): return torch.as_tensor(x, device=DEVICE)
def orth(rng, n, r):
    q, z = np.linalg.qr(rng.standard_normal((n,r)).astype('float32'))
    return q*np.sign(np.diag(z))

class Task:
    def __init__(self, k, decoder, seed=2718):
        self.k=k; self.n=3*k; self.pools=[]; ids=[]
        for name in ('B2','T20','T21'):
            rows=list(csv.DictReader((LABELS/name/f'labels_k{k}.csv').open()))
            for c in range(k):
                f=[int(r['feature']) for r in rows if int(r['cluster'])==c]
                self.pools.append(np.arange(len(ids),len(ids)+len(f))); ids.extend(f)
        raw=decoder[ids].astype('float32'); raw/=np.linalg.norm(raw,axis=1,keepdims=True)
        self.xdec=t(raw); rng=np.random.default_rng(seed+k)
        # Each expert has a D->d->D teacher map with full-rank d-dimensional core.
        self.dw=max(1,D//(3*k)); self.teacher=[]
        for _ in range(self.n):
            E=t(orth(rng,D,self.dw))
            F=t(orth(rng,D,self.dw))
            core=t(orth(rng,self.dw,self.dw))
            self.teacher.append((E,core,F))
    def batch(self,rng):
        p=2*self.k; pairs=np.tile(np.arange(p*p),B//(p*p)); rng.shuffle(pairs)
        path,mix=pairs//p,pairs%p
        g=np.stack((path%self.k,self.k+path),1); ids=np.empty_like(g)
        for side in range(2):
            for j in np.unique(g[:,side]):
                m=g[:,side]==j; ids[m,side]=rng.choice(self.pools[j],m.sum())
        ids=t(ids); amp=t(rng.uniform(.5,1.5,(B,2,1)).astype('float32'))
        x=(self.xdec[ids]*amp).sum(1)
        # Factorized teacher application; never materialize B x D x D.
        y=torch.zeros_like(x)
        for side in range(2):
            for j in np.unique(g[:,side]):
                rows=t(np.flatnonzero(g[:,side]==j)).long()
                E,core,F=self.teacher[j]
                z=self.xdec[ids[rows,side]] @ F
                z=z @ core.T @ E.T
                y.index_add_(0,rows,z*amp[rows,side])
        # mixed carries an intentionally permuted expert pair with equal load.
        bad=np.stack((mix%self.k,self.k+mix),1)
        return x,y,t(g),t(bad)

class Model(nn.Module):
    def __init__(self,k,mode,rank):
        super().__init__(); self.mode=mode; n=1 if mode=='dense' else 3*k
        self.e=nn.ModuleList(nn.Linear(D,D,bias=False) if mode=='dense' else nn.Sequential(nn.Linear(D,rank,bias=False),nn.Linear(rank,D,bias=False)) for _ in range(n))
        self.gate=nn.Linear(D,n) if mode=='learned' else None
    def forward(self,x,g,bad):
        if self.mode=='dense': return self.e[0](x)
        route=bad if self.mode=='mixed' else g
        if self.gate is not None: route=self.gate(x).topk(2,1).indices
        out=torch.zeros_like(x)
        for j,layer in enumerate(self.e):
            rows=(route==j).any(1).nonzero().flatten()
            if rows.numel(): out.index_add_(0,rows,layer(x[rows]))
        return out

@torch.no_grad()
def eval_model(m,task,rng,nb=16):
    e=p=0.
    for _ in range(nb):
        x,y,g,b=task.batch(rng); q=m(x,g,b); e+=(q-y).square().sum().item(); p+=y.square().sum().item()
    return e/p

def run(args,k,mode,seed):
    out=args.out/f'{args.optimizer}_k{k}_{mode}_{seed}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): return
    task=Task(k,args.decoder); rank= D if mode=='dense' else max(1,D//(3*k))
    torch.manual_seed(seed); m=Model(k,mode,rank).cuda()
    if args.optimizer == 'sgd':
        opt=torch.optim.SGD(m.parameters(),lr=args.lr)
    else:
        opt=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=0)
    rng=np.random.default_rng(seed); start=time.time(); curve=out/'curve.jsonl'
    with curve.open('w') as f:
        for step in range(1,args.steps+1):
            x,y,g,b=task.batch(rng); loss=(m(x,g,b)-y).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
            if step==1 or step%args.log_every==0 or step==args.steps:
                v=eval_model(m,task,np.random.default_rng(8101),args.eval_batches); f.write(json.dumps({'step':step,'nmse':v})+'\n'); f.flush()
    res={'optimizer':args.optimizer,'k':k,'mode':mode,'seed':seed,'teacher_internal':task.dw,'student_rank':rank,'steps':args.steps,'train_seconds':time.time()-start,'test_nmse':eval_model(m,task,np.random.default_rng(8102),args.eval_batches),'params':sum(p.numel() for p in m.parameters())}
    (out/'result.json').write_text(json.dumps(res,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--steps',type=int,default=6000); ap.add_argument('--eval-batches',type=int,default=16); ap.add_argument('--log-every',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--optimizer',choices=('adamw','sgd'),default='adamw'); ap.add_argument('--worker',type=int,default=0); ap.add_argument('--workers',type=int,default=1)
    a=ap.parse_args(); torch.set_num_threads(1)
    with np.load(SAE) as z: a.decoder=z['W_dec']
    cells=list(itertools.product((2,4),('dense','specified','learned','mixed'),(11,22,33)))
    for i,(k,m,s) in enumerate(cells):
        if i%a.workers==a.worker: run(a,k,m,s)
