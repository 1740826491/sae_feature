"""Paired online synthetic regression using real Gemma decoder directions."""
import argparse
import csv
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path('/mnt/workspace/Hrj/gemma_hierarchy')
LABELS = ROOT / 'results/dclm_1m_9b_residpost_top150_car_A/decoder_similarity_clusters'
SAE = ROOT / 'models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz'
BATCH = 384
DEVICE = 'cuda'


def tensor(x):
    return torch.as_tensor(x, device=DEVICE)


def basis(rng, n, r):
    q, t = np.linalg.qr(rng.standard_normal((n, r)).astype('float32'))
    return q * np.sign(np.diag(t))


class Data:
    def __init__(self, k, width, decoder):
        self.k = k
        ids, self.pools = [], []
        for name in ('B2', 'T20', 'T21'):
            with (LABELS / name / f'labels_k{k}.csv').open() as f:
                rows = list(csv.DictReader(f))
            for j in range(k):
                group = [int(x['feature']) for x in rows if int(x['cluster']) == j]
                assert group
                self.pools.append(np.arange(len(ids), len(ids) + len(group)))
                ids.extend(group)
        self.feature_ids = np.asarray(ids)
        raw = decoder[ids].astype('float32')
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        dim = raw.shape[1]
        rng = np.random.default_rng(2718 + k)
        self.latent = torch.empty((len(ids), 8), device=DEVICE)
        outputs = []
        for group in self.pools:
            self.latent[group[0]:group[-1]+1] = tensor(raw[group] @ basis(rng, dim, 8) * math.sqrt(dim))
            outputs.append(basis(rng, 64, 8).T)
        self.outputs = tensor(np.stack(outputs))
        if width < dim:
            # Shared prefix of one basis for both narrow models.
            proj = basis(np.random.default_rng(177), dim, dim//2)[:, :width]
            raw = raw @ proj * math.sqrt(dim/width)
        self.inputs = tensor(raw)

    def batch(self, rng, component=None):
        p = 2*self.k
        pairs = np.tile(np.arange(p*p), BATCH//(p*p))
        rng.shuffle(pairs)
        path, mixed = pairs//p, pairs%p
        groups = np.stack([path % self.k, self.k+path], axis=1)
        ids = np.empty_like(groups)
        for side in range(2):
            for g in np.unique(groups[:, side]):
                mask = groups[:, side] == g
                ids[mask, side] = rng.choice(self.pools[g], mask.sum())
        ids, groups = tensor(ids), tensor(groups)
        amp = tensor(rng.uniform(.5, 1.5, (BATCH, 2, 1)).astype('float32'))
        if component is not None:
            amp[:, 1-component] = 0
        x = (self.inputs[ids] * amp).sum(1)
        z = self.latent[ids] * amp
        y = ((z + .1*z.tanh()).unsqueeze(-2) @ self.outputs[groups]).squeeze(-2).sum(1)/math.sqrt(8)
        return x, y, groups, tensor(np.stack([mixed % self.k, self.k+mixed], axis=1)), tensor(path)


class Student(nn.Module):
    def __init__(self, width, k, mode):
        super().__init__()
        self.mode = mode
        n = 1 if mode.startswith('dense') else 3*k
        self.enc = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(n))
        self.dec = nn.ModuleList(nn.Linear(width, 64, bias=False) for _ in range(n))
        for layer in [*self.enc, *self.dec]:
            nn.init.xavier_uniform_(layer.weight)
        self.gate = nn.Linear(width, n) if mode == 'learned' else None

    def forward(self, x, groups, mixed, capture=False):
        if self.mode.startswith('dense'):
            h = self.enc[0](x)
            h = h + .1*h.tanh()
            return self.dec[0](h), h
        route = mixed if self.mode == 'mixed' else groups
        if self.gate is not None:
            logits = self.gate(x)
            route = logits.topk(2, dim=1).indices
            prob = logits.softmax(1)
        out = x.new_zeros(len(x), 64)
        blocks = []
        for i, (enc, dec) in enumerate(zip(self.enc, self.dec)):
            rows = (route == i).any(1).nonzero().flatten()
            if not rows.numel():
                if capture:
                    blocks.append(torch.zeros_like(x))
                continue
            # Bound GEMM shapes for the PPU runtime compiler.
            features = nn.functional.pad(x[rows], (0, 0, 0, (-rows.numel()) % 32))
            h = enc(features)
            h = h + .1*h.tanh()
            value = dec(h)[:rows.numel()]
            h = h[:rows.numel()]
            if self.gate is not None:
                w = prob[rows, i:i+1]
                value = value * (1 + 2*(w-w.detach()))
            out = out.index_add(0, rows, value)
            if capture:
                blocks.append(torch.zeros_like(x).index_copy(0, rows, h))
        return out, torch.cat(blocks, 1) if capture else None


@torch.no_grad()
def evaluate(model, data, seed, batches, capacity=False):
    rng = np.random.default_rng(seed)
    err = torch.zeros(2*data.k, device=DEVICE)
    power = torch.zeros_like(err)
    total = gram = None
    for _ in range(batches):
        x,y,g,m,p = data.batch(rng)
        pred,h = model(x,g,m,capacity)
        err.index_add_(0,p,(pred-y).square().sum(1))
        power.index_add_(0,p,y.square().sum(1))
        if capacity:
            if total is None:
                total = h.sum(0)
                gram = h.T @ h
            else:
                total += h.sum(0)
                gram += h.T @ h
    per = (err/power).tolist()
    result = dict(nmse=float(np.mean(per)), paths=per)
    if capacity:
        gram -= torch.outer(total,total)/(batches*BATCH)
        result['capacity'] = (gram.trace().square()/gram.square().sum()).item()
    return result


def run(cell, decoder, args):
    k,mode,seed = cell
    folder = args.out / f'k{k}_{mode}_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        return
    width = {'dense_full':3584,'dense_half':1792}.get(mode,(args.width_factor*3584)//(3*k))
    data = Data(k,width,decoder)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = Student(width,k,mode).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
    best, streak, hit, elapsed = float('inf'), 0, None, 0.
    with (folder/'curve.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            start = time.monotonic()
            x,y,g,m,_ = data.batch(rng)
            pred,_ = model(x,g,m)
            loss = (pred-y).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic()-start
            if step == 1 or step%100 == 0 or step == args.steps:
                metrics = evaluate(model,data,8101,args.eval_batches)
                if not math.isfinite(metrics['nmse']):
                    raise RuntimeError(f'nonfinite validation: {cell} step {step}')
                if step%100 == 0:
                    streak = streak+1 if metrics['nmse'] <= .10 else 0
                    if streak == 3 and hit is None:
                        hit = dict(step=step,samples=step*BATCH,seconds=elapsed,test=evaluate(model,data,8102,args.eval_batches))
                if metrics['nmse'] < best:
                    best = metrics['nmse']
                    torch.save(model.state_dict(),folder/'best.pt')
                print(json.dumps(dict(step=step,seconds=elapsed,**metrics)),file=log)
    model.load_state_dict(torch.load(folder/'best.pt',weights_only=True))
    test = evaluate(model,data,8102,args.eval_batches,True)
    result = dict(k=k,mode=mode,seed=seed,width=width,steps=args.steps,batch=BATCH,hit=hit,best_val=best,test=test,train_seconds=elapsed)
    tmp = folder/'result.tmp'
    tmp.write_text(json.dumps(result))
    tmp.replace(folder/'result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--eval-batches',type=int,default=128)
    parser.add_argument('--matched-dense',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--width-factor', type=int, choices=(1,2), default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    # Full cells are independent; evaluation and training use separate streams.
    modes = ('specified','learned','mixed','dense_half','dense_full')
    if args.width_factor == 2:
        modes = modes[:3]  # Reuse the existing paired Dense controls.
    if args.matched_dense:
        modes = ('dense_matched',)
    cells = [(k,m,s) for s,k,m in itertools.product((11,22,33),(2,4),modes)]
    if args.smoke:
        cells = [(k,m,11) for k in (2,4) for m in modes]
    with np.load(SAE) as z:
        decoder = z['W_dec']
    assert decoder.shape[1] == 3584
    for cell in cells[args.worker::args.workers]:
        run(cell,decoder,args)
