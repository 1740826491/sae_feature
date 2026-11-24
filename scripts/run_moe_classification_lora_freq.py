"""Classification MoE sweep with independent LoRA experts and frequency strata.

Inputs stay 3584-dimensional.  Every B2/T20/T21 cluster has its own fixed
teacher mapping.  Feature draws are weighted by observed activation frequency;
the evaluator reports common-only, long-tail-only, and mixed samples.
"""

from __future__ import annotations

import argparse, csv, itertools, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path('/mnt/workspace/Hrj/gemma_hierarchy')
LABELS = ROOT / 'results/dclm_1m_9b_residpost_top150_car_A/decoder_similarity_clusters'
FREQ = ROOT / 'results/dclm_1m_9b_residpost_top150_car_A/relation_nested_classes'
SAE = ROOT / 'models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz'
INPUT_DIM, BATCH, DEVICE = 3584, 384, 'cuda'


def t(x): return torch.as_tensor(x, device=DEVICE)


def read_freq():
    out = {}
    with (FREQ / 'bottom_cluster_labels.csv').open() as f:
        for r in csv.DictReader(f): out[int(r['feature'])] = float(r['activation_docs'])
    with (FREQ / 'top_cluster_labels.csv').open() as f:
        for r in csv.DictReader(f): out[int(r['feature'])] = float(r['activation_docs'])
    return out


class Data:
    def __init__(self, k, decoder):
        self.k, self.classes = k, 2 * k
        freq = read_freq()
        pools, ids, pool_freq = [], [], []
        for name in ('B2', 'T20', 'T21'):
            with (LABELS / name / f'labels_k{k}.csv').open() as f: rows = list(csv.DictReader(f))
            for j in range(k):
                group = [int(r['feature']) for r in rows if int(r['cluster']) == j]
                start = len(ids); ids.extend(group); pools.append(np.arange(start, len(ids)))
                pool_freq.append(np.asarray([max(freq[x], 1.0) for x in group], dtype='float64'))
        self.pools = pools
        self.pool_freq = [p / p.sum() for p in pool_freq]
        all_freq = np.concatenate([np.asarray([freq[x] for x in ids[i]], dtype='float64') for i in []]) if False else np.asarray([freq[x] for x in ids])
        q25, q75 = np.quantile(all_freq, [.25, .75])
        self.longtail = np.asarray(all_freq <= q25)
        self.common = np.asarray(all_freq >= q75)
        raw = decoder[ids].astype('float32'); raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        self.inputs = t(raw)
        rng = np.random.default_rng(9000 + k)
        self.latent = t(rng.standard_normal((len(ids), 8)).astype('float32'))
        # Distinct fixed teacher map for every B2/T20/T21 cluster.
        self.teacher = t((rng.standard_normal((3 * k, 8, self.classes)).astype('float32') / math.sqrt(8)))

    def batch(self, rng):
        p = 2 * self.k
        pair = rng.integers(0, p * p, size=BATCH)
        path, mixed = pair // p, pair % p
        groups = np.stack([path % self.k, self.k + path], 1)
        mixed_groups = np.stack([mixed % self.k, self.k + mixed], 1)
        ids = np.empty_like(groups)
        for side in range(2):
            for g in np.unique(groups[:, side]):
                mask = groups[:, side] == g
                ids[mask, side] = rng.choice(self.pools[g], mask.sum(), p=self.pool_freq[g])
        ids_t = t(ids); amp = t(rng.uniform(.5, 1.5, (BATCH, 2, 1)).astype('float32'))
        x = (self.inputs[ids_t] * amp).sum(1)
        z = self.latent[ids_t] * amp; phi = z + .1 * z.tanh()
        logits = torch.einsum('bsi,bsic->bsc', phi, self.teacher[t(groups)]).sum(1) / math.sqrt(8)
        target = logits.argmax(1)
        lt = self.longtail[ids].any(1); common = self.common[ids].all(1)
        bucket = np.where(lt, 1, np.where(common, 0, 2))  # 0 common-only, 1 long-tail, 2 mixed/other
        return x, target, t(groups), t(mixed_groups), t(path), t(bucket)


class IndependentLoRAExperts(nn.Module):
    def __init__(self, inp, hidden, out, n, rank, alpha):
        super().__init__(); self.n = n; self.scale = alpha / rank
        # Independent expert base references and independent LoRA parameters.
        self.base_enc = nn.Parameter(torch.zeros(n, hidden, inp), requires_grad=False)
        self.base_dec = nn.Parameter(torch.zeros(n, out, hidden), requires_grad=False)
        self.enc_a = nn.Parameter(torch.randn(n, rank, inp) * .01)
        self.enc_b = nn.Parameter(torch.randn(n, hidden, rank) * .01)
        self.dec_a = nn.Parameter(torch.randn(n, rank, hidden) * .01)
        self.dec_b = nn.Parameter(torch.randn(n, out, rank) * .01)

    def forward_one(self, x, i):
        h = F.linear(x, self.base_enc[i]) + F.linear(F.linear(x, self.enc_a[i]), self.enc_b[i]) * self.scale
        h = h + .1 * h.tanh()
        y = F.linear(h, self.base_dec[i]) + F.linear(F.linear(h, self.dec_a[i]), self.dec_b[i]) * self.scale
        return y, h


class Student(nn.Module):
    def __init__(self, k, mode, hidden, rank):
        super().__init__(); self.mode, self.classes, self.n = mode, 2 * k, 3 * k
        if mode.startswith('dense'):
            self.enc, self.dec = nn.Linear(INPUT_DIM, hidden, bias=False), nn.Linear(hidden, self.classes, bias=False)
            nn.init.xavier_uniform_(self.enc.weight); nn.init.xavier_uniform_(self.dec.weight); self.moe = None; self.gate = None
        else:
            self.moe = IndependentLoRAExperts(INPUT_DIM, hidden, self.classes, self.n, rank, 2 * rank)
            self.gate = nn.Linear(INPUT_DIM, self.n) if mode == 'learned' else None

    def forward(self, x, groups, mixed, capture=False):
        if self.moe is None:
            h = self.enc(x); h = h + .1 * h.tanh(); return self.dec(h), h
        route = mixed if self.mode == 'mixed' else groups
        if self.gate is not None:
            gl = self.gate(x); route = gl.topk(2, 1).indices; prob = gl.softmax(1)
        if self.mode == 'random':
            route = torch.rand(len(x), self.n, device=x.device).topk(2, 1).indices
        out = x.new_zeros(len(x), self.classes); hdim = self.moe.enc_b.shape[1]; blocks = torch.zeros(len(x), self.n * hdim, device=x.device, dtype=x.dtype) if capture else None
        for i in range(self.n):
            rows = (route == i).any(1).nonzero().flatten()
            if not rows.numel(): continue
            y, h = self.moe.forward_one(x[rows], i)
            if self.gate is not None: y = y * (1 + 2 * (prob[rows, i:i+1] - prob[rows, i:i+1].detach()))
            out.index_add_(0, rows, y)
            if capture:
                blocks[rows, i * hdim:(i + 1) * hdim] = h
        return out, blocks


@torch.no_grad()
def evaluate(model, data, seed, batches, capacity=False):
    rng = np.random.default_rng(seed); loss = torch.zeros(3, device=DEVICE); correct = torch.zeros(3, device=DEVICE); count = torch.zeros(3, device=DEVICE); sh = gram = None
    for _ in range(batches):
        x, y, g, m, path, bucket = data.batch(rng); pred, h = model(x, g, m, capacity); b = F.cross_entropy(pred, y, reduction='none'); ok = (pred.argmax(1) == y).float()
        for j in range(3):
            mask = bucket == j; loss[j] += b[mask].sum(); correct[j] += ok[mask].sum(); count[j] += mask.sum()
        if capacity:
            sh = h.sum(0) if sh is None else sh + h.sum(0); gram = h.T @ h if gram is None else gram + h.T @ h
    out = {'cross_entropy': (loss / count).tolist(), 'accuracy': (correct / count).tolist(), 'bucket_counts': count.tolist()}
    if capacity:
        total = count.sum(); gram -= torch.outer(sh, sh) / total; out['capacity'] = (gram.trace().square() / gram.square().sum()).item()
    return out


def run(cell, decoder, args):
    k, mode, seed = cell; out = args.out / f'k{k}_{mode}_{seed}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists(): return
    hidden = {'dense_full': INPUT_DIM, 'dense_half': 1792, 'dense_budget': max(64, INPUT_DIM // (3 * k))}.get(mode, INPUT_DIM)
    rank = max(1, 16 // k); data = Data(k, decoder); rng = np.random.default_rng(seed); torch.manual_seed(seed); model = Student(k, mode, hidden, rank).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0); best = float('inf'); best_state = None; elapsed = 0.
    with (out / 'curve.jsonl').open('w', buffering=1) as log:
        for step in range(1, args.steps + 1):
            st = time.monotonic(); x, y, g, m, _, _ = data.batch(rng); pred, _ = model(x, g, m); l = F.cross_entropy(pred, y); opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); torch.cuda.synchronize(); elapsed += time.monotonic() - st
            if step == 1 or step % 100 == 0 or step == args.steps:
                val = evaluate(model, data, 8101, args.eval_batches); val['step'], val['seconds'] = step, elapsed; print(json.dumps(val), file=log)
                score = float(np.mean(val['cross_entropy']))
                if score < best: best, best_state = score, {n: v.detach().cpu() for n, v in model.state_dict().items()}
    torch.save(best_state, out / 'best.pt'); model.load_state_dict(best_state)
    # Full covariance is disabled for K>=8: the concatenated expert
    # representation is too wide for a dense Gram matrix on a 96GB card.
    test = evaluate(model, data, 8102, args.eval_batches, k <= 4)
    (out / 'result.json').write_text(json.dumps({'k': k, 'mode': mode, 'seed': seed, 'input_dim': INPUT_DIM, 'hidden': hidden, 'lora_rank': None if mode.startswith('dense') else rank, 'steps': args.steps, 'test': test, 'train_seconds': elapsed, 'frequency_split': 'bottom/top quartiles within selected pools', 'teacher_mapping': 'independent mapping per B2/T20/T21 cluster'}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1); ap.add_argument('--steps', type=int, default=6000); ap.add_argument('--eval-batches', type=int, default=128); ap.add_argument('--lr', type=float, default=.001); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args(); torch.set_num_threads(1)
    modes = ('specified', 'random', 'dense_full'); cells = [(k, m, s) for s, k, m in itertools.product((11, 22, 33), (2, 4, 8, 16), modes)]
    if args.smoke: cells = [(k, m, 11) for k in (2, 4, 8, 16) for m in modes]
    with np.load(SAE) as z: decoder = z['W_dec']
    if decoder.shape[1] != INPUT_DIM: raise ValueError(decoder.shape)
    for cell in cells[args.worker::args.workers]: run(cell, decoder, args)
