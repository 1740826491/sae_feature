"""No-projection classification experiment with long-tail-specific teachers.

Each B2/T20/T21 cluster owns a distinct teacher mapping.  The input remains
the full 3584-dimensional SAE decoder space.  MoE experts share a base MLP;
expert-specific parameters are low-rank LoRA updates on both linear layers.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/mnt/workspace/Hrj/gemma_hierarchy")
LABELS = ROOT / "results/dclm_1m_9b_residpost_top150_car_A/decoder_similarity_clusters"
SAE = ROOT / "models/gemma-scope-9b-pt-res/layer_27/width_131k/average_l0_56/params.npz"
BATCH = 384
INPUT_DIM = 3584
DEVICE = "cuda"


def tensor(x):
    return torch.as_tensor(x, device=DEVICE)


def orth(rng, n, r):
    q, t = np.linalg.qr(rng.standard_normal((n, r)).astype("float32"))
    return q * np.sign(np.diag(t))


class Data:
    def __init__(self, k: int, decoder: np.ndarray):
        self.k = k
        self.num_classes = 2 * k
        ids: list[int] = []
        self.pools: list[np.ndarray] = []
        for name in ("B2", "T20", "T21"):
            with (LABELS / name / f"labels_k{k}.csv").open() as f:
                rows = list(csv.DictReader(f))
            for j in range(k):
                group = [int(x["feature"]) for x in rows if int(x["cluster"]) == j]
                if not group:
                    raise ValueError(f"empty {name} cluster {j}")
                start = len(ids)
                ids.extend(group)
                self.pools.append(np.arange(start, len(ids)))

        raw = decoder[ids].astype("float32")
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        self.inputs = tensor(raw * 1.0)  # no projection: always 3584 dimensions

        rng = np.random.default_rng(9000 + k)
        latent = []
        for _ in ids:
            latent.append(rng.standard_normal(8).astype("float32"))
        self.latent = tensor(np.asarray(latent))

        # One distinct teacher map for every B2/T20/T21 cluster.
        teacher = []
        for _ in range(3 * k):
            teacher.append((rng.standard_normal((8, self.num_classes)).astype("float32") / math.sqrt(8)))
        self.teacher = tensor(np.asarray(teacher))

    def batch(self, rng):
        path_count = 2 * self.k
        pairs = np.tile(np.arange(path_count * path_count), BATCH // (path_count * path_count))
        pairs = np.concatenate([pairs, rng.choice(path_count * path_count, BATCH - len(pairs), replace=False)])
        rng.shuffle(pairs)
        path, mixed = pairs // path_count, pairs % path_count

        # Paths 0..k-1 use T20; paths k..2k-1 use T21.
        groups = np.stack([path % self.k, self.k + path], axis=1)
        mixed_groups = np.stack([mixed % self.k, self.k + mixed], axis=1)
        ids = np.empty_like(groups)
        for side in range(2):
            for g in np.unique(groups[:, side]):
                mask = groups[:, side] == g
                ids[mask, side] = rng.choice(self.pools[g], mask.sum())

        ids_t = tensor(ids)
        amp = tensor(rng.uniform(0.5, 1.5, (BATCH, 2, 1)).astype("float32"))
        x = (self.inputs[ids_t] * amp).sum(1)
        z = self.latent[ids_t] * amp
        phi = z + 0.1 * z.tanh()
        logits = torch.einsum("bsi,bsic->bsc", phi, self.teacher[tensor(groups)])
        logits = logits.sum(1) / math.sqrt(8)
        target = logits.argmax(1)
        return x, target, tensor(groups), tensor(mixed_groups), tensor(path)


class SharedLoRAExperts(nn.Module):
    def __init__(self, input_dim: int, hidden: int, out_dim: int, experts: int, rank: int, alpha: float):
        super().__init__()
        self.experts = experts
        self.scale = alpha / rank
        self.enc_base = nn.Linear(input_dim, hidden, bias=False)
        self.dec_base = nn.Linear(hidden, out_dim, bias=False)
        self.enc_a = nn.Parameter(torch.empty(experts, rank, input_dim))
        self.enc_b = nn.Parameter(torch.empty(experts, hidden, rank))
        self.dec_a = nn.Parameter(torch.empty(experts, rank, hidden))
        self.dec_b = nn.Parameter(torch.empty(experts, out_dim, rank))
        nn.init.xavier_uniform_(self.enc_base.weight)
        nn.init.xavier_uniform_(self.dec_base.weight)
        for p in (self.enc_a, self.enc_b, self.dec_a, self.dec_b):
            nn.init.normal_(p, std=0.02)

    def expert_forward(self, x, expert):
        h = F.linear(x, self.enc_base.weight)
        h = h + F.linear(F.linear(x, self.enc_a[expert]), self.enc_b[expert]) * self.scale
        h = h + 0.1 * h.tanh()
        y = F.linear(h, self.dec_base.weight)
        y = y + F.linear(F.linear(h, self.dec_a[expert]), self.dec_b[expert]) * self.scale
        return y, h


class Student(nn.Module):
    def __init__(self, input_dim: int, hidden: int, k: int, mode: str, rank: int):
        super().__init__()
        self.mode = mode
        self.num_classes = 2 * k
        if mode.startswith("dense"):
            self.dense_enc = nn.Linear(input_dim, hidden, bias=False)
            self.dense_dec = nn.Linear(hidden, self.num_classes, bias=False)
            nn.init.xavier_uniform_(self.dense_enc.weight)
            nn.init.xavier_uniform_(self.dense_dec.weight)
            self.gate = None
            self.moe = None
        else:
            self.moe = SharedLoRAExperts(input_dim, hidden, self.num_classes, 3 * k, rank, alpha=2 * rank)
            self.gate = nn.Linear(input_dim, 3 * k) if mode == "learned" else None

    def forward(self, x, groups, mixed, capture=False):
        if self.mode.startswith("dense"):
            h = self.dense_enc(x)
            h = h + 0.1 * h.tanh()
            return self.dense_dec(h), h

        route = mixed if self.mode == "mixed" else groups
        if self.gate is not None:
            logits = self.gate(x)
            route = logits.topk(2, dim=1).indices
            prob = logits.softmax(1)

        out = x.new_zeros(len(x), self.num_classes)
        blocks = []
        for i in range(self.moe.experts):
            rows = (route == i).any(1).nonzero().flatten()
            if not rows.numel():
                continue
            value, h = self.moe.expert_forward(x[rows], i)
            if self.gate is not None:
                value = value * (1 + 2 * (prob[rows, i:i + 1] - prob[rows, i:i + 1].detach()))
            out.index_add_(0, rows, value)
            if capture:
                block = torch.zeros(len(x), h.shape[1], device=x.device, dtype=x.dtype)
                blocks.append(block.index_copy(0, rows, h))
        return out, torch.cat(blocks, 1) if capture and blocks else None


@torch.no_grad()
def evaluate(model, data, seed, batches, capacity=False):
    rng = np.random.default_rng(seed)
    total_loss = total_correct = total = 0
    path_correct = torch.zeros(2 * data.k, device=DEVICE)
    path_total = torch.zeros_like(path_correct)
    sum_h = gram = None
    for _ in range(batches):
        x, target, groups, mixed, path = data.batch(rng)
        logits, h = model(x, groups, mixed, capacity)
        total_loss += F.cross_entropy(logits, target, reduction="sum").item()
        total_correct += (logits.argmax(1) == target).sum().item()
        total += len(target)
        path_total.index_add_(0, path, torch.ones_like(path, dtype=path_total.dtype))
        path_correct.index_add_(0, path, (logits.argmax(1) == target).float())
        if capacity:
            if sum_h is None:
                sum_h, gram = h.sum(0), h.T @ h
            else:
                sum_h += h.sum(0)
                gram += h.T @ h
    result = {
        "cross_entropy": total_loss / total,
        "accuracy": total_correct / total,
        "path_accuracy": (path_correct / path_total).tolist(),
    }
    if capacity:
        gram -= torch.outer(sum_h, sum_h) / total
        result["capacity"] = (gram.trace().square() / gram.square().sum()).item()
    return result


def run(cell, decoder, args):
    k, mode, seed = cell
    folder = args.out / f"k{k}_{mode}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        return
    hidden = 1792 if mode == "dense_half" else INPUT_DIM
    data = Data(k, decoder)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = Student(INPUT_DIM, hidden, k, mode, args.lora_rank).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    best = float("inf")
    best_state = None
    elapsed = 0.0
    with (folder / "curve.jsonl").open("w", buffering=1) as log:
        for step in range(1, args.steps + 1):
            start = time.monotonic()
            x, target, groups, mixed, _ = data.batch(rng)
            pred, _ = model(x, groups, mixed)
            loss = F.cross_entropy(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - start
            if step == 1 or step % 100 == 0 or step == args.steps:
                metrics = evaluate(model, data, 8101, args.eval_batches)
                metrics.update(step=step, seconds=elapsed)
                print(json.dumps(metrics), file=log)
                if metrics["cross_entropy"] < best:
                    best = metrics["cross_entropy"]
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"no checkpoint produced for {cell}")
    torch.save(best_state, folder / "best.pt")
    model.load_state_dict(best_state)
    test = evaluate(model, data, 8102, args.eval_batches, True)
    result = {
        "k": k, "mode": mode, "seed": seed, "input_dim": INPUT_DIM,
        "hidden": hidden, "lora_rank": args.lora_rank if not mode.startswith("dense") else None,
        "classes": 2 * k, "steps": args.steps, "batch": BATCH,
        "best_val_cross_entropy": best, "test": test, "train_seconds": elapsed,
        "teacher_mapping": "one independent mapping per B2/T20/T21 cluster",
    }
    (folder / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--eval-batches", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    modes = ("specified", "learned", "mixed", "dense_half", "dense_full")
    cells = [(k, m, s) for s, k, m in itertools.product((11, 22, 33), (2, 4), modes)]
    if args.smoke:
        cells = [(k, m, 11) for k in (2, 4) for m in modes]
    with np.load(SAE) as z:
        decoder = z["W_dec"]
    if decoder.shape[1] != INPUT_DIM:
        raise ValueError(f"expected decoder dimension {INPUT_DIM}, got {decoder.shape}")
    for cell in cells[args.worker::args.workers]:
        run(cell, decoder, args)
