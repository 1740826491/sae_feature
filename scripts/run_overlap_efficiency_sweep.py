#!/usr/bin/env python3
"""Overlap-matched synthetic learning/parameter-efficiency experiment.

The experiment keeps a single synthetic teacher/data mechanism and changes only
the pairwise Jaccard overlap between expert feature pools.  Inputs are sums of
real Gemma-Scope SAE decoder directions.  Common features use one shared
teacher map; private features use an expert-specific teacher map.  This makes
overlap represent duplicated/common work rather than label noise.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Cell:
    overlap_name: str
    target_overlap: float
    width: int
    seed: int
    mode: str


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def orthogonal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((rows, cols)).astype(np.float32))
    return q * np.sign(np.diag(r))


def load_overlap_points(path: Path) -> list[tuple[str, float]]:
    rows = list(csv.DictReader(path.open()))
    required = {"evaluation_depth", "feature_weighted_jaccard", "domain_weighted_jaccard_same_documents"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"bad overlap table: expected {sorted(required)}")
    by_depth: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        depth = int(row["evaluation_depth"])
        slot = by_depth.setdefault(depth, {"feature": [], "domain": []})
        slot["feature"].append(float(row["feature_weighted_jaccard"]))
        slot["domain"].append(float(row["domain_weighted_jaccard_same_documents"]))
    points = [("ideal_zero", 0.0)]
    for depth in sorted(by_depth):
        for kind in ("feature", "domain"):
            value = float(np.mean(by_depth[depth][kind]))
            points.append((f"real_{kind}_depth{depth}", value))
    points.append(("high_overlap_control", 0.75))
    return points


class Generator:
    def __init__(self, decoder: np.ndarray, experts: int, pool_size: int, overlap: float,
                 active_features: int, seed: int, device: torch.device):
        self.experts = experts
        self.pool_size = pool_size
        self.active_features = active_features
        # If every pool consists of C common and P-C disjoint private features,
        # pairwise Jaccard q=C/(2P-C), hence C=2qP/(1+q).
        common = int(round(2.0 * overlap * pool_size / (1.0 + overlap)))
        common = min(pool_size, max(0, common))
        private = pool_size - common
        self.realized_overlap = common / max(1, 2 * pool_size - common)
        self.common_count = common
        self.private_count = private
        total = common + experts * private
        if total > len(decoder):
            raise ValueError(f"need {total} decoder rows, have {len(decoder)}")
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(decoder), total, replace=False)
        directions = decoder[ids].astype(np.float32, copy=True)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
        self.directions = torch.as_tensor(directions, device=device)
        self.pools = []
        for expert in range(experts):
            private_ids = np.arange(common + expert * private, common + (expert + 1) * private)
            self.pools.append(np.concatenate([np.arange(common), private_ids]))

        input_dim = directions.shape[1]
        latent_dim, output_dim = 8, 64
        teacher_rng = np.random.default_rng(2718)
        maps = []
        for _ in range(experts + 1):  # map 0 shared, maps 1..K private
            u = orthogonal(teacher_rng, input_dim, latent_dim)
            v = orthogonal(teacher_rng, output_dim, latent_dim).T
            maps.append((u, v))
        targets = np.empty((total, output_dim), dtype=np.float32)
        for feature in range(total):
            owner = 0 if feature < common else 1 + (feature - common) // max(1, private)
            u, v = maps[owner]
            z = directions[feature] @ u * math.sqrt(input_dim)
            targets[feature] = ((z + 0.1 * np.tanh(z)) @ v) / math.sqrt(latent_dim)
        self.targets = torch.as_tensor(targets, device=device)
        self.device = device

    def batch(self, seed: int, step: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(stable_seed(seed, step, "data"))
        groups = np.arange(batch_size, dtype=np.int64) % self.experts
        rng.shuffle(groups)
        chosen = np.empty((batch_size, self.active_features), dtype=np.int64)
        for expert in range(self.experts):
            mask = np.flatnonzero(groups == expert)
            if len(mask):
                chosen[mask] = rng.choice(self.pools[expert], (len(mask), self.active_features), replace=True)
        amp = rng.uniform(0.5, 1.5, (batch_size, self.active_features, 1)).astype(np.float32)
        ids = torch.as_tensor(chosen, device=self.device)
        scale = torch.as_tensor(amp, device=self.device)
        x = (self.directions[ids] * scale).sum(1) / math.sqrt(self.active_features)
        y = (self.targets[ids] * scale).sum(1) / math.sqrt(self.active_features)
        return x, y, torch.as_tensor(groups, device=self.device)


class ExpertMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int):
        super().__init__()
        self.up = nn.Linear(input_dim, width, bias=False)
        self.down = nn.Linear(width, output_dim, bias=False)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        return self.down(h + 0.1 * h.tanh())


class Model(nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int, experts: int, mode: str):
        super().__init__()
        self.mode = mode
        self.experts = experts
        count = 1 if mode.startswith("dense") else experts
        dense_width = width * experts if mode == "dense_total" else width
        local_width = dense_width if count == 1 else width
        self.blocks = nn.ModuleList(ExpertMLP(input_dim, local_width, output_dim) for _ in range(count))

    def forward(self, x: torch.Tensor, group: torch.Tensor, random_route: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode.startswith("dense"):
            return self.blocks[0](x)
        route = group if self.mode == "oracle_moe" else random_route
        assert route is not None
        out = x.new_zeros((len(x), self.blocks[0].down.out_features))
        for expert, block in enumerate(self.blocks):
            idx = torch.nonzero(route == expert, as_tuple=False).flatten()
            if idx.numel():
                out.index_copy_(0, idx, block(x.index_select(0, idx)))
        return out


@torch.no_grad()
def evaluate(model: Model, data: Generator, seed: int, batches: int, batch_size: int) -> float:
    model.eval()
    err = power = 0.0
    for index in range(batches):
        x, y, group = data.batch(seed, 10_000_000 + index, batch_size)
        random_route = torch.remainder(torch.arange(len(x), device=x.device) + index, data.experts)
        pred = model(x, group, random_route)
        err += float((pred - y).square().sum())
        power += float(y.square().sum())
    model.train()
    return err / max(power, 1e-12)


def parameter_counts(model: Model, mode: str, experts: int) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    active = total if mode.startswith("dense") else total // experts
    return total, active


def run_cell(cell: Cell, args: argparse.Namespace, decoder: np.ndarray, device: torch.device) -> None:
    folder = args.out / cell.overlap_name / f"w{cell.width}_{cell.mode}_s{cell.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        return
    torch.manual_seed(cell.seed)
    data = Generator(decoder, args.experts, args.pool_size, cell.target_overlap,
                     args.active_features, args.generator_seed, device)
    model = Model(decoder.shape[1], cell.width, 64, args.experts, cell.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    total_params, active_params = parameter_counts(model, cell.mode, args.experts)
    best = float("inf")
    hits = {str(t): None for t in args.thresholds}
    streaks = {str(t): 0 for t in args.thresholds}
    started = time.monotonic()
    curve_path = folder / "curve.jsonl"
    with curve_path.open("w", buffering=1) as curve:
        for step in range(1, args.steps + 1):
            x, y, group = data.batch(cell.seed, step, args.batch_size)
            random_route = torch.remainder(torch.randperm(len(x), device=device), args.experts)
            pred = model(x, group, random_route)
            loss = (pred - y).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                nmse = evaluate(model, data, cell.seed + 90000, args.eval_batches, args.batch_size)
                if not math.isfinite(nmse):
                    raise RuntimeError(f"non-finite nmse for {cell}")
                best = min(best, nmse)
                record = {"step": step, "samples": step * args.batch_size, "nmse": nmse,
                          "train_mse": float(loss.detach()), "seconds": time.monotonic() - started}
                print(json.dumps(record), file=curve)
                for threshold in args.thresholds:
                    key = str(threshold)
                    streaks[key] = streaks[key] + 1 if nmse <= threshold else 0
                    if streaks[key] >= 3 and hits[key] is None:
                        hits[key] = record
    test_nmse = evaluate(model, data, cell.seed + 91000, args.test_batches, args.batch_size)
    result = {**asdict(cell), "realized_overlap": data.realized_overlap,
              "common_features": data.common_count, "private_features_per_expert": data.private_count,
              "experts": args.experts, "pool_size": args.pool_size, "active_features": args.active_features,
              "steps": args.steps, "batch_size": args.batch_size, "best_val_nmse": best,
              "test_nmse": test_nmse, "hits": hits, "total_params": total_params,
              "active_params": active_params, "seconds": time.monotonic() - started}
    tmp = folder / "result.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(folder / "result.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--overlap-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--active-features", type=int, default=4)
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--modes", nargs="+", default=["dense_active", "dense_total", "oracle_moe", "random_moe"])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--test-batches", type=int, default=64)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.1])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--generator-seed", type=int, default=20261126)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only-overlap", default=None)
    parser.add_argument("--only-width", type=int, default=None)
    parser.add_argument("--only-seed", type=int, default=None)
    args = parser.parse_args()
    if args.worker < 0 or args.worker >= args.workers:
        raise ValueError("worker must be in [0, workers)")
    args.out.mkdir(parents=True, exist_ok=True)
    points = load_overlap_points(args.overlap_table)
    cells = [Cell(name, value, width, seed, mode)
             for name, value in points for width in args.widths for seed in args.seeds for mode in args.modes]
    if args.only_overlap is not None:
        cells = [c for c in cells if c.overlap_name == args.only_overlap]
    if args.only_width is not None:
        cells = [c for c in cells if c.width == args.only_width]
    if args.only_seed is not None:
        cells = [c for c in cells if c.seed == args.only_seed]
    manifest = {"status": "running", "points": points, "cells": len(cells),
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "scientific_question": "How do sample and parameter efficiency degrade at measured real feature-pool overlap?",
                "overlap_definition": "pairwise Jaccard of equal-size expert feature pools"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with np.load(args.sae) as archive:
        decoder = archive["W_dec"]
    if decoder.ndim != 2 or decoder.shape[1] != 3584:
        raise ValueError(f"unexpected decoder shape {decoder.shape}")
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for index, cell in enumerate(cells):
        if index % args.workers == args.worker:
            run_cell(cell, args, decoder, device)
    if args.worker == 0:
        expected = len(cells)
        complete = len(list(args.out.glob("*/w*_*/result.json")))
        (args.out / f"worker_{args.worker}_complete.json").write_text(json.dumps({"expected_global": expected, "complete_now": complete}))


if __name__ == "__main__":
    main()
