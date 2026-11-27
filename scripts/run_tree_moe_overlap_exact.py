#!/usr/bin/env python3
"""Exact population-GD overlap extension of HengjieCao's Tree-MoE T3 experiment.

The original controlled experiment assigns every leaf feature to exactly one leaf.
This runner keeps the same 1-2-8 tree, level-separated representation, perfect
routing, population GD, and per-expert 1/pi learning-rate correction, but permits
one canonical leaf feature (one direction, one target) to belong to multiple leaf
pools.  Thus overlap changes feature frequency without silently creating different
tasks for duplicated copies of a feature.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def balanced_memberships(experts: int, pool_size: int, target_jaccard: float, seed: int):
    """Create approximately exchangeable feature ownership with fixed pool sizes.

    If every feature has owner multiplicity m, expected pairwise Jaccard is
    (m-1)/(2*(experts-1)-(m-1)).  We mix floor/ceil multiplicities to realize
    arbitrary targets, then use a greedy balanced assignment.
    """
    if not 0 <= target_jaccard < 1:
        raise ValueError("target_jaccard must be in [0, 1)")
    rng = np.random.default_rng(seed)
    mean_m = 1 + 2 * target_jaccard * (experts - 1) / (1 + target_jaccard)
    lo, hi = int(math.floor(mean_m)), int(math.ceil(mean_m))
    total_slots = experts * pool_size
    if lo == hi:
        mults = [lo] * int(round(total_slots / lo))
    else:
        # Search a small integer solution whose total incidence is exact and
        # whose incidence-weighted mean multiplicity is closest to mean_m.
        best = None
        for nhi in range(total_slots // hi + 1):
            rem = total_slots - nhi * hi
            if rem >= 0 and rem % lo == 0:
                nlo = rem // lo
                achieved = (nlo * lo * lo + nhi * hi * hi) / total_slots
                score = abs(achieved - mean_m)
                if best is None or score < best[0]:
                    best = (score, nlo, nhi)
        if best is None:
            raise RuntimeError("could not construct exact incidence budget")
        _, nlo, nhi = best
        mults = [lo] * nlo + [hi] * nhi
    rng.shuffle(mults)

    owners = []
    loads = np.zeros(experts, dtype=int)
    pair_load = np.zeros((experts, experts), dtype=int)
    for m in mults:
        # Randomized greedy choice balances both marginal pool sizes and pairs.
        candidates = []
        for _ in range(128):
            cand = np.sort(rng.choice(experts, m, replace=False))
            score = loads[cand].sum() * 1000 + pair_load[np.ix_(cand, cand)].sum()
            candidates.append((score + rng.random(), cand))
        cand = min(candidates, key=lambda x: x[0])[1]
        owners.append(cand.tolist())
        loads[cand] += 1
        for a in cand:
            for b in cand:
                if a < b:
                    pair_load[a, b] += 1
    if not np.all(loads == pool_size):
        raise RuntimeError(f"unbalanced ownership: {loads.tolist()}")
    return owners


def ownership_stats(owners, experts):
    pools = [set() for _ in range(experts)]
    for f, own in enumerate(owners):
        for e in own:
            pools[e].add(f)
    js = []
    for i in range(experts):
        for j in range(i + 1, experts):
            js.append(len(pools[i] & pools[j]) / len(pools[i] | pools[j]))
    hist = {}
    for own in owners:
        hist[len(own)] = hist.get(len(own), 0) + 1
    return {
        "pairwise_jaccard_mean": float(np.mean(js)),
        "pairwise_jaccard_std": float(np.std(js)),
        "unique_leaf_features": len(owners),
        "owner_multiplicity_mean_feature": float(np.mean([len(x) for x in owners])),
        "owner_multiplicity_mean_activation": float(np.average([len(x) for x in owners], weights=[len(x) for x in owners])),
        "multiplicity_hist": hist,
    }


def make_problem(jaccard, target_conflict, seed, width=358, dense_width=3584,
                 k=8, block=128, dout=64, device="cpu"):
    """Return level-block directions, targets, path-feature lists and weights."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    leaf_owners = balanced_memberships(8, block, jaccard, seed + 101)
    stats = ownership_stats(leaf_owners, 8)

    # Canonical features: 128 root, 2*128 middle, and the unique leaf features.
    counts = [block, 2 * block, len(leaf_owners)]
    offsets = np.cumsum([0] + counts)
    nfeat = offsets[-1]
    din = 3 * width
    U_tree = torch.zeros(din, nfeat)
    levels = np.concatenate([np.full(counts[l], l) for l in range(3)])
    for l in range(3):
        a, b = offsets[l], offsets[l + 1]
        z = torch.randn(width, b - a, generator=g)
        z /= z.norm(dim=0, keepdim=True)
        U_tree[l * width:(l + 1) * width, a:b] = z
    # Match the senior's stageZ comparison: Dense reads one roomy 3584-d mixed
    # representation, while Tree experts read 358-d level blocks.  Using the
    # 1074-d concatenated tree representation for Dense creates an unintended
    # capacity bottleneck when the global canonical feature count exceeds 1074.
    U_dense = torch.randn(dense_width, nfeat, generator=g)
    U_dense /= U_dense.norm(dim=0, keepdim=True)
    T = torch.randn(dout, nfeat, generator=g) / math.sqrt(dout)
    # Path-conditional teachers separate two meanings of overlap:
    # rho=0: the same canonical feature has the same target in every domain;
    # rho>0: its target contains a domain-specific component.  Only multiply-owned
    # leaf features receive this component; root/middle and private leaf features
    # retain the original fixed t_f from the theory.
    T_path = T.unsqueeze(0).repeat(8, 1, 1)
    if target_conflict > 0:
        for leaf in range(8):
            for f, own in enumerate(leaf_owners):
                if leaf not in own or len(own) == 1:
                    continue
                idx = offsets[2] + f
                z = torch.randn(dout, generator=g)
                z = z / z.norm() * T[:, idx].norm()
                mixed = math.sqrt(1 - target_conflict) * T[:, idx] + math.sqrt(target_conflict) * z
                T_path[leaf, :, idx] = mixed / mixed.norm() * T[:, idx].norm()

    paths = []
    for leaf in range(8):
        root = list(range(offsets[0], offsets[1]))
        mid = list(range(offsets[1] + (leaf // 4) * block, offsets[1] + (leaf // 4 + 1) * block))
        lf = [offsets[2] + f for f, own in enumerate(leaf_owners) if leaf in own]
        assert len(lf) == block
        paths.append([root, mid, lf])
    stats["target_conflict"] = target_conflict
    return U_dense.to(device), U_tree.to(device), T_path.to(device), levels, paths, stats


def train_one(jaccard, target_conflict, model_kind, lr, steps, seed, width, dense_width,
              k, block, dout, threshold, device):
    U_dense, U_tree, T_path, levels, paths, stats = make_problem(
        jaccard, target_conflict, seed, width, dense_width, k, block, dout, device)
    nfeat = U_dense.shape[1]
    # Each path is equiprobable and each member of an on-path pool activates k/block.
    w_path = (1 / 8) * (k / block) * (13 / 12)
    if model_kind == "dense":
        params = [torch.nn.Parameter(torch.zeros(dout, dense_width, device=device))]
        groups = [{"params": params, "lr": lr}]
    else:
        # 1 root + 2 middle + 8 leaf experts. Each expert reads only its level block.
        params = [torch.nn.Parameter(torch.zeros(dout, width, device=device)) for _ in range(11)]
        pis = [1.0] + [0.5] * 2 + [0.125] * 8
        groups = [{"params": [p], "lr": lr / pi} for p, pi in zip(params, pis)]
    opt = torch.optim.SGD(groups)

    def errors():
        acc = torch.zeros(nfeat, device=device)
        freq = torch.zeros(nfeat, device=device)
        with torch.no_grad():
            for leaf, pools in enumerate(paths):
                expert_ids = [0, 1 + leaf // 4, 3 + leaf]
                for level, ids in enumerate(pools):
                    idx = torch.tensor(ids, device=device)
                    if model_kind == "dense":
                        pred = params[0] @ U_dense[:, idx]
                    else:
                        pred = params[expert_ids[level]] @ U_tree[level * width:(level + 1) * width, idx]
                    target = T_path[leaf, :, idx]
                    e = (pred - target).pow(2).sum(0) / target.pow(2).sum(0)
                    acc.index_add_(0, idx, e / 8)
                    freq.index_add_(0, idx, torch.full_like(e, 1 / 8))
        return acc / freq.clamp_min(1e-12), freq

    logs, hit = [], {"root": None, "middle": None, "leaf": None, "leaf_private": None, "leaf_shared": None}
    schedule = set([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, steps])
    for step in range(steps + 1):
        if step in schedule:
            e, freq = errors()
            leaf_mask = torch.tensor(levels == 2, device=device)
            mult = freq * 8
            vals = {
                "root": float(e[torch.tensor(levels == 0, device=device)].mean()),
                "middle": float(e[torch.tensor(levels == 1, device=device)].mean()),
                "leaf": float((e[leaf_mask] * freq[leaf_mask]).sum() / freq[leaf_mask].sum()),
                "leaf_private": float(e[leaf_mask & (mult < 1.5)].mean()) if bool((leaf_mask & (mult < 1.5)).any()) else None,
                "leaf_shared": float(e[leaf_mask & (mult >= 1.5)].mean()) if bool((leaf_mask & (mult >= 1.5)).any()) else None,
            }
            logs.append({"step": step, **vals})
            for key, val in vals.items():
                if val is not None and hit[key] is None and val <= threshold:
                    hit[key] = step
        if step == steps:
            break
        loss = torch.zeros((), device=device)
        for leaf, pools in enumerate(paths):
            expert_ids = [0, 1 + leaf // 4, 3 + leaf]
            for level, ids in enumerate(pools):
                idx = torch.tensor(ids, device=device)
                if model_kind == "dense":
                    pred = params[0] @ U_dense[:, idx]
                else:
                    pred = params[expert_ids[level]] @ U_tree[level * width:(level + 1) * width, idx]
                loss = loss + w_path * (pred - T_path[leaf, :, idx]).pow(2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            break
    return {"jaccard_target": jaccard, "target_conflict": target_conflict,
            "model": model_kind, "lr": lr, "seed": seed,
            "hit_steps": hit, "curve": logs, "ownership": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jaccards", default="0,0.1,0.1546,0.2504,0.4761,0.6044")
    ap.add_argument("--target-conflicts", default="0,0.25,0.5,1")
    ap.add_argument("--lrs", default="0.25,0.5,1,2,4,8,16,32,64")
    ap.add_argument("--seeds", default="7,8,9")
    ap.add_argument("--steps", type=int, default=610)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--dense-width", type=int, default=3584)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--dout", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = vars(args).copy()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for q in map(float, args.jaccards.split(",")):
        for rho in map(float, args.target_conflicts.split(",")):
            for model in ("dense", "tree"):
                for lr in map(float, args.lrs.split(",")):
                    for seed in map(int, args.seeds.split(",")):
                        name = f"q{q:g}_rho{rho:g}_{model}_lr{lr:g}_s{seed}.json"
                        path = out / name
                        if path.exists():
                            continue
                        try:
                            result = train_one(q, rho, model, lr, args.steps, seed, args.width, args.dense_width, args.k,
                                               args.block, args.dout, args.threshold, args.device)
                        except Exception as exc:
                            result = {"error": repr(exc), "jaccard_target": q, "target_conflict": rho,
                                      "model": model, "lr": lr, "seed": seed}
                        path.write_text(json.dumps(result))
                        print(name, result.get("hit_steps", result.get("error")), flush=True)


if __name__ == "__main__":
    main()
