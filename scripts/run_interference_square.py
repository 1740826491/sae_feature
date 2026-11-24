"""Square-output interference ablation: T, T+C, T+O, and T+C+O.

This reuses the Square Output teacher construction.  A single shared low-rank
student is trained on one target leaf alone or together with a common
component and/or other leaf components.  The target-only evaluation isolates
how much the added components interfere with learning the target mapping.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_linear_rank import Task, D, P, SAE, basis, tensor

BATCH = 384
TARGET = 2                 # T20_0
COMMON = 0                 # B2_0, shared by two paths in Square Output
OTHER_LEAVES = (3, 4, 5)   # T20_1, T21_0, T21_1
LAM = 0.25


class InterferenceTask:
    def __init__(self, decoder):
        # Exactly the Square Output teacher, including its fixed lift to D.
        base = Task(decoder)
        lift = tensor(basis(np.random.default_rng(3584), D, P).T)
        base.base = base.base @ lift
        base.delta = base.delta @ lift
        self.data = base.data
        self.base = base.base
        self.delta = base.delta

    def _draw_ids(self, rng, group, n):
        pool = self.data.pools[group]
        return rng.choice(pool, n)

    def batch(self, rng, condition):
        """Return x, y, and the target-only y for one condition."""
        has_common = condition in ("TC", "TCO")
        has_other = condition in ("TO", "TCO")
        groups = [TARGET]
        if has_common:
            groups.append(COMMON)
        if has_other:
            groups.append(None)  # draw a leaf from the union below

        ids = []
        ids.append(self._draw_ids(rng, TARGET, BATCH))
        if has_common:
            ids.append(self._draw_ids(rng, COMMON, BATCH))
        if has_other:
            pools = np.concatenate([self.data.pools[g] for g in OTHER_LEAVES])
            ids.append(rng.choice(pools, BATCH))
        ids = np.stack(ids, axis=1)
        amp = rng.uniform(0.5, 1.5, (BATCH, len(ids[0]), 1)).astype("float32")
        ids_t = tensor(ids)
        amp_t = tensor(amp)
        x = (self.data.inputs[ids_t] * amp_t).sum(1)
        y_parts = (self.base[ids_t] + LAM * self.delta[ids_t]) * amp_t
        y = y_parts.sum(1)
        target_y = y_parts[:, 0]
        return x, y, target_y


class Student(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, rank, bias=False),
            nn.Linear(rank, D, bias=False),
        )
        nn.init.zeros_(self.net[1].weight)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(task, model, seed, batches):
    rng = np.random.default_rng(seed)
    sums = {k: torch.zeros(2, device="cuda") for k in ("T", "TC", "TO", "TCO")}
    target_sums = torch.zeros(2, device="cuda")
    for _ in range(batches):
        x, _, target_y = task.batch(rng, "T")
        target_pred = model(x)
        target_sums += torch.stack(((target_pred - target_y).square().sum(), target_y.square().sum()))
    for condition in sums:
        for _ in range(batches):
            x, y, target_y = task.batch(rng, condition)
            pred = model(x)
            sums[condition][0] += (pred - y).square().sum()
            sums[condition][1] += y.square().sum()
    out = {}
    for condition in ("T", "TC", "TO", "TCO"):
        err, power = sums[condition].tolist()
        out[f"{condition}_nmse"] = err / power
    err, power = target_sums.tolist()
    out["target_only_nmse"] = err / power
    return out


def run_cell(task, condition, seed, args, out):
    folder = out / f"{condition}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        return
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Student(args.rank).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    steps = 2 if args.smoke else args.steps
    elapsed = 0.0
    with (folder / "curve.jsonl").open("w", buffering=1) as log:
        for step in range(1, steps + 1):
            start = time.monotonic()
            x, y, _ = task.batch(rng, condition)
            pred = model(x)
            loss = (pred - y).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - start
            if step == 1 or step % 100 == 0 or step == steps:
                metrics = evaluate(task, model, 8101, 1 if args.smoke else args.eval_batches)
                assert all(math.isfinite(v) for v in metrics.values())
                print(json.dumps(dict(step=step, seconds=elapsed, **metrics)), file=log)
    final = evaluate(task, model, 8102, 1 if args.smoke else args.eval_batches)
    torch.save(model.state_dict(), folder / "final.pt")
    result = dict(condition=condition, seed=seed, rank=args.rank,
                  steps=steps, batch=BATCH, train_seconds=elapsed, **final)
    (folder / "result.json").write_text(json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-batches", type=int, default=128)
    ap.add_argument("--rank", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    with np.load(SAE) as z:
        task = InterferenceTask(z["W_dec"])
    conditions = ("T", "TC", "TO", "TCO")
    cells = [(condition, seed) for seed in (11, 22, 33) for condition in conditions]
    if args.smoke:
        cells = [(condition, 11) for condition in conditions]
    for cell in cells[args.worker::args.workers]:
        run_cell(task, cell[0], cell[1], args, args.out)


if __name__ == "__main__":
    main()
