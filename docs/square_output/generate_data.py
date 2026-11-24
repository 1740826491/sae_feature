"""Export deterministic batches from the exact square-output experiment."""
import argparse
from pathlib import Path
import numpy as np
from run_linear_rank import Task, D, P, SAE, basis, tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=8102)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(SAE) as weights:
        task = Task(weights['W_dec'])
    lift = tensor(basis(np.random.default_rng(3584), D, P).T)
    task.base = task.base @ lift
    task.delta = task.delta @ lift
    rng = np.random.default_rng(args.seed)
    for i in range(args.batches):
        task.batch(rng, D, .25, export=args.out / f'batch_{i:05d}.npz')
