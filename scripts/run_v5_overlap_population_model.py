#!/usr/bin/env python3
"""V5-style exact population-GD model with overlapping leaf ownership.

Each feature has one fixed target.  A feature owned by m of L leaf routes is
seen globally with frequency m/L but conditionally with frequency one inside
each owning expert.  Thus its ideal conditional-gradient speedup is L/m.

For the symmetric common/private construction, every leaf pool has the same
size, a fraction s of its features are shared by all leaves and 1-s are
private.  Pairwise pool Jaccard q implies s=2q/(1+q).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def first_hit(error_fn, threshold: float, limit: int = 10_000_000) -> int:
    for step in range(limit + 1):
        if error_fn(step) <= threshold:
            return step
    raise RuntimeError("threshold not reached")


def residual_sq(rate: float, step: int) -> float:
    return (1.0 - rate) ** (2 * step)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--overlap-table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--leaves", type=int, default=8)
    p.add_argument("--conditional-rate", type=float, default=0.05,
                   help="eta/n in the orthogonal population model")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.1])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    raw = list(csv.DictReader(a.overlap_table.open()))
    by_depth: dict[int, dict[str, list[float]]] = {}
    for row in raw:
        d = int(row["evaluation_depth"])
        x = by_depth.setdefault(d, {"feature": [], "domain": []})
        x["feature"].append(float(row["feature_weighted_jaccard"]))
        x["domain"].append(float(row["domain_weighted_jaccard_same_documents"]))
    points = [("ideal_zero", 0.0)]
    for d in sorted(by_depth):
        for kind in ("feature", "domain"):
            vals = by_depth[d][kind]
            points.append((f"real_{kind}_depth{d}", sum(vals) / len(vals)))
    points.append(("high_overlap_control", 0.75))

    L, alpha = a.leaves, a.conditional_rate
    rows = []
    for name, q in points:
        shared_fraction = 2 * q / (1 + q) if q < 1 else 1.0
        # Root is present on all L paths; middle on L/2; a private leaf on 1;
        # the symmetric shared-leaf component is present on all L paths.
        components = {
            "root_common": L,
            "middle": L // 2,
            "leaf_private": 1,
            "leaf_shared": L,
        }
        for threshold in a.thresholds:
            for component, owners in components.items():
                dense_rate = alpha * owners / L
                tree_rate = alpha
                dense_steps = first_hit(lambda t, r=dense_rate: residual_sq(r, t), threshold)
                tree_steps = first_hit(lambda t, r=tree_rate: residual_sq(r, t), threshold)
                rows.append({
                    "overlap_name": name, "pairwise_jaccard": q,
                    "shared_fraction_in_each_leaf_pool": shared_fraction,
                    "component": component, "owner_multiplicity": owners,
                    "threshold": threshold, "dense_steps": dense_steps,
                    "tree_steps": tree_steps,
                    "tree_speedup": dense_steps / tree_steps,
                    "parameter_copies_in_tree": owners,
                })

            # Overall leaf error mixes private and all-leaf-shared features.
            dense_error = lambda t: (
                (1 - shared_fraction) * residual_sq(alpha / L, t)
                + shared_fraction * residual_sq(alpha, t)
            )
            tree_error = lambda t: residual_sq(alpha, t)
            dense_steps = first_hit(dense_error, threshold)
            tree_steps = first_hit(tree_error, threshold)
            rows.append({
                "overlap_name": name, "pairwise_jaccard": q,
                "shared_fraction_in_each_leaf_pool": shared_fraction,
                "component": "leaf_mixture", "owner_multiplicity": "mixture",
                "threshold": threshold, "dense_steps": dense_steps,
                "tree_steps": tree_steps,
                "tree_speedup": dense_steps / tree_steps,
                "parameter_copies_in_tree": 1 + shared_fraction * (L - 1),
            })

    with (a.out / "v5_overlap_population_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (a.out / "complete.json").write_text(json.dumps({
        "status": "complete", "leaves": L, "conditional_rate": alpha,
        "model": "orthogonal fixed-target exact population GD",
        "rows": len(rows),
        "invariants": ["same feature targets", "same global steps", "same alpha", "no sampling noise"],
    }, indent=2))


if __name__ == "__main__":
    main()
