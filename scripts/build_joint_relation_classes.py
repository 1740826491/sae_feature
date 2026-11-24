"""Materialize joint classes containing CAR parents and their child composites."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--top-labels", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    parents_by_child = defaultdict(set)
    with Path(args.edges).open() as f:
        for row in csv.DictReader(f):
            parents_by_child[int(row["child"])].add(int(row["parent"]))

    top_rows = {}
    with Path(args.top_labels).open() as f:
        for row in csv.DictReader(f):
            top_rows[int(row["feature"])] = row

    groups = defaultdict(list)
    for child, row in top_rows.items():
        signature = ";".join(map(str, sorted(parents_by_child[child])))
        key = (signature, row["nested_top_cluster"])
        groups[key].append(child)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "joint_relation_members.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "feature", "role", "parent_signature", "nested_top_cluster"])
        for i, ((signature, nested), children) in enumerate(sorted(groups.items())):
            class_id = f"joint_{i:04d}"
            parents = [int(x) for x in signature.split(";") if x]
            for parent in parents:
                writer.writerow([class_id, parent, "bottom_constituent", signature, nested])
            for child in sorted(children):
                writer.writerow([class_id, child, "top_composite", signature, nested])

    with (out / "joint_relation_summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "num_bottom_features", "num_top_features", "bottom_features", "top_features", "nested_top_cluster"])
        for i, ((signature, nested), children) in enumerate(sorted(groups.items())):
            class_id = f"joint_{i:04d}"
            parents = [int(x) for x in signature.split(";") if x]
            writer.writerow([class_id, len(parents), len(children), ";".join(map(str, parents)), ";".join(map(str, sorted(children))), nested])


if __name__ == "__main__":
    main()
