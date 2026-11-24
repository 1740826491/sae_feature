"""Select the six non-B1 experts and valid CAR edges for the main MoE run."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottom-labels", required=True)
    ap.add_argument("--top-labels", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    bottom = {}
    with Path(args.bottom_labels).open() as f:
        for row in csv.DictReader(f):
            cluster = int(row["bottom_cluster"])
            if cluster in (0, 2):
                bottom[int(row["feature"])] = f"B{cluster}"

    top = {}
    with Path(args.top_labels).open() as f:
        for row in csv.DictReader(f):
            nested = row["nested_top_cluster"]
            if nested in {"bottom_0_top_0", "bottom_0_top_1", "bottom_2_top_0", "bottom_2_top_1"}:
                top[int(row["feature"])] = {
                    "expert": nested.replace("bottom_", "T").replace("_top_", ""),
                    "parent_cluster": row["parent_bottom_cluster"],
                    "nested": nested,
                }

    edges = []
    with Path(args.edges).open() as f:
        for row in csv.DictReader(f):
            parent, child = int(row["parent"]), int(row["child"])
            if parent in bottom and child in top:
                edges.append({**row, "bottom_expert": bottom[parent], "top_expert": top[child]["expert"]})

    children = {int(row["child"]) for row in edges}
    with (out / "bottom_labels.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["feature", "expert"])
        for feature, expert in sorted(bottom.items()):
            w.writerow([feature, expert])

    with (out / "top_labels.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["feature", "expert", "parent_bottom_cluster", "nested_top_cluster"])
        for feature, info in sorted(top.items()):
            if feature in children:
                w.writerow([feature, info["expert"], info["parent_cluster"], info["nested"]])

    with (out / "valid_edges.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["parent", "child", "child_frequency", "joint_containment", "bottom_expert", "top_expert"])
        w.writeheader()
        w.writerows(edges)

    experts = ["B0", "B2", "T00", "T01", "T20", "T21"]
    (out / "expert_manifest.json").write_text(
        json.dumps(
            {
                "experts": experts,
                "dropped_bottom_expert": "B1",
                "bottom_feature_count": len(bottom),
                "top_feature_count": len(children),
                "valid_edge_count": len(edges),
                "sampling_rule": "sample only from valid_edges.csv; never independently pair unrelated features",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

