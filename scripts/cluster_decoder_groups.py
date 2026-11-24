"""Cluster selected SAE features by decoder-vector similarity.

The SAE decoder rows are L2-normalized before KMeans, so Euclidean distance
on the normalized rows is equivalent to cosine-distance ordering.  This is
deliberately independent of activation frequency and CAR edges.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def read_group_ids(bottom_path: Path, top_path: Path) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"B2": [], "T20": [], "T21": []}
    with bottom_path.open() as f:
        for row in csv.DictReader(f):
            feature = int(row["feature"])
            if row.get("bottom_cluster") == "2":
                groups["B2"].append(feature)
    with top_path.open() as f:
        for row in csv.DictReader(f):
            feature = int(row["feature"])
            nested = row.get("nested_top_cluster", "")
            if nested == "bottom_2_top_0":
                groups["T20"].append(feature)
            elif nested == "bottom_2_top_1":
                groups["T21"].append(feature)
    for name in groups:
        groups[name] = sorted(set(groups[name]))
        if not groups[name]:
            raise ValueError(f"no features found for {name} in the supplied label files")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Gemma Scope params.npz")
    ap.add_argument("--bottom-labels", required=True)
    ap.add_argument("--top-labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 4])
    args = ap.parse_args()

    params = np.load(args.params, mmap_mode="r")
    decoder = params["W_dec"]
    if decoder.ndim != 2:
        raise ValueError(f"expected W_dec[features, dim], got {decoder.shape}")

    groups = read_group_ids(Path(args.bottom_labels), Path(args.top_labels))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"method": "L2-normalized SAE decoder rows + sklearn KMeans", "groups": {}}

    for group, feature_ids in groups.items():
        ids = np.asarray(feature_ids, dtype=np.int64)
        x = np.asarray(decoder[ids], dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError(f"zero decoder row in {group}")
        x /= norms

        group_dir = out / group
        group_dir.mkdir(exist_ok=True)
        group_summary = {"feature_count": len(feature_ids), "results": {}}
        for k in args.ks:
            if k > len(feature_ids):
                raise ValueError(f"k={k} exceeds {group} feature count")
            model = KMeans(
                n_clusters=k,
                random_state=0,
                n_init=10,
                max_iter=100,
                algorithm="lloyd",
            )
            labels = model.fit_predict(x)
            inertia = float(model.inertia_)
            sizes = np.bincount(labels, minlength=k).tolist()
            centers = model.cluster_centers_.astype(np.float32)
            centers /= np.linalg.norm(centers, axis=1, keepdims=True)
            assigned_cos = np.sum(x * centers[labels], axis=1)
            center_cos = centers @ centers.T
            name = f"labels_k{k}.csv"
            with (group_dir / name).open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["feature", "cluster", "decoder_norm"])
                for fid, label, norm in zip(feature_ids, labels, norms[:, 0]):
                    w.writerow([fid, int(label), float(norm)])
            np.save(group_dir / f"centers_k{k}.npy", centers)
            group_summary["results"][str(k)] = {
                "inertia": inertia,
                "sizes": sizes,
                "mean_cosine_to_own_center": float(assigned_cos.mean()),
                "min_cosine_to_own_center": float(assigned_cos.min()),
                "center_cosine_matrix": center_cos.tolist(),
                "labels": str(group_dir / name),
                "centers": str(group_dir / f"centers_k{k}.npy"),
            }
        (group_dir / "summary.json").write_text(json.dumps(group_summary, indent=2))
        summary["groups"][group] = group_summary

    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
