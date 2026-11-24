"""Relation-aware hierarchy labels.

1. Cluster CAR parent/constituent features into three bottom groups by document
   activation patterns.
2. Map each CAR child/composite feature to the bottom-group signature of its
   parents.
3. Split composites inside each dominant bottom group into two activation-
   pattern subgroups, preserving the bottom-to-top mapping.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer


def load_docs(input_dir: Path):
    docs = []
    for path in sorted(input_dir.glob("doc_features_rank*.npy")):
        docs.extend(np.load(path, allow_pickle=True).tolist())
    return docs


def activation_vectors(docs, feature_ids):
    feature_ids = np.asarray(sorted(feature_ids), dtype=np.int32)
    lookup = {int(fid): i for i, fid in enumerate(feature_ids)}
    rows, cols = [], []
    for doc_id, feature_list in enumerate(docs):
        for fid in set(map(int, feature_list)):
            row_id = lookup.get(fid)
            if row_id is not None:
                rows.append(row_id)
                cols.append(doc_id)
    incidence = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(feature_ids), len(docs)),
    ).tocsr()
    vectors = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(incidence)
    return feature_ids, incidence, vectors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bottom-k", type=int, default=3)
    ap.add_argument("--top-sub-k", type=int, default=2)
    args = ap.parse_args()

    input_dir = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    docs = load_docs(input_dir)

    parents_by_child = defaultdict(set)
    parent_nodes, child_nodes = set(), set()
    with Path(args.edges).open() as f:
        for row in csv.DictReader(f):
            parent, child = int(row["parent"]), int(row["child"])
            parents_by_child[child].add(parent)
            parent_nodes.add(parent)
            child_nodes.add(child)

    parent_ids, parent_incidence, parent_vectors = activation_vectors(docs, parent_nodes)
    bottom_model = MiniBatchKMeans(
        n_clusters=args.bottom_k, random_state=0, n_init=3, batch_size=512, max_iter=100
    )
    bottom_labels = bottom_model.fit_predict(parent_vectors)
    bottom_of = {int(fid): int(label) for fid, label in zip(parent_ids, bottom_labels)}

    child_ids, child_incidence, child_vectors = activation_vectors(docs, child_nodes)
    top_groups = defaultdict(list)
    child_signatures = {}
    child_dominant = {}
    for row_id, child in enumerate(child_ids):
        clusters = [bottom_of[p] for p in parents_by_child[int(child)] if p in bottom_of]
        counts = np.bincount(clusters, minlength=args.bottom_k) if clusters else np.zeros(args.bottom_k, dtype=int)
        dominant = int(np.argmax(counts)) if len(clusters) else -1
        signature = ";".join(map(str, sorted(set(clusters)))) if clusters else "none"
        child_signatures[int(child)] = signature
        child_dominant[int(child)] = dominant
        top_groups[dominant].append(row_id)

    top_sub = {}
    summaries = []
    for dominant, row_ids in sorted(top_groups.items()):
        if dominant < 0 or len(row_ids) < args.top_sub_k:
            for row_id in row_ids:
                top_sub[int(child_ids[row_id])] = 0
            summaries.append({"bottom_cluster": dominant, "subclusters": 1, "sizes": [len(row_ids)]})
            continue
        model = MiniBatchKMeans(
            n_clusters=args.top_sub_k,
            random_state=dominant,
            n_init=3,
            batch_size=512,
            max_iter=100,
        )
        labels = model.fit_predict(child_vectors[row_ids])
        for row_id, sub in zip(row_ids, labels):
            top_sub[int(child_ids[row_id])] = int(sub)
        summaries.append(
            {
                "bottom_cluster": dominant,
                "subclusters": args.top_sub_k,
                "sizes": np.bincount(labels, minlength=args.top_sub_k).tolist(),
            }
        )

    with (output / "bottom_cluster_labels.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "bottom_cluster", "activation_docs"])
        for row_id, (fid, label) in enumerate(zip(parent_ids, bottom_labels)):
            writer.writerow([int(fid), int(label), int(parent_incidence.getrow(row_id).nnz)])

    with (output / "top_cluster_labels.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["feature", "parent_bottom_cluster", "parent_cluster_signature", "top_subcluster", "nested_top_cluster", "activation_docs"]
        )
        for row_id, fid in enumerate(child_ids):
            fid = int(fid)
            dominant = child_dominant[fid]
            sub = top_sub[fid]
            writer.writerow(
                [fid, dominant, child_signatures[fid], sub, f"bottom_{dominant}_top_{sub}", int(child_incidence.getrow(row_id).nnz)]
            )

    (output / "complete.json").write_text(
        json.dumps(
            {
                "method": "CAR parent mapping + document activation-pattern clustering",
                "documents": len(docs),
                "bottom_features": len(parent_ids),
                "top_features": len(child_ids),
                "bottom_k": args.bottom_k,
                "top_sub_k": args.top_sub_k,
                "top_group_summaries": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
