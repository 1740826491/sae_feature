"""Nested activation-pattern labels: 3 coarse groups, 2 subgroups per group."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer


def load_feature_ids(labels_path: Path, label: str) -> np.ndarray:
    with labels_path.open() as f:
        return np.array(
            [int(row["feature"]) for row in csv.DictReader(f) if row["label"] == label],
            dtype=np.int32,
        )


def build_vectors(input_dir: Path, feature_ids: np.ndarray):
    feature_to_row = {int(fid): i for i, fid in enumerate(feature_ids)}
    documents = []
    for path in sorted(input_dir.glob("doc_features_rank*.npy")):
        documents.extend(np.load(path, allow_pickle=True).tolist())

    rows, cols = [], []
    for doc_id, feature_list in enumerate(documents):
        for fid in set(map(int, feature_list)):
            row_id = feature_to_row.get(fid)
            if row_id is not None:
                rows.append(row_id)
                cols.append(doc_id)
    incidence = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(feature_ids), len(documents)),
    ).tocsr()
    vectors = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(incidence)
    return documents, incidence, vectors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--labels", required=True, help="car_exact_labels.csv")
    ap.add_argument("--label", default="composite")
    ap.add_argument("--output", required=True)
    ap.add_argument("--coarse-k", type=int, default=3)
    ap.add_argument("--sub-k", type=int, default=2)
    args = ap.parse_args()

    input_dir = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    feature_ids = load_feature_ids(Path(args.labels), args.label)
    documents, incidence, vectors = build_vectors(input_dir, feature_ids)

    coarse_model = MiniBatchKMeans(
        n_clusters=args.coarse_k, random_state=0, n_init=3, batch_size=1024, max_iter=100
    )
    coarse = coarse_model.fit_predict(vectors)
    fine = np.full(len(feature_ids), -1, dtype=np.int32)
    summaries = []

    for coarse_id in range(args.coarse_k):
        members = np.flatnonzero(coarse == coarse_id)
        if len(members) < args.sub_k:
            fine[members] = 0
            summaries.append({"coarse_cluster": coarse_id, "subclusters": 1, "sizes": [int(len(members))]})
            continue
        sub_model = MiniBatchKMeans(
            n_clusters=args.sub_k, random_state=coarse_id, n_init=3, batch_size=512, max_iter=100
        )
        sub = sub_model.fit_predict(vectors[members])
        fine[members] = sub
        summaries.append(
            {
                "coarse_cluster": coarse_id,
                "subclusters": args.sub_k,
                "sizes": np.bincount(sub, minlength=args.sub_k).tolist(),
            }
        )

    with (output / "nested_cluster_labels.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "coarse_cluster", "sub_cluster", "nested_cluster", "activation_docs"])
        for i, fid in enumerate(feature_ids):
            writer.writerow(
                [
                    int(fid),
                    int(coarse[i]),
                    int(fine[i]),
                    f"coarse_{int(coarse[i])}_sub_{int(fine[i])}",
                    int(incidence.getrow(i).nnz),
                ]
            )

    (output / "nested_cluster_summary.json").write_text(
        json.dumps(
            {
                "method": "TF-IDF document activation patterns; coarse K-means followed by within-coarse K-means",
                "label": args.label,
                "documents": len(documents),
                "features": len(feature_ids),
                "coarse_k": args.coarse_k,
                "sub_k": args.sub_k,
                "summaries": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

