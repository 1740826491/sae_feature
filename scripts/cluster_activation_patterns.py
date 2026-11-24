"""Cluster high-level SAE features by document activation patterns.

Input: doc_features_rank*.npy, where each document stores sparse SAE feature IDs.
The script builds feature x document incidence vectors, applies TF-IDF weighting,
compares candidate K values with cosine silhouette, and writes labels for the best K.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory containing doc_features_rank*.npy")
    ap.add_argument("--features", required=True, help="CSV with a composites column")
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", nargs="+", type=int, default=[4, 8, 12, 16])
    ap.add_argument("--sample-size", type=int, default=3000)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with Path(args.features).open() as f:
        row = next(csv.DictReader(f))
    feature_ids = np.array([int(x) for x in row["composites"].split(";")], dtype=np.int32)
    feature_to_row = {int(fid): i for i, fid in enumerate(feature_ids)}

    documents = []
    for path in sorted(inp.glob("doc_features_rank*.npy")):
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

    rng = np.random.default_rng(0)
    sample = rng.choice(len(feature_ids), size=min(args.sample_size, len(feature_ids)), replace=False)
    candidates = []
    best = None
    for k in args.k:
        model = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=1024, max_iter=100)
        labels = model.fit_predict(vectors)
        score = float(silhouette_score(vectors[sample], labels[sample], metric="cosine"))
        sizes = np.bincount(labels, minlength=k).tolist()
        candidates.append({"k": k, "silhouette_cosine": score, "sizes": sizes})
        if best is None or score > best["score"]:
            best = {"k": k, "score": score, "labels": labels}

    with (out / "cluster_labels.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "cluster", "activation_docs"])
        for i, fid in enumerate(feature_ids):
            writer.writerow([int(fid), int(best["labels"][i]), int(incidence.getrow(i).nnz)])

    with (out / "cluster_summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "silhouette_cosine", "sizes"])
        for item in candidates:
            writer.writerow([item["k"], item["silhouette_cosine"], ";".join(map(str, item["sizes"]))])

    (out / "complete.json").write_text(
        json.dumps(
            {
                "method": "TF-IDF document activation patterns + MiniBatchKMeans",
                "documents": len(documents),
                "features": len(feature_ids),
                "candidates": candidates,
                "selected_k": best["k"],
                "selected_silhouette_cosine": best["score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

