#!/usr/bin/env python3
"""Test depth-conditioned, domain-group nesting on document feature sets.

This is deliberately a group-level test.  It does not require the support of
one individual deep feature to be contained in one individual shallow feature.
Groups are learned on train documents, thresholds are fixed on validation
documents, and all reported containment numbers are computed on held-out test
documents.  Frequency-matched random groups provide the null distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--doc-features", required=True)
    p.add_argument("--domain-files", required=True)
    p.add_argument("--feature-depth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--group-sizes", type=int, nargs="+", default=[32, 128, 512])
    p.add_argument("--depths", type=int, nargs="+", default=None,
                   help="zero-based hierarchy depths to test; e.g. 0 1 2 3 for four levels")
    p.add_argument("--min-train-docs", type=int, default=5)
    p.add_argument("--random-repeats", type=int, default=100)
    p.add_argument("--bootstrap-repeats", type=int, default=500)
    p.add_argument("--min-test-support", type=int, default=30)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20261126)
    return p.parse_args()


def load_depth(path: Path) -> tuple[dict[int, int], dict[int, np.ndarray]]:
    feature_depth: dict[int, int] = {}
    by_depth: dict[int, list[int]] = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            feature = int(row["feature"])
            depth = int(row["depth"])
            feature_depth[feature] = depth
            by_depth[depth].append(feature)
    return feature_depth, {d: np.asarray(sorted(x), dtype=np.int32) for d, x in by_depth.items()}


def domain_names(domain_dir: Path) -> list[str]:
    files = sorted(domain_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"no domain txt files in {domain_dir}")
    return [p.stem.split("_", 1)[-1] for p in files]


def load_documents(doc_dir: Path, names: list[str]) -> tuple[list[np.ndarray], np.ndarray]:
    paths = sorted(doc_dir.glob("doc_features_rank*.npy"), key=lambda p: int(p.stem.split("rank")[-1]))
    if len(paths) != len(names):
        raise ValueError(f"found {len(paths)} rank arrays but {len(names)} domain files")
    docs: list[np.ndarray] = []
    labels: list[int] = []
    for label, path in enumerate(paths):
        rows = np.load(path, allow_pickle=True)
        docs.extend(np.unique(np.asarray(x, dtype=np.int32)) for x in rows)
        labels.extend([label] * len(rows))
    return docs, np.asarray(labels, dtype=np.int16)


def stratified_split(labels: np.ndarray, train_frac: float, val_frac: float, rng: np.random.Generator):
    split = np.empty(len(labels), dtype=np.int8)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_frac))
        n_val = int(round(len(idx) * val_frac))
        split[idx[:n_train]] = 0
        split[idx[n_train:n_train + n_val]] = 1
        split[idx[n_train + n_val:]] = 2
    return split


def doc_depth_sets(docs: list[np.ndarray], feature_depth: dict[int, int], depths: list[int]):
    result = {d: [] for d in depths}
    for doc in docs:
        buckets = {d: [] for d in depths}
        for f in doc:
            d = feature_depth.get(int(f))
            if d in buckets:
                buckets[d].append(int(f))
        for d in depths:
            result[d].append(np.asarray(buckets[d], dtype=np.int32))
    return result


def train_counts(depth_docs, labels, split, features_by_depth, n_domains):
    counts = {}
    train_idx = np.flatnonzero(split == 0)
    for depth, features in features_by_depth.items():
        pos = {int(f): i for i, f in enumerate(features)}
        c = np.zeros((n_domains, len(features)), dtype=np.int32)
        for i in train_idx:
            if len(depth_docs[depth][i]):
                cols = [pos[int(f)] for f in depth_docs[depth][i] if int(f) in pos]
                c[int(labels[i]), cols] += 1
        counts[depth] = c
    return counts


def build_group(features, counts, domain, group_size, min_docs):
    total = counts.sum(axis=0).astype(np.float64)
    own = counts[domain].astype(np.float64)
    other = total - own
    # Smoothed log rate ratio.  Equal weighting of domains prevents large
    # domains from winning merely because they contain more documents.
    own_rate = (own + 0.5) / (own.sum() + 0.5 * len(own))
    other_rate = (other + 0.5) / (other.sum() + 0.5 * len(other))
    score = np.log(own_rate / other_rate)
    eligible = np.flatnonzero((total >= min_docs) & (score > 0))
    order = eligible[np.lexsort((-total[eligible], -score[eligible]))]
    chosen = order[:group_size]
    return features[chosen], score[chosen], total[chosen], score, total


def build_depth_matrix(docs_at_depth: list[np.ndarray], features: np.ndarray):
    pos = {int(f): i for i, f in enumerate(features)}
    rows, cols = [], []
    for row, doc in enumerate(docs_at_depth):
        for f in doc:
            col = pos.get(int(f))
            if col is not None:
                rows.append(row)
                cols.append(col)
    data = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(docs_at_depth), len(features)))
    denom = np.maximum(1, np.asarray(matrix.sum(axis=1)).ravel())
    return matrix, denom, pos


def support_scores(matrix, denom, feature_pos, group: np.ndarray) -> np.ndarray:
    cols = [feature_pos[int(f)] for f in group if int(f) in feature_pos]
    if not cols:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    overlap = np.asarray(matrix[:, cols].sum(axis=1)).ravel()
    return (overlap / denom).astype(np.float32)


def choose_threshold(scores, labels, domain, val_mask):
    x = scores[val_mask]
    y = labels[val_mask] == domain
    candidates = np.unique(x)
    if len(candidates) > 256:
        candidates = np.quantile(candidates, np.linspace(0, 1, 256))
    best = None
    for threshold in candidates:
        pred = x >= threshold
        tpr = pred[y].mean() if y.any() else 0.0
        fpr = pred[~y].mean() if (~y).any() else 0.0
        key = (tpr - fpr, tpr, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    threshold = best[1] if best else math.inf
    # A zero threshold makes every document positive, which is uninformative.
    if threshold <= 0:
        positive = candidates[candidates > 0]
        threshold = float(positive.min()) if len(positive) else math.inf
    return threshold


def containment(shallow, deep, test_mask):
    s = shallow[test_mask]
    d = deep[test_mask]
    n_deep = int(d.sum())
    n_shallow = int(s.sum())
    both = int((s & d).sum())
    return {
        "test_docs": int(test_mask.sum()),
        "deep_support_docs": n_deep,
        "shallow_support_docs": n_shallow,
        "intersection_docs": both,
        "forward_containment": both / n_deep if n_deep else float("nan"),
        "reverse_containment": both / n_shallow if n_shallow else float("nan"),
        "violation_rate": (n_deep - both) / n_deep if n_deep else float("nan"),
    }


def bootstrap_forward(shallow, deep, test_indices, repeats, rng):
    values = []
    s = shallow[test_indices]
    d = deep[test_indices]
    for _ in range(repeats):
        take = rng.integers(0, len(test_indices), size=len(test_indices))
        denom = d[take].sum()
        if denom:
            values.append(float((s[take] & d[take]).sum() / denom))
    if not values:
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))


def frequency_matched_group(features, total, target_total, rng):
    # Match each requested feature within a log2-frequency bin, without replacement.
    bins = np.floor(np.log2(total + 1)).astype(int)
    target_bins = np.floor(np.log2(target_total + 1)).astype(int)
    chosen = []
    used = set()
    for b in target_bins:
        pool = np.flatnonzero(bins == b)
        pool = [int(i) for i in pool if int(i) not in used]
        if not pool:
            pool = [int(i) for i in range(len(features)) if int(i) not in used]
        j = int(rng.choice(pool))
        used.add(j)
        chosen.append(int(features[j]))
    return np.asarray(chosen, dtype=np.int32)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    a = parse_args()
    if a.train_frac + a.val_frac >= 1:
        raise ValueError("train-frac + val-frac must be < 1")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    names = domain_names(Path(a.domain_files))
    feature_depth, features_by_depth = load_depth(Path(a.feature_depth))
    docs, labels = load_documents(Path(a.doc_features), names)
    depths = sorted(features_by_depth)
    if a.depths is not None:
        missing = sorted(set(a.depths) - set(depths))
        if missing:
            raise ValueError(f"requested depths absent from feature_depth.csv: {missing}")
        depths = sorted(set(a.depths))
        features_by_depth = {d: features_by_depth[d] for d in depths}
    split = stratified_split(labels, a.train_frac, a.val_frac, rng)
    depth_docs = doc_depth_sets(docs, feature_depth, depths)
    depth_matrix = {d: build_depth_matrix(depth_docs[d], features_by_depth[d]) for d in depths}
    counts = train_counts(depth_docs, labels, split, features_by_depth, len(names))
    val_mask = split == 1
    test_mask = split == 2
    test_indices = np.flatnonzero(test_mask)

    groups = {}
    group_rows = []
    score_cache = {}
    activation_cache = {}
    metadata = {}
    for size in a.group_sizes:
        for domain, name in enumerate(names):
            for depth in depths:
                group, enrichment, frequency, all_score, all_total = build_group(
                    features_by_depth[depth], counts[depth], domain, size, a.min_train_docs
                )
                if len(group) < 2:
                    continue
                key = (size, domain, depth)
                groups[key] = group
                scores = support_scores(*depth_matrix[depth], group)
                threshold = choose_threshold(scores, labels, domain, val_mask)
                score_cache[key] = scores
                activation_cache[key] = scores >= threshold
                metadata[key] = (threshold, all_score, all_total, frequency)
                for f, e, freq in zip(group, enrichment, frequency):
                    group_rows.append({"group_size_target": size, "domain": name, "domain_id": domain,
                                       "depth": depth, "feature": int(f), "train_log_enrichment": float(e),
                                       "train_activation_docs": int(freq)})

    rows = []
    null_rows = []
    random_activation_cache = {}
    for key, target_group in groups.items():
        size, domain, depth = key
        _, _, all_total, target_frequency = metadata[key]
        random_activation_cache[key] = []
        for _ in range(a.random_repeats):
            random_group = frequency_matched_group(
                features_by_depth[depth], all_total, target_frequency, rng
            )
            random_score = support_scores(*depth_matrix[depth], random_group)
            random_threshold = choose_threshold(random_score, labels, domain, val_mask)
            random_activation_cache[key].append(random_score >= random_threshold)

    for size in a.group_sizes:
        for domain, name in enumerate(names):
            available = [d for d in depths if (size, domain, d) in groups]
            for shallow in available:
                for deep in available:
                    if shallow >= deep:
                        continue
                    skey, dkey = (size, domain, shallow), (size, domain, deep)
                    stats = containment(activation_cache[skey], activation_cache[dkey], test_mask)
                    lo, hi = bootstrap_forward(activation_cache[skey], activation_cache[dkey],
                                               test_indices, a.bootstrap_repeats, rng)
                    row = {"group_size_target": size, "domain": name, "domain_id": domain,
                           "shallow_depth": shallow, "deep_depth": deep,
                           "shallow_group_features": len(groups[skey]),
                           "deep_group_features": len(groups[dkey]),
                           "shallow_threshold": metadata[skey][0], "deep_threshold": metadata[dkey][0],
                           **stats, "forward_ci_low": lo, "forward_ci_high": hi}

                    null_values = []
                    for repeat in range(a.random_repeats):
                        null = containment(random_activation_cache[skey][repeat],
                                           random_activation_cache[dkey][repeat],
                                           test_mask)["forward_containment"]
                        if not math.isnan(null):
                            null_values.append(null)
                            null_rows.append({"group_size_target": size, "domain": name,
                                              "shallow_depth": shallow, "deep_depth": deep,
                                              "repeat": repeat, "forward_containment": null})
                    if null_values:
                        row["random_forward_mean"] = float(np.mean(null_values))
                        row["random_forward_std"] = float(np.std(null_values))
                        row["forward_excess_over_random"] = stats["forward_containment"] - row["random_forward_mean"]
                        row["random_p_ge_observed"] = (1 + sum(v >= stats["forward_containment"] for v in null_values)) / (1 + len(null_values))
                    else:
                        row.update(random_forward_mean=float("nan"), random_forward_std=float("nan"),
                                   forward_excess_over_random=float("nan"), random_p_ge_observed=float("nan"))
                    rows.append(row)

    write_csv(out / "group_members.csv", group_rows)
    write_csv(out / "pair_metrics.csv", rows)
    write_csv(out / "random_null.csv", null_rows)
    split_rows = [{"doc_index": i, "domain": names[int(labels[i])],
                   "split": ["train", "validation", "test"][int(split[i])]}
                  for i in range(len(labels))]
    write_csv(out / "document_split.csv", split_rows)

    valid = [r for r in rows if not math.isnan(r["forward_excess_over_random"])
             and r["deep_support_docs"] >= a.min_test_support]
    summary = {
        "claim": "domain-by-depth feature groups have deeper-to-shallower support containment beyond matched random groups",
        "unit": "document-level binary SAE feature presence",
        "documents": len(docs),
        "domains": names,
        "depths": depths,
        "group_sizes": a.group_sizes,
        "pair_tests": len(rows),
        "inferential_pair_tests": len(valid),
        "minimum_deep_test_support": a.min_test_support,
        "mean_forward_containment": float(np.mean([r["forward_containment"] for r in valid])) if valid else None,
        "mean_random_forward_containment": float(np.mean([r["random_forward_mean"] for r in valid])) if valid else None,
        "mean_excess_over_random": float(np.mean([r["forward_excess_over_random"] for r in valid])) if valid else None,
        "fraction_pairs_above_random_p05": float(np.mean([r["random_p_ge_observed"] <= 0.05 for r in valid])) if valid else None,
        "warning": "This tests group support nesting, not an individual-feature tree and not token-level activation magnitude.",
        "config": vars(a),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    (out / "complete.json").write_text(json.dumps({"status": "complete", "pair_tests": len(rows)}, indent=2))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
