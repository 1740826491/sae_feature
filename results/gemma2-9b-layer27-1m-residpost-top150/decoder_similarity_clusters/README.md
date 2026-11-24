# Decoder-similarity clusters

These labels cluster only the selected B2/T20/T21 feature pools by their SAE
decoder vectors.  Each decoder row is L2-normalized first; KMeans therefore
clusters directions in decoder space rather than activation frequency or CAR
co-occurrence.

## Groups

- `B2`: bottom constituent features whose relation label is `bottom_cluster=2`.
- `T20`: top features with `nested_top_cluster=bottom_2_top_0`.
- `T21`: top features with `nested_top_cluster=bottom_2_top_1`.

Each group contains `labels_k2.csv` and `labels_k4.csv`.  A row is
`feature,cluster,decoder_norm`.  The matching `centers_k*.npy` files contain
the normalized KMeans centers.  `summary.json` records cluster sizes, inertia,
mean cosine similarity to the assigned center, and the cosine matrix between
cluster centers.

This is an independent decoder-space grouping. It does not claim that a
decoder cluster is a CAR parent class; use the original CAR labels and edges
when a real bottom-to-top relation is required.
