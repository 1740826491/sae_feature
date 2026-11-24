# Gemma 2 9B SAE hierarchy

This directory contains the 50M-token hierarchy result for `google/gemma-2-9b` at layer 27.

## Model and SAE

- Model: `google/gemma-2-9b`
- SAE: `google/gemma-scope-9b-pt-res`
- SAE path: `layer_27/width_131k/average_l0_56/params.npz`
- Data: DCLM
- Tokens: 50M total, processed as two 25M shards on two 8-card instances

## Legacy CAR configuration

The graph was built with the legacy CAR procedure:

```text
minimum_count=50
containment=0.85
reverse=0.85
frequency_ratio=1.10
joint_containment=0.75
minimum_constituents=2
maximum_constituents=8
max_base_candidates=4096
```

## Files

- `car_exact_edges.csv`: parent-to-child CAR edges.
- `car_exact_labels.csv`: feature frequency, depth, and leaf/composite label.
- `car_exact_complete.json`: resolved run metadata and summary.

The model weights, SAE parameter matrix, and raw sparse activation shards are intentionally not included. Download the model and SAE from Hugging Face, accept the Gemma license, and use the configuration above.
