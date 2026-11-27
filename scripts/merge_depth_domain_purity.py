#!/usr/bin/env python3
import csv, json, sys
from collections import defaultdict
from pathlib import Path

base = Path(sys.argv[1])
out = base / "depth_domain_purity"
out.mkdir(exist_ok=True)
depth = {}
with (base / "feature_depth.csv").open() as f:
    for r in csv.DictReader(f):
        depth[int(r["feature"])] = int(r["depth"])
purity = base.parent / "sft_domain_8way_gemma9b_1125" / "document_domain_purity.csv"
rows = []
with purity.open() as f:
    for r in csv.DictReader(f):
        fid = int(r["feature"])
        if fid in depth:
            r["depth"] = str(depth[fid])
            rows.append(r)
fields = ["feature", "frequency", "depth", "activation_docs", "domain_entropy", "specificity", "dominant_domain", "dominant_prob"]
with (out / "feature_depth_purity.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
groups = defaultdict(list)
for r in rows:
    groups[int(r["depth"])].append(float(r["specificity"]))
with (out / "depth_summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["depth", "features", "mean_specificity", "median_specificity", "min", "max"])
    for d in sorted(groups):
        x = sorted(groups[d])
        n = len(x)
        w.writerow([d, n, sum(x) / n, x[n // 2], x[0], x[-1]])
(out / "complete.json").write_text(json.dumps({"features_with_both": len(rows), "depths": sorted(groups)}, indent=2))
print(json.dumps({"features_with_both": len(rows), "depths": sorted(groups)}))
