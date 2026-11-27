#!/usr/bin/env python3
"""Held-out feature concentration after SAE-guided document routing.

Consumes the completed document-routing experiment.  It never refits the SAE
or the router.  Random controls permute the already observed assignments and
therefore preserve expert load exactly.
"""
from __future__ import annotations

import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import csr_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--routing-results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument("--reliable-min-events", type=int, default=20)
    p.add_argument("--heatmap-features", type=int, default=120)
    p.add_argument("--representative-experts", type=int, default=8)
    p.add_argument("--representative-dim", type=int, default=32)
    p.add_argument("--representative-seed", type=int, default=42)
    return p.parse_args()


def load_manifest(root, routing):
    m = json.loads((routing / "manifest.json").read_text())
    cfg = m["config"]
    def resolve(x):
        x = Path(x)
        return x if x.is_absolute() else root / x
    return m, resolve(cfg["doc_features"]), resolve(cfg["feature_depth"])


def load_docs(doc_dir):
    arrays = sorted(doc_dir.glob("doc_features_rank*.npy"), key=lambda p: int(p.stem.split("rank")[-1]))
    docs = []
    for p in arrays:
        x = np.load(p, allow_pickle=True)
        docs.extend(np.unique(np.asarray(v, dtype=np.int32)) for v in x)
    return docs


def load_depths(path):
    by = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            by[int(r["depth"])].append(int(r["feature"]))
    return {d: np.asarray(sorted(v), dtype=np.int32) for d, v in by.items()}


def load_test_indices(path):
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["split"] == "test": out.append(int(r["doc_index"]))
    return np.asarray(out, dtype=np.int64)


def make_matrix(docs, test, features):
    pos = {int(f): j for j, f in enumerate(features)}
    rr, cc = [], []
    for row, ix in enumerate(test):
        for f in docs[int(ix)]:
            j = pos.get(int(f))
            if j is not None: rr.append(row); cc.append(j)
    return csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)), shape=(len(test), len(features)))


def load_eval_features(path):
    out = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["role"] == "evaluation": out[(int(r["split_seed"]), int(r["depth"]))].append(int(r["feature"]))
    return {k: np.asarray(sorted(set(v)), dtype=np.int32) for k, v in out.items()}


def load_assignments(path, test_pos):
    groups = defaultdict(lambda: np.full(len(test_pos), -1, dtype=np.int16))
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["assignment_variant"] != "hash_zero_all_documents": continue
            ix = int(r["doc_index"])
            j = test_pos.get(ix)
            if j is None: continue
            key = (int(r["split_seed"]), int(r["routing_depth"]), int(r["experts"]), int(r["routing_dim"]))
            groups[key][j] = int(r["expert"])
    return dict(groups)


def profile_counts(X, labels, k):
    rows = np.arange(len(labels)); A = csr_matrix((np.ones(len(rows)), (labels, rows)), shape=(k, len(rows)))
    return np.asarray((A @ X).toarray(), dtype=np.float64), np.bincount(labels, minlength=k).astype(np.float64)


def weighted_jaccard(freq, loads):
    vals, weights = [], []
    for i in range(len(freq)):
        for j in range(i + 1, len(freq)):
            mn = np.minimum(freq[i], freq[j]).sum(); mx = np.maximum(freq[i], freq[j]).sum()
            if mx: vals.append(mn / mx); weights.append(loads[i] * loads[j])
    return float(np.average(vals, weights=weights)) if vals and sum(weights) else math.nan


def metrics(counts, loads, n_docs):
    total = counts.sum(0); valid = total > 0
    q = np.zeros_like(counts); q[:, valid] = counts[:, valid] / total[valid]
    local = counts / np.maximum(loads[:, None], 1)
    global_freq = total / n_docs
    dom = q.argmax(0)
    idx = np.arange(counts.shape[1])
    purity = q[dom, idx]
    lift = local[dom, idx] / np.maximum(global_freq, 1e-12)
    eff = 1 / np.maximum((q * q).sum(0), 1e-12)
    entropy = -(np.where(q > 0, q * np.log(np.maximum(q, 1e-300)), 0)).sum(0) / math.log(counts.shape[0])
    return dict(total=total, q=q, local=local, global_freq=global_freq, dominant=dom,
                purity=purity, lift=lift, effective_experts=eff, entropy=entropy,
                weighted_jaccard=weighted_jaccard(local, loads))


def freq_bin(n):
    if n < 5: return "1-4"
    if n < 10: return "5-9"
    if n < 20: return "10-19"
    if n < 50: return "20-49"
    return "50+"


def quantiles(x):
    x = np.asarray(x, float)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75))}


def write_csv(path, rows):
    if not rows: return
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def choose_heatmap_features(m, limit, reliable):
    ids = np.flatnonzero(m["total"] >= reliable)
    if len(ids) <= limit: return ids
    order = ids[np.argsort(m["total"][ids])]
    half = limit // 2
    chosen = np.concatenate([order[:half], order[-(limit-half):]])
    return chosen[np.lexsort((m["total"][chosen], m["dominant"][chosen]))]


def heat_color(x):
    stops = [(0.0,(8,8,18)),(.25,(65,15,95)),(.5,(175,45,80)),(.75,(245,120,45)),(1.0,(255,245,180))]
    x = float(np.clip(x, 0, 1))
    for (a,ca),(b,cb) in zip(stops[:-1], stops[1:]):
        if x <= b:
            t=(x-a)/(b-a); return tuple(int(ca[i]*(1-t)+cb[i]*t) for i in range(3))
    return stops[-1][1]


def text(draw, xy, value, fill=(25,25,25)):
    draw.text(xy, value, fill=fill, font=ImageFont.load_default())


def plot_heatmap(out, depth, actual, random_m, loads, selected):
    k = len(loads)
    ideal = np.zeros((k, len(selected)))
    ideal[actual["dominant"][selected], np.arange(len(selected))] = 1
    mats = [random_m["q"][:, selected].T, actual["q"][:, selected].T, ideal.T]
    names = [f"Load-matched random\nWJ={random_m['weighted_jaccard']:.3f}",
             f"SAE routing\nWJ={actual['weighted_jaccard']:.3f}", "Ideal decoupling\nWJ=0"]
    cell_w, cell_h, gap, top, left = 22, 5, 45, 75, 45
    panel_w = k*cell_w; height = top + len(selected)*cell_h + 55
    width = left + 3*panel_w + 2*gap + 35
    img=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(img)
    text(draw,(left,12),f"Depth {depth}: feature concentration with incomplete decoupling")
    for pi,(z,name) in enumerate(zip(mats,names)):
        x0=left+pi*(panel_w+gap); text(draw,(x0,35),name.replace("\n","  "))
        for r in range(z.shape[0]):
            for c in range(z.shape[1]):
                draw.rectangle((x0+c*cell_w,top+r*cell_h,x0+(c+1)*cell_w-1,top+(r+1)*cell_h-1),fill=heat_color(z[r,c]))
        for c in range(k): text(draw,(x0+c*cell_w+7,height-42),str(c))
        text(draw,(x0,height-25),"Expert")
    text(draw,(5,top),"Held-out features")
    # Color key.
    x0=width-25
    for y in range(100): draw.line((x0,top+y,x0+12,top+y),fill=heat_color(1-y/99))
    text(draw,(x0-2,top+105),"P(e|f)")
    img.save(out / f"depth_{depth}_feature_expert_heatmap.png")


def plot_overview(out, depth_rows):
    depths = sorted(set(r["depth"] for r in depth_rows))
    W,H=1260,420; img=Image.new("RGB",(W,H),"white"); draw=ImageDraw.Draw(img)
    panels=[("median_lift","Dominant-expert frequency lift"),("median_purity","Dominant-expert attribution"),("weighted_jaccard","Weighted Jaccard overlap")]
    colors={"sae":(30,105,190),"random":(210,85,55)}
    for pi,(metric,title) in enumerate(panels):
        x0=35+pi*410; y0=55; pw=350; ph=290
        text(draw,(x0,y0-28),title); draw.rectangle((x0,y0,x0+pw,y0+ph),outline=(90,90,90))
        vals={m:[next(r[metric] for r in depth_rows if r["depth"]==d and r["method"]==m) for d in depths] for m in colors}
        allv=[v for z in vals.values() for v in z]; lo=min(0,min(allv)); hi=max(allv); hi=hi if hi>lo else lo+1
        for m,ys in vals.items():
            pts=[]
            for i,(d,v) in enumerate(zip(depths,ys)):
                x=x0+30+i*(pw-60)/max(1,len(depths)-1); y=y0+ph-25-(v-lo)/(hi-lo)*(ph-50); pts.append((x,y))
                draw.ellipse((x-4,y-4,x+4,y+4),fill=colors[m]); text(draw,(x-3,y0+ph+8),str(d))
            draw.line(pts,fill=colors[m],width=3); text(draw,(x0+220,y0+15+18*list(colors).index(m)),m,colors[m])
        text(draw,(x0+140,y0+ph+25),"Depth")
        text(draw,(x0,y0+ph+25),f"range {lo:.2f}-{hi:.2f}")
    img.save(out / "depth_concentration_overview.png")


def main():
    a = parse_args(); root = Path(a.root); routing = Path(a.routing_results); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest, doc_dir, depth_file = load_manifest(root, routing)
    docs = load_docs(doc_dir); all_depths = load_depths(depth_file)
    depths = {int(d): all_depths[int(d)] for d in manifest["depths"]}
    test = load_test_indices(routing / "document_split.csv")
    assert len(docs) == manifest["documents"] and len(test) > 0
    matrices = {d: make_matrix(docs, test, fs) for d, fs in depths.items()}
    eval_features = load_eval_features(routing / "feature_splits.csv")
    test_pos = {int(x): i for i, x in enumerate(test)}
    assignments = load_assignments(routing / "assignments.csv", test_pos)
    feature_rows, config_rows = [], []
    representative = {}
    rng = np.random.default_rng(20261127)
    for key, labels in sorted(assignments.items()):
        seed, rd, k, dim = key
        if rd not in matrices or (seed, rd) not in eval_features: continue
        assert np.all(labels >= 0)
        all_features = depths[rd]; pos = {int(f): j for j, f in enumerate(all_features)}
        chosen_f = np.asarray([f for f in eval_features[(seed, rd)] if int(f) in pos], dtype=np.int32)
        cols = np.asarray([pos[int(f)] for f in chosen_f]); X = matrices[rd][:, cols]
        counts, loads = profile_counts(X, labels, k); actual = metrics(counts, loads, len(test))
        random_metrics = []
        for _ in range(a.random_repeats):
            perm = rng.permutation(labels); rc, rl = profile_counts(X, perm, k); random_metrics.append(metrics(rc, rl, len(test)))
        random_mean = {name: np.mean([m[name] for m in random_metrics], axis=0)
                       for name in ["q", "purity", "lift", "effective_experts", "entropy", "local"]}
        random_mean["total"] = actual["total"]; random_mean["dominant"] = random_mean["q"].argmax(0)
        random_mean["weighted_jaccard"] = float(np.mean([m["weighted_jaccard"] for m in random_metrics]))
        reliable = actual["total"] >= a.reliable_min_events
        tail_cut = np.quantile(actual["total"][reliable], .25) if reliable.any() else math.nan
        tail = reliable & (actual["total"] <= tail_cut)
        for method, m in [("sae", actual), ("random", random_mean)]:
            for subset, mask in [("reliable", reliable), ("long_tail_q1", tail)]:
                config_rows.append({"split_seed": seed, "depth": rd, "experts": k, "routing_dim": dim,
                    "method": method, "subset": subset, "features": int(mask.sum()),
                    "median_lift": float(np.median(m["lift"][mask])) if mask.any() else math.nan,
                    "median_purity": float(np.median(m["purity"][mask])) if mask.any() else math.nan,
                    "median_effective_experts": float(np.median(m["effective_experts"][mask])) if mask.any() else math.nan,
                    "median_entropy": float(np.median(m["entropy"][mask])) if mask.any() else math.nan,
                    "weighted_jaccard": float(m["weighted_jaccard"])})
        for j, fid in enumerate(chosen_f):
            feature_rows.append({"split_seed": seed, "depth": rd, "experts": k, "routing_dim": dim,
                "feature": int(fid), "test_events": int(actual["total"][j]), "frequency_bin": freq_bin(actual["total"][j]),
                "sae_purity": float(actual["purity"][j]), "random_purity": float(random_mean["purity"][j]),
                "sae_lift": float(actual["lift"][j]), "random_lift": float(random_mean["lift"][j]),
                "sae_effective_experts": float(actual["effective_experts"][j]), "random_effective_experts": float(random_mean["effective_experts"][j])})
        if k == a.representative_experts and dim == a.representative_dim and seed == a.representative_seed:
            representative[rd] = (actual, random_mean, loads, chosen_f)
    write_csv(out / "feature_metrics.csv", feature_rows); write_csv(out / "config_summary.csv", config_rows)
    depth_rows = []
    for d in sorted(depths):
        for method in ["sae", "random"]:
            rows = [r for r in config_rows if r["depth"] == d and r["method"] == method and r["subset"] == "long_tail_q1"]
            depth_rows.append({"depth": d, "method": method, "configs": len(rows),
                "median_lift": float(np.median([r["median_lift"] for r in rows])),
                "median_purity": float(np.median([r["median_purity"] for r in rows])),
                "median_effective_experts": float(np.median([r["median_effective_experts"] for r in rows])),
                "median_entropy": float(np.median([r["median_entropy"] for r in rows])),
                "weighted_jaccard": float(np.median([r["weighted_jaccard"] for r in rows]))})
    write_csv(out / "depth_summary.csv", depth_rows)
    for d, (actual, random_m, loads, fids) in representative.items():
        selected = choose_heatmap_features(actual, a.heatmap_features, a.reliable_min_events)
        plot_heatmap(out, d, actual, random_m, loads, selected)
        np.savez_compressed(out / f"depth_{d}_representative_heatmap.npz", feature_ids=fids[selected],
            sae_q=actual["q"][:, selected], random_q=random_m["q"][:, selected],
            test_events=actual["total"][selected], loads=loads)
    plot_overview(out, depth_rows)
    # Frequency-bin summary uses every configuration but keeps finite-sample bins explicit.
    bin_rows = []
    for d in sorted(depths):
        for b in ["1-4", "5-9", "10-19", "20-49", "50+"]:
            z = [r for r in feature_rows if r["depth"] == d and r["frequency_bin"] == b]
            if not z: continue
            bin_rows.append({"depth": d, "frequency_bin": b, "observations": len(z),
                "median_sae_lift": float(np.median([r["sae_lift"] for r in z])),
                "median_random_lift": float(np.median([r["random_lift"] for r in z])),
                "median_sae_purity": float(np.median([r["sae_purity"] for r in z])),
                "median_random_purity": float(np.median([r["random_purity"] for r in z]))})
    write_csv(out / "frequency_bin_summary.csv", bin_rows)
    result = {"status": "complete", "source_manifest": str(routing / "manifest.json"),
        "documents": len(docs), "test_documents": len(test), "depths": sorted(depths),
        "assignment_configs": len(assignments), "analyzed_configs": len({(r['split_seed'],r['depth'],r['experts'],r['routing_dim']) for r in config_rows}),
        "random_repeats": a.random_repeats, "reliable_min_test_events": a.reliable_min_events,
        "representative": {"experts": a.representative_experts, "routing_dim": a.representative_dim, "split_seed": a.representative_seed},
        "definitions": {"purity": "max_e P(expert=e | feature activation)",
            "local_frequency_lift": "max_e P(feature active | expert=e) / P(feature active)",
            "random": "assignment-label permutation preserving exact expert loads",
            "long_tail": "bottom test-frequency quartile among features with at least reliable_min_test_events"}}
    (out / "manifest.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
