#!/usr/bin/env python3
"""Build the Zhangxin/Hyper directional hierarchy from document-level stats.

The input is the existing per-rank frequency NPY and coactivation CSV files.
This deliberately does not use CAR Exact's one-pass candidate construction.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import networkx as nx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--min-feature-count', type=int, default=16)
    ap.add_argument('--min-pair-count', type=int, default=16)
    ap.add_argument('--min-parent-given-child', type=float, default=.80)
    ap.add_argument('--max-child-given-parent', type=float, default=.50)
    ap.add_argument('--min-frequency-ratio', type=float, default=2.0)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    freq = np.zeros(131072, dtype=np.int64)
    files = sorted(args.input.glob('frequency_rank*.npy'))
    for p in files: freq += np.load(p)
    pair = defaultdict(int)
    for p in sorted(args.input.glob('coactivation_rank*.csv')):
        with p.open() as f:
            rd = csv.DictReader(f)
            for r in rd:
                i,j,c = int(r['a']),int(r['b']),int(r['coactivation'])
                pair[(i,j)] += c
    g = nx.DiGraph()
    for (i,j), cij in pair.items():
        ci,cj = int(freq[i]),int(freq[j])
        if ci < args.min_feature_count or cj < args.min_feature_count or cij < args.min_pair_count: continue
        if ci >= cj: parent,child,pc,cc=i,j,ci,cj
        else: parent,child,pc,cc=j,i,cj,ci
        p_parent_child=cij/cc; p_child_parent=cij/pc; ratio=pc/max(cc,1)
        if p_parent_child >= args.min_parent_given_child and p_child_parent <= args.max_child_given_parent and ratio >= args.min_frequency_ratio:
            g.add_edge(parent,child,score=p_parent_child-p_child_parent)
    if not nx.is_directed_acyclic_graph(g):
        raise RuntimeError('directional graph is cyclic before reduction')
    red = nx.transitive_reduction(g)
    with (args.output/'hierarchy_edges.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['parent','child','parent_count','child_count','score'])
        for u,v in red.edges(): w.writerow([u,v,int(freq[u]),int(freq[v]),g[u][v].get('score',0.0)])
    roots=[n for n in red if red.in_degree(n)==0]
    depth={n:0 for n in roots}
    for n in nx.topological_sort(red):
        for c in red.successors(n): depth[c]=max(depth.get(c,0),depth[n]+1)
    with (args.output/'feature_depth.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['feature','frequency','depth'])
        for n in sorted(depth): w.writerow([n,int(freq[n]),depth[n]])
    json.dump({'input':str(args.input),'edges':red.number_of_edges(),'nodes':red.number_of_nodes(),'max_depth':max(depth.values(),default=0),'thresholds':vars(args)}, (args.output/'complete.json').open('w'), indent=2, default=str)
    print(json.dumps({'edges':red.number_of_edges(),'nodes':red.number_of_nodes(),'max_depth':max(depth.values(),default=0)}))
if __name__ == '__main__': main()
