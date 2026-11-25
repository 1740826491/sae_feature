"""Fast empirical audit of the V5 quantities on the Square Output setup."""
import argparse, csv, json
from pathlib import Path
import numpy as np

D = 3584
PI = np.array([.5, .5, .25, .25, .25, .25], dtype=float)
GROUPS = ('B2_0','B2_1','T20_0','T20_1','T21_0','T21_1')

def load_groups(label_root):
    groups=[]; feature_to_row={}; features=[]
    for name,k in (('B2',0),('B2',1),('T20',2),('T20',3),('T21',4),('T21',5)):
        path=Path(label_root)/name/f'labels_k2.csv'
        rows=list(csv.DictReader(path.open()))
        ids=[int(r['feature']) for r in rows if int(r['cluster'])==k%2]
        groups.append(np.asarray(ids,dtype=np.int64)); features.extend(ids)
    return groups

def capacity(X):
    X=np.asarray(X,dtype=np.float64); X=X-X.mean(0,keepdims=True)
    gram=X@X.T
    tr=np.trace(gram)
    tr2=np.square(gram).sum()
    eig=np.linalg.eigvalsh(gram)
    return {'capacity':float(tr*tr/tr2), 'lambda_max_unnormalized':float(eig[-1])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sae',required=True); ap.add_argument('--labels',required=True)
    ap.add_argument('--sample',required=True); ap.add_argument('--out',required=True); ap.add_argument('--max-per-group',type=int,default=512)
    a=ap.parse_args()
    with np.load(a.sae) as z: W=z['W_dec'].astype('float32')
    W/=np.linalg.norm(W,axis=1,keepdims=True).clip(min=1e-12)
    groups=load_groups(a.labels)
    rng=np.random.default_rng(8102)
    sampled=[g if len(g)<=a.max_per_group else rng.choice(g,a.max_per_group,replace=False) for g in groups]
    overlap=np.zeros((6,6)); counts=np.zeros((6,6),dtype=int)
    for i in range(6):
        A=W[sampled[i]]
        for j in range(i,6):
            B=W[sampled[j]]
            C=np.square(A@B.T)
            if i==j:
                C=C[~np.eye(len(C),dtype=bool)].reshape(len(C),len(C)-1)
            overlap[i,j]=overlap[j,i]=float(C.mean())
            counts[i,j]=counts[j,i]=C.size
    s=13/12
    dense_sir=[]; tree_sir=[]; per_group={}
    for i in range(6):
        A=W[sampled[i]]
        leaks=np.zeros((len(A),6))
        for j in range(6):
            B=W[sampled[j]]; C=np.square(A@B.T)
            if i==j: leaks[:,j]=(C.sum(1)-1)/(len(B)-1)
            else: leaks[:,j]=C.mean(1)
        I_dense=(leaks*(PI*s)).sum(1)-PI[i]*s*0 # self was handled above
        I_tree=leaks[:,i]*PI[i]*s
        p=PI[i]/len(groups[i])
        dense_sir.extend((p*s/I_dense).tolist()); tree_sir.extend((p*s/I_tree).tolist())
        per_group[GROUPS[i]]={'n_features':int(len(groups[i])),'pi':float(PI[i]),'p_feature':float(p),
                              'mean_dense_sir':float(np.mean(p*s/I_dense)),'mean_tree_sir':float(np.mean(p*s/I_tree))}
    data=np.load(a.sample)
    ids=data['feature_ids']; amp=data['strength']
    all_ids=np.concatenate(groups); row={int(x):int(x) for x in all_ids}
    comps=[]; blocks=[]
    for n in range(len(ids)):
        block=np.zeros((6,D),dtype=np.float32)
        for q in range(2):
            fid=int(ids[n,q]); g=next(i for i,x in enumerate(groups) if fid in set(x.tolist()))
            block[g]+=W[fid]*amp[n,q]
        blocks.append(block); comps.append(block.sum(0))
    X=np.asarray(comps); Xtree=np.asarray([b.reshape(-1) for b in blocks])
    result={'groups':GROUPS,'group_sizes':[int(len(g)) for g in groups],
            'overlap_mean_cos2':overlap.tolist(),'random_cos2_baseline':1/D,
            'sir_mean_dense':float(np.mean(dense_sir)),'sir_mean_tree':float(np.mean(tree_sir)),
            'per_group':per_group,'capacity_dense':capacity(X),'capacity_tree_blockdiag':capacity(Xtree),
            'lambda_leaf_expected':float((.25/np.mean([len(groups[i]) for i in (2,3,4,5)]))*s),
            'matrix_flops_forward_dense':D*D,'matrix_flops_forward_specified_rank300':4*D*300,
            'dense_over_specified_flops_ratio':D*D/(4*D*300)}
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
