#!/usr/bin/env python3
import argparse,itertools,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--spec-root',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--representative',action='store_true');p.add_argument('--fixed-lr',type=float);p.add_argument('--common-lrs',nargs='+',type=float);p.add_argument('--fixed-routing-only',action='store_true');p.add_argument('--fixed-overlaps',nargs='+',default=['ideal','real_max']);p.add_argument('--dims',nargs='+',type=int,default=[64,128,256,358]);p.add_argument('--n-per-node',type=int,default=128);p.add_argument('--one-seed',action='store_true');a=p.parse_args()
 overlaps=['real_max'] if a.representative else (a.fixed_overlaps if (a.fixed_lr is not None or a.common_lrs) else ['ideal','real_max','random_assignment']);dims=[128] if a.representative else a.dims;seeds=[(7,0)] if (a.representative or a.one_seed) else [(7,0),(8,1),(9,2)]
 lrs=a.common_lrs if a.common_lrs else ([a.fixed_lr] if a.fixed_lr is not None else ([.25,.5,1,2,4,8,16] if a.representative else [.25,.3536,.5,.7071,1,1.4142,2,2.8284,4,5.6569,8,11.3137,16,22.6274,32,45.2548,64]))
 cfg=[]
 for ov,d,(ts,sd),lr in itertools.product(overlaps,dims,seeds,lrs):
  base={'spec':f'{a.spec_root}/{ov}','k':8,'n_per_node':a.n_per_node,'directions':'sae','canonical_duplicate_features':True,'target_seed':ts,'seed':sd,'steps':1500,'n_logs':70,'lr':lr}
  cfg.append({**base,'name':f'{ov}_d{d}_dense_lr{lr}_s{sd}','model':'dense','opt':'gd','mode':'population'})
  # Dense remains a full 3584->3584 map (rank=0).  Only MoE experts use the
  # processing bottleneck 3584->d->3584.  input_mode=level performs the
  # theory's ideal separation without changing representation dimension.
  tree={'model':'tree','rank':d,'input_mode':'level','per_expert_lr':True}
  cfg.append({**base,**tree,'name':f'{ov}_d{d}_treeperfect_lr{lr}_s{sd}','routing':'perfect','opt':'gd','mode':'population'})
  cfg.append({**base,**tree,'name':f'{ov}_d{d}_treesharedlr_lr{lr}_s{sd}','routing':'perfect','per_expert_lr':False,'opt':'gd','mode':'population'})
  cfg.append({**base,**tree,'name':f'{ov}_d{d}_treerandom_lr{lr}_s{sd}','routing':'independent','opt':'gd','mode':'population'})
  # Learned gates use the senior's sampled-Adam protocol; same data/spec/dimension.
  if not a.fixed_routing_only:
   for gate in ('hier','flat3'):
    cfg.append({**base,**tree,'name':f'{ov}_d{d}_{gate}_lr{lr}_s{sd}','routing':'learned','gate':gate,'gate_topk':3,'per_expert_lr':False,'opt':'adam','mode':'sampled','batch':512,'aux_coef':.01,'steps':3000})
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text('\n'.join(json.dumps(x) for x in cfg)+'\n');print(len(cfg))
if __name__=='__main__':main()
