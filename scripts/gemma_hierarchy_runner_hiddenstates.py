#!/usr/bin/env python3
import argparse,csv,json,os,time
from pathlib import Path
from collections import defaultdict
import numpy as np, torch
import torch.distributed as dist
from transformers import AutoTokenizer,AutoModelForCausalLM

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--sae',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--layer',type=int,default=34);p.add_argument('--max-tokens',type=int,default=500_000_000);p.add_argument('--seq-len',type=int,default=1024);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--topk',type=int,default=76);a=p.parse_args()
 rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0));local_limit=(a.max_tokens+world-1)//world
 if world>1: dist.init_process_group('nccl');torch.cuda.set_device(local)
 dev=torch.device('cuda',local);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':dev},local_files_only=True).eval()
 raw=np.load(a.sae);z={k:torch.from_numpy(v).to(dtype=torch.bfloat16) for k,v in raw.items()};del raw;freq=torch.zeros(z['b_enc'].numel(),dtype=torch.int64,device=dev);pairs=defaultdict(int);total=0;start=time.time();chunk=4096
 try:
  fs=sorted(Path(a.data).rglob('*.txt'));pending=[]
  def process_batch(texts):
   nonlocal total
   enc=tok(texts,add_special_tokens=False,truncation=True,max_length=a.seq_len,padding=True,return_tensors='pt');ids=enc['input_ids'].to(dev);mask=enc['attention_mask'].to(dev);total+=int(mask.sum().item())
   outputs=model(input_ids=ids,attention_mask=mask,use_cache=False,output_hidden_states=True)
   h=outputs.hidden_states[a.layer+1].float()
   h=h.to(torch.bfloat16);bdec=z['b_dec'].to(dev);bestv=torch.full((h.shape[0],h.shape[1],a.topk),-float('inf'),device=dev);besti=torch.full((h.shape[0],h.shape[1],a.topk),-1,dtype=torch.long,device=dev)
   for lo in range(0,z['W_enc'].shape[1],chunk):
    hi=min(lo+chunk,z['W_enc'].shape[1]);we=z['W_enc'][:,lo:hi].to(dev);be=z['b_enc'][lo:hi].to(dev);th=z['threshold'][lo:hi].to(dev);sc=torch.relu((h-bdec)@we+be-th);v,i=torch.topk(sc,min(a.topk,sc.shape[-1]),dim=-1);i=i+lo;mv=torch.cat((bestv,v),dim=-1);mi=torch.cat((besti,i),dim=-1);bestv,ord=torch.topk(mv,a.topk,dim=-1);besti=torch.gather(mi,2,ord);del we,be,th,sc,v,i,mv,mi
   for row in range(ids.shape[0]):
    valid=mask[row].bool();present=besti[row][valid][bestv[row][valid]>0].unique();freq[present]+=1;present=present.cpu().numpy()
    for i in range(len(present)):
     for j in range(i+1,len(present)):pairs[(int(present[i]),int(present[j]))]+=1
   if rank==0: print(json.dumps({'tokens':total,'batch_size':len(texts),'elapsed_s':time.time()-start}),flush=True)
  for fi,path in enumerate(fs):
   if fi%world!=rank:continue
   with path.open(errors='ignore') as f:
    for line in f:
     line=line.strip()
     if not line:continue
     try: text=json.loads(line)
     except json.JSONDecodeError: text=line
     if isinstance(text,dict):text=text.get('text','')
     if isinstance(text,str) and text.strip():pending.append(text.strip())
     if len(pending)>=a.batch_size:process_batch(pending);pending=[]
     if total>=local_limit:break
    if pending and total<local_limit:process_batch(pending)
    if total>=local_limit:break
 finally: pass
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);np.save(out/f'frequency_rank{rank}.npy',freq.cpu().numpy())
 with (out/f'coactivation_rank{rank}.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['a','b','coactivation']);w.writerows((i,j,c) for (i,j),c in pairs.items())
 (out/f'progress_rank{rank}.json').write_text(json.dumps({'tokens':total,'rank':rank,'world':world,'seconds':time.time()-start}))
if __name__=='__main__':main()
