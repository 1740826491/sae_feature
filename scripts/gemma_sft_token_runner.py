#!/usr/bin/env python3
"""Eight-rank SFT token activation extraction with document alignment."""
import argparse,json,os,time
from pathlib import Path
import numpy as np,torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM,AutoTokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--sae',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--layer',type=int,default=27);p.add_argument('--max-tokens',type=int,default=1_000_000);p.add_argument('--seq-len',type=int,default=1024);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--topk',type=int,default=150);a=p.parse_args()
 rank=int(os.getenv('RANK',0));world=int(os.getenv('WORLD_SIZE',1));local=int(os.getenv('LOCAL_RANK',0));limit=(a.max_tokens+world-1)//world
 if world>1:dist.init_process_group('nccl')
 torch.cuda.set_device(local);dev=torch.device('cuda',local);out=Path(a.out);(out/'token_activations').mkdir(parents=True,exist_ok=True)
 files=sorted(Path(a.data).glob('*.txt'))
 if len(files)!=world:raise RuntimeError(f'expected {world} domain files, found {len(files)}')
 source=files[rank];domain=source.stem.split('_',1)[-1]
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map={'':dev},local_files_only=True).eval()
 raw=np.load(a.sae);sae={k:torch.from_numpy(v).to(dev,dtype=torch.bfloat16) for k,v in raw.items()};del raw
 cap=[];handle=model.model.layers[a.layer].register_forward_hook(lambda _m,_i,o:cap.append(o[0] if isinstance(o,tuple) else o));total=0;docs=0;part=0;nan_rows=0;start=time.time();chunk=4096
 pending=[];pending_ids=[]
 def process(texts,doc_ids):
  nonlocal total,part,nan_rows
  enc=tok(texts,add_special_tokens=True,truncation=True,max_length=a.seq_len,padding=True,return_tensors='pt');ids=enc['input_ids'].to(dev);mask=enc['attention_mask'].to(dev);cap.clear()
  with torch.inference_mode():model(input_ids=ids,attention_mask=mask,use_cache=False)
  h=cap[0].to(torch.bfloat16);bestv=torch.full((*h.shape[:2],a.topk),-float('inf'),device=dev);besti=torch.full((*h.shape[:2],a.topk),-1,dtype=torch.long,device=dev)
  with torch.inference_mode():
   for lo in range(0,sae['W_enc'].shape[1],chunk):
    hi=min(lo+chunk,sae['W_enc'].shape[1]);sc=torch.relu((h-sae['b_dec'])@sae['W_enc'][:,lo:hi]+sae['b_enc'][lo:hi]-sae['threshold'][lo:hi]);v,i=torch.topk(sc,min(a.topk,sc.shape[-1]),dim=-1);mv=torch.cat((bestv,v),-1);mi=torch.cat((besti,i+lo),-1);bestv,order=torch.topk(mv,a.topk,dim=-1);besti=torch.gather(mi,2,order)
  ip=[];vp=[];dp=[];pp=[]
  for row,doc_id in enumerate(doc_ids):
   valid=mask[row].bool();n=int(valid.sum());iv=besti[row][valid].cpu().numpy().astype(np.int32);vv=bestv[row][valid].float().cpu().numpy().astype(np.float32);ip.append(iv);vp.append(vv);dp.append(np.full(n,doc_id,np.int32));pp.append(np.arange(n,dtype=np.int32));nan_rows+=int(np.isnan(vv).all(1).sum())
  count=sum(len(x) for x in dp);np.savez_compressed(out/'token_activations'/f'part_rank{rank}_{part:06d}.npz',feature_ids=np.concatenate(ip),activation_values=np.concatenate(vp),document_id=np.concatenate(dp),position=np.concatenate(pp),domain_id=np.int16(rank),domain=np.asarray(domain),token_count=np.int64(count),topk=np.int32(a.topk));part+=1;total+=count
 try:
  with source.open(errors='ignore') as f:
   for line_no,line in enumerate(f):
    line=line.strip()
    if not line:continue
    try:text=json.loads(line)
    except json.JSONDecodeError:text=line
    if isinstance(text,dict):text=text.get('text','')
    if not isinstance(text,str) or not text.strip():continue
    pending.append(text.strip());pending_ids.append(line_no);docs+=1
    if len(pending)>=a.batch_size:process(pending,pending_ids);pending=[];pending_ids=[]
    if total>=limit:break
   if pending and total<limit:process(pending,pending_ids)
 finally:handle.remove()
 (out/f'progress_rank{rank}.json').write_text(json.dumps({'rank':rank,'world':world,'domain':domain,'tokens':total,'documents':docs,'parts':part,'all_nan_rows':nan_rows,'seconds':time.time()-start},indent=2))
if __name__=='__main__':main()
