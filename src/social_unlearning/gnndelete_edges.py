"""GNNDelete node-embedding-loss adaptation to ODSR joint user deletion.
Executes the official DeletionLayer. Backbone parameters stay frozen.
"""
import ast,copy,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def deletion_class(source):
 tree=ast.parse((source/'framework/models/deletion.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeletionLayer');ns={'torch':torch,'nn':nn};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source/'framework/models/deletion.py'),'exec'),ns);return ns['DeletionLayer']


def neighborhoods(raw,deleted,layers):
 # Heterogeneous union, with user-item endpoints offset into the item block.
 edges=np.concatenate([np.column_stack([raw.train[:,0],raw.U+raw.train[:,1]]),raw.social]);active=np.zeros(raw.U+raw.I,bool);removed=np.isin(edges[:,0],deleted)|np.isin(edges[:,1],deleted);forget=edges[removed];active[np.unique(forget)]=True;masks=[]
 for _ in range(layers):
  touch=active[edges[:,0]]|active[edges[:,1]];active=active.copy();active[np.unique(edges[touch])]=True;masks.append(active.copy())
 return masks,forget,edges


class GNNDeleteODSR(nn.Module):
 def __init__(self,body,masks,source,operators=None):
  super().__init__();self.body=body;self.device=body.device;self.user_num=body.user_num;self.item_num=body.item_num
  for p in body.parameters():p.requires_grad_(False)
  layer=deletion_class(source);self.operators=operators if operators is not None else nn.ModuleList([layer(body.emb_size,None).to(self.device) for _ in body.layers])
  for k,mask in enumerate(masks):self.register_buffer('mask_'+str(k),torch.as_tensor(mask,device=self.device,dtype=torch.bool))
 def forward(self,current_ssl_mode=None,return_layers=False,apply=True):
  # Match ODSR's evaluation forward, adding shared masked operators after each layer.
  self.body.eval();b=self.body;hu=b.embedding_dict['user_emb'];hi=b.embedding_dict['item_emb'];su=hu.clone();ul=[hu];il=[hi];states=[]
  for k,layer in enumerate(b.layers):
   side=torch.sparse.mm(b.sparse_norm_adj,torch.cat([hu,hi]));_,inew=torch.split(side,[self.user_num,self.item_num]);unew,clean,_=layer(hi,su,b.adj_ui,b.adj_uu,current_ssl_mode,b.eps)
   if apply:
    mask=getattr(self,'mask_'+str(k));joined=self.operators[k](torch.cat([unew,inew]),mask);unew,inew=torch.split(joined,[self.user_num,self.item_num]);clean=self.operators[k](clean,mask[:self.user_num])
   hu,hi,su=unew,inew,clean;ul.append(hu);il.append(hi);states.append(torch.cat([hu,hi]))
  if return_layers:return states
  w=b.alphas.softmax(0);return (torch.stack(ul,1)*w.view(1,-1,1)).sum(1),(torch.stack(il,1)*w.view(1,-1,1)).sum(1)


def train_deletion(original,raw,data,cfg,directory,device,seed,provenance,max_steps=1500):
 from run_unlearning import ROOT,compact,evaluate,atomic_json
 source=ROOT/'external/baseline_sources/GNNDelete';masks,forget,edges=neighborhoods(raw,data.deleted_users,original.n_layers);selection=np.concatenate([data.keep,np.arange(raw.U,raw.U+raw.I)]);model=GNNDeleteODSR(compact(original,data,cfg,device),[m[selection] for m in masks],source)
 directory.mkdir(parents=True,exist_ok=True)
 if (directory/'deployment.pt').exists():
  ck=torch.load(directory/'deployment.pt',map_location=device,weights_only=False)
  if ck['provenance']!=provenance:raise ValueError('GNNDelete checkpoint provenance mismatch')
  model.load_state_dict(ck['model']);return model,ck['training']
 # Original-graph witnesses are request-side only; shared operators are deployed on edge-deleted graph with persistent rows.
 witness=GNNDeleteODSR(copy.deepcopy(original),masks,source,model.operators)
 with torch.no_grad():teacher=witness(return_layers=True,apply=False);retain_teacher=model(return_layers=True,apply=False)
 rng=np.random.default_rng(seed);sample=forget[rng.choice(len(forget),min(4096,len(forget)),replace=False)];neg=[];occupied=set(map(tuple,edges));occupied.update((int(v),int(u)) for u,v in edges)
 for u,v in sample:
  while True:
   x=int(rng.integers(raw.U));y=int(rng.integers(raw.U,raw.U+raw.I)) if v>=raw.U else int(rng.integers(raw.U))
   if x!=y and (x,y) not in occupied:neg.append((x,y));break
 fp=torch.as_tensor(sample,device=device).T;npair=torch.as_tensor(np.array(neg),device=device).T
 nonend=np.ones(raw.U+raw.I,bool);nonend[np.unique(forget)]=False;ni_masks=[torch.as_tensor((m&nonend)[selection],device=device) for m in masks]
 optimizer=torch.optim.Adam(model.operators.parameters(),lr=.001);best=-float('inf');bad=0;history=[];start=time.perf_counter()
 for step in range(1,max_steps+1):
  optimizer.zero_grad(set_to_none=True);ws=witness(return_layers=True);rs=model(return_layers=True);dec=sum(F.mse_loss(torch.cat([z[fp[0]],z[fp[1]]]),torch.cat([t[npair[0]],t[npair[1]]])) for z,t in zip(ws,teacher));ni=sum(F.mse_loss(z[m],t[m]) if m.any() else z.sum()*0 for z,t,m in zip(rs,retain_teacher,ni_masks));loss=.5*dec+.5*ni
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite GNNDelete loss')
  loss.backward();optimizer.step()
  if step%25==0 or step==max_steps:
   metrics,_=evaluate(model,data,data.val,device);improved=metrics['recall@20']>best
   if improved:best=metrics['recall@20'];best_state=copy.deepcopy(model.state_dict());best_step=step;bad=0
   else:bad+=1
   row=dict(step=step,dec=float(dec.detach()),ni=float(ni.detach()),validation=metrics);history.append(row);atomic_json(directory/'history.json',history);print(row,flush=True)
   torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,provenance=provenance),directory/'last.pt')
   if bad>=10:break
 model.load_state_dict(best_state);torch.cuda.synchronize(device) if device.type=='cuda' else None;training=dict(steps=step,best_step=best_step,optimization_validation_seconds=time.perf_counter()-start,deleted_edge_probe=len(sample),ni_nodes=[int(m.sum()) for m in ni_masks]);torch.save(dict(model=model.state_dict(),config=cfg,provenance=provenance,training=training,retained_original_ids=data.keep),directory/'deployment.tmp');(directory/'deployment.tmp').replace(directory/'deployment.pt');return model,training
