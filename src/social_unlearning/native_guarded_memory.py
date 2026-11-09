"""Memory-bounded execution of V4; same samples, objective and gradient guard.
Stage each loss backward before constructing the next graph. Exact row-chunked
InfoNCE retains the full in-batch candidate pool, including duplicate users.
No AMP, reduced batch size, extra detach, or changed training budget.
"""
import torch
from torch.nn import functional as F
from .native_guarded import GuardedNativeForgetRetain,weighted_infonce

class _ChunkedCrossEntropy(torch.autograd.Function):
 @staticmethod
 def forward(ctx,a,b,weights,temperature,chunk_size):
  ctx.save_for_backward(a,b,weights);ctx.temperature=temperature;ctx.chunk_size=chunk_size
  total=a.new_zeros(())
  for start in range(0,len(a),chunk_size):
   stop=min(start+chunk_size,len(a));logits=a[start:stop]@b.T/temperature
   total+=(F.cross_entropy(logits,torch.arange(start,stop,device=a.device),reduction='none')*weights[start:stop]).sum()
  return total/len(a)
 @staticmethod
 def backward(ctx,grad_output):
  a,b,weights=ctx.saved_tensors;t=ctx.temperature;da=torch.zeros_like(a);db=torch.zeros_like(b)
  for start in range(0,len(a),ctx.chunk_size):
   stop=min(start+ctx.chunk_size,len(a));g=(a[start:stop]@b.T/t).softmax(1)
   g[torch.arange(stop-start,device=a.device),torch.arange(start,stop,device=a.device)]-=1
   g*=weights[start:stop,None]*(grad_output/len(a))
   da[start:stop]=g@b/t;db.add_(g.T@a[start:stop]/t)
  return da,db,None,None,None

def chunked_inbatch_infonce(anchor,candidate,temperature,weights,chunk_size=256):
 return _ChunkedCrossEntropy.apply(F.normalize(anchor,dim=1),F.normalize(candidate,dim=1),weights.detach(),temperature,chunk_size)

def gradients(loss,params):
 return [torch.zeros_like(p) if g is None else g for p,g in zip(params,torch.autograd.grad(loss,params,allow_unused=True))]

def accumulate(destination,source):
 for dest,src in zip(destination,source):dest.add_(src)

class MemoryGuardedNativeForgetRetain(GuardedNativeForgetRetain):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  # V4 consumes only scalar sample weights, not the older KL teacher table.
  self.weights.pop('targets',None);self.weights.pop('candidates',None)
  # Original parameters are immutable; a detached view avoids a duplicate table.
  self.initial_deleted=self.original.embedding_dict['user_emb'].detach()
 def backward_step(self,student,u,p,n,indices,params,project=True,chunk_size=256):
  # Keep the same forward/RNG order as V4; ordinary backward consumes no RNG.
  fu,fi=student();rec=F.softplus(-(fu[u]*(fi[p]-fi[n])).sum(1));rw=torch.ones_like(rec) if self.uniform else self.weights['rec'][indices].detach()
  retain_rec=(rw*rec).mean();reg=student.reg*(fu[u].norm()+fi[p].norm()+fi[n].norm());gr=gradients(retain_rec+reg,params);rrec=retain_rec.detach();rreg=reg.detach();del fu,fi,rec,retain_rec,reg
  av,_=student('adaptive');bv,_=student();cw=torch.ones(len(u),device=u.device) if self.uniform or not self.contrastive_weighting else self.weights['anchor'][u].detach()
  retain_cl=chunked_inbatch_infonce(av[u],bv[u],student.ssl_temp,cw,chunk_size);accumulate(gr,gradients(student.ssl_reg*retain_cl,params));rcl=retain_cl.detach();del av,bv,retain_cl
  ids=self.rng.integers(len(self.forget_pairs),size=min(self.forget_batch,len(self.forget_pairs)));pairs=self.forget_pairs[ids];du,dp=torch.as_tensor(pairs,device=u.device).T;dn=torch.as_tensor(self.sample_negatives(pairs[:,0]),device=u.device)
  self.original.train(student.training);old_u,old_i=self.request_forward(student);fw=torch.ones(len(du),device=u.device) if self.uniform else self.forget_rec[torch.as_tensor(ids,device=u.device)]
  forget_rec=(fw*F.softplus((old_u[du]*(old_i[dp]-old_i[dn])).sum(1))).mean();gf=gradients(forget_rec,params);frec=forget_rec.detach();del old_u,old_i,forget_rec
  old_av,_=self.request_forward(student,'adaptive');old_bv,_=self.request_forward(student);aw=torch.ones(len(du),device=u.device) if self.uniform or not self.contrastive_weighting else self.forget_anchor[du]
  forget_cl=weighted_infonce(old_av[du],old_bv[du],old_bv[self.keep[u.unique()]],student.ssl_temp,aw)
  accumulate(gf,gradients(-self.forget_cl_strength*student.ssl_reg*forget_cl,params));fcl=forget_cl.detach();del old_av,old_bv,forget_cl
  dot=sum((a*b).sum() for a,b in zip(gr,gf));nr=sum(a.square().sum() for a in gr);conflict=float(dot)<0
  if project and conflict:
   for a,b in zip(gr,gf):b.add_(a,alpha=-float(dot/(nr+1e-12)))
  nf=sum(b.square().sum() for b in gf);scale=min(1.,float((nr/(nf+1e-12)).sqrt())) if project else 1.
  for p,a,b in zip(params,gr,gf):p.grad=a.add_(b,alpha=self.forget_strength*scale)
  retain=rrec+student.ssl_reg*rcl+rreg;forget=frec-self.forget_cl_strength*student.ssl_reg*fcl
  self.components=dict(retain=retain,forget=forget)
  self.last_components={k:float(v) for k,v in dict(retain_rec=rrec,retain_cl=rcl,forget_rec=frec,forget_cl=fcl,total=retain+self.forget_strength*forget).items()}
  # The caller clips with error_if_nonfinite=True; avoid one GPU sync per parameter.
  if not torch.isfinite(retain+self.forget_strength*forget):raise ValueError('Nonfinite staged objective')
  return dict(conflict=conflict,forget_gradient_scale=scale,retain_gradient_norm=float(nr.sqrt()))
