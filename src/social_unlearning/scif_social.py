"""Selective damped-influence adapter for joint ODSR user deletion.
SCIF/ERASE-inspired social adaptation, not unchanged benchmark reproduction:
mean-loss gradient change accounts for rebuilt graphs and contrastive pools.
Selects retained-user embeddings only. No certificate is transferred.
"""
import torch
from torch.nn import functional as F
from social_unlearning.baselines import RequestGraph


def conjugate_gradient(operator,b,max_iter=50,tolerance=1e-5):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rr=(r*r).sum();initial=float(rr.sqrt());trace=[]
    if initial==0:return x,dict(status='zero_rhs',iterations=0,relative_residual=0.,trace=[])
    for iteration in range(1,max_iter+1):
        q=operator(p).detach();curvature=(p*q).sum()
        if not torch.isfinite(q).all() or not torch.isfinite(curvature):return x,dict(status='nonfinite_operator',iterations=iteration,trace=trace)
        if float(curvature)<=0:return x,dict(status='negative_curvature',iterations=iteration,trace=trace)
        alpha=rr/curvature;x=x+alpha*p;r=r-alpha*q;new_rr=(r*r).sum();relative=float(new_rr.sqrt())/initial;trace.append(relative)
        if relative<=tolerance:return x,dict(status='converged',iterations=iteration,relative_residual=relative,trace=trace)
        p=r+(new_rr/rr)*p;rr=new_rr
    return x,dict(status='iteration_limit',iterations=max_iter,relative_residual=trace[-1],trace=trace)


def native_probe(model,u,p,n):
    fu,fi=model();a,_=model('adaptive');b,_=model()
    rec=F.softplus(-(fu[u]*(fi[p]-fi[n])).sum(1)).mean();cl=F.cross_entropy(F.normalize(a[u],dim=1)@F.normalize(b[u],dim=1).T/model.ssl_temp,torch.arange(len(u),device=u.device))
    return rec+model.ssl_reg*cl+model.reg*(fu[u].norm()+fi[p].norm()+fi[n].norm())


def gradient_change(original,student,data,raw,seed,batch_size=512):
    import numpy as np
    rng=np.random.default_rng(seed);request=RequestGraph(original,student,data,raw,seed);student.eval()
    ids=rng.choice(len(data.train),min(batch_size,len(data.train)),replace=False);pairs=data.train[ids];u,p=torch.as_tensor(pairs,device=student.device).T;n=torch.as_tensor(request.negatives(data.keep[pairs[:,0]]),device=student.device)
    du_pairs=request.forget_pairs[rng.choice(len(request.forget_pairs),min(batch_size,len(request.forget_pairs)),replace=False)];du,dp=torch.as_tensor(du_pairs,device=student.device).T;dn=torch.as_tensor(request.negatives(du_pairs[:,0]),device=student.device)
    deleted=np.setdiff1d(np.arange(raw.U),data.keep);extra=torch.as_tensor(rng.choice(deleted,min(len(deleted),max(1,round(len(u)*len(deleted)/raw.U))),replace=False),device=student.device)
    for param in student.parameters():param.requires_grad_(False)
    selected=student.embedding_dict['user_emb'];selected.requires_grad_(True)
    new_loss=native_probe(student,u,p,n);new_grad=torch.autograd.grad(new_loss,selected)[0].detach()
    old_u,old_i=request.forward(student);old_a,_=request.forward(student,'adaptive');old_b,_=request.forward(student)
    global_u=request.keep[u];query=F.normalize(old_a[global_u],dim=1);candidates=F.normalize(torch.cat([old_b[global_u],old_b[extra]]),dim=1)
    old_rec=F.softplus(-(old_u[global_u]*(old_i[p]-old_i[n])).sum(1)).mean();old_cl=F.cross_entropy(query@candidates.T/student.ssl_temp,torch.arange(len(u),device=u.device));old_ret=old_rec+student.ssl_reg*old_cl
    forget_rec=F.softplus(-(old_u[du]*(old_i[dp]-old_i[dn])).sum(1)).mean();aq=F.normalize(old_a[du],dim=1);bp=F.normalize(old_b[du],dim=1);bc=F.normalize(old_b[global_u],dim=1);pos=(aq*bp).sum(1)/student.ssl_temp;logits=torch.cat([pos[:,None],aq@bc.T/student.ssl_temp],1);forget_cl=(torch.logsumexp(logits,1)-pos).mean()
    fraction=len(request.forget_pairs)/len(raw.train);old_loss=(1-fraction)*old_ret+fraction*(forget_rec+student.ssl_reg*forget_cl)
    old_grad=torch.autograd.grad(old_loss,selected)[0].detach();change=new_grad-old_grad
    return change,(u,p,n),dict(retained_probe=len(u),forget_probe=len(du),deleted_candidate_probe=len(extra),removed_interaction_fraction=fraction,gradient_change_norm=float(change.norm()),scope='retained user table only',loss_units='mean-loss gradient difference; source request scaling absorbed into consistently normalized difference')
