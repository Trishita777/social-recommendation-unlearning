"""Broader native objectives and residual-corrected Newton updates; no new loss."""
import math
import torch
from torch import nn
from .influence_spillover import SurvivingObjective, norm, dot


class PairGradient(torch.autograd.Function):
    """Memory-bounded exact first/second derivatives of sigmoid squared error."""
    @staticmethod
    def forward(ctx,u,i,users,items,labels,scale):
        ctx.save_for_backward(u,i,users,items,labels);ctx.scale=scale
        gu=torch.zeros_like(u);gi=torch.zeros_like(i)
        for start in range(0,len(users),65536):
            a,b,y=users[start:start+65536],items[start:start+65536],labels[start:start+65536]
            x,z=u[a],i[b];p=(x*z).sum(1).sigmoid();c=(p-y)*p*(1-p)*scale
            gu.index_add_(0,a,c[:,None]*z);gi.index_add_(0,b,c[:,None]*x)
        return gu,gi

    @staticmethod
    def backward(ctx,vu,vi):
        u,i,users,items,labels=ctx.saved_tensors
        vu=torch.zeros_like(u) if vu is None else vu;vi=torch.zeros_like(i) if vi is None else vi
        hu=torch.zeros_like(u);hi=torch.zeros_like(i)
        for start in range(0,len(users),65536):
            a,b,y=users[start:start+65536],items[start:start+65536],labels[start:start+65536]
            x,z=u[a],i[b];vx,vz=vu[a],vi[b];p=(x*z).sum(1).sigmoid();s=p*(1-p)
            first=(p-y)*s*ctx.scale;second=(s*s+(p-y)*s*(1-2*p))*ctx.scale
            directional=(vx*z+x*vz).sum(1)
            hu.index_add_(0,a,(second*directional)[:,None]*z+first[:,None]*vz)
            hi.index_add_(0,b,(second*directional)[:,None]*x+first[:,None]*vx)
        return hu,hi,None,None,None,None


class PairLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,i,users,items,labels,scale):
        ctx.save_for_backward(u,i,users,items,labels);ctx.scale=scale
        total=u.new_zeros(())
        for start in range(0,len(users),65536):
            a,b,y=users[start:start+65536],items[start:start+65536],labels[start:start+65536]
            total+=.5*((u[a]*i[b]).sum(1).sigmoid()-y).square().sum()*scale
        return total

    @staticmethod
    def backward(ctx,grad):
        gu,gi=PairGradient.apply(*ctx.saved_tensors,ctx.scale)
        return grad*gu,grad*gi,None,None,None,None


class FullDiffNetLoss(nn.Module):
    def __init__(self,model,batch,user_count):
        super().__init__();self.model=model;self.batch=batch
        self.scale=1/math.ceil(user_count/512)
    def forward(self):
        u,i=self.model();return PairLoss.apply(u,i,*self.batch,self.scale)


class MeanObjective:
    def __init__(self,objectives):
        self.objectives=objectives;self.params=objectives[0].params;self.names=objectives[0].names
    def loss(self,old):
        for o in self.objectives:o.old.model.train();o.new.model.train()
        return sum(o.loss(old) for o in self.objectives)/len(self.objectives)
    def gradient(self,old,create_graph=False):
        if create_graph:return SurvivingObjective.gradient(self,old,True)
        total=None;losses=[]
        for o in self.objectives:
            o.old.model.train();o.new.model.train()
            loss,g=o.gradient(old);losses.append(loss.detach())
            total=tuple(v.detach() for v in g) if total is None else tuple(a+v.detach() for a,v in zip(total,g))
        return torch.stack(losses).mean(),tuple(v/len(self.objectives) for v in total)
    discrepancy=SurvivingObjective.discrepancy
    def hvp(self,vector):
        total=None
        for o in self.objectives:
            o.old.model.train();o.new.model.train();h=o.hvp(vector)
            total=h if total is None else tuple(a+b for a,b in zip(total,h))
        return tuple(v/len(self.objectives) for v in total)


class RetainedStationarity:
    """Newton corrector solves H_R delta = -g_R at the CURRENT parameters."""
    def __init__(self,objective):self.objective=objective;self.params=objective.params
    def discrepancy(self):
        loss,g=self.objective.gradient(False)
        return tuple(-v.detach() for v in g),dict(retained_objective=float(loss.detach()),
            retained_gradient_norm=norm(g),correction_gradient_norm=norm(g),rhs='negative retained native gradient')
    def hvp(self,vector):
        if isinstance(self.objective,MeanObjective):
            total=None
            for o in self.objective.objectives:
                o.old.model.train();o.new.model.train()
                h=RetainedStationarity(o).hvp(vector)
                total=h if total is None else tuple(a+b for a,b in zip(total,h))
            return tuple(v/len(self.objective.objectives) for v in total)
        _,g=self.objective.gradient(False,create_graph=True)
        result=torch.autograd.grad(sum((a*b).sum() for a,b in zip(g,vector)),self.params,allow_unused=True)
        return tuple(torch.zeros_like(p) if h is None else h.detach() for p,h in zip(self.params,result))


def descent_step(objective,delta,max_backtracks=12):
    before,g=objective.gradient(False)
    before=float(before.detach());slope=float(dot(g,delta))
    initial=[p.detach().clone() for p in objective.params]
    if not math.isfinite(slope) or slope>=0:return dict(accepted=False,reason='not_a_descent_direction',slope=slope)
    trials=[]
    for k in range(max_backtracks):
        alpha=2.**(-k)
        with torch.no_grad():
            for p,old,d in zip(objective.params,initial,delta):p.copy_(old+alpha*d)
        with torch.no_grad():after=float(objective.loss(False))
        trials.append(dict(alpha=alpha,objective=after))
        if math.isfinite(after) and after<=before+1e-4*alpha*slope:
            return dict(accepted=True,alpha=alpha,objective_before=before,objective_after=after,slope=slope,trials=trials)
    with torch.no_grad():
        for p,old in zip(objective.params,initial):p.copy_(old)
    return dict(accepted=False,reason='no_native_objective_decrease',objective_before=before,trials=trials)
