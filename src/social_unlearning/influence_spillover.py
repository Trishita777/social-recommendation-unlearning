"""Original-minus-retained native objective and surviving-coordinate influence solve.

No additional recommendation loss or dataset-size multiplier is introduced.
Deleted user rows are detached constants in the original objective and do not
exist in the retained model. Randomness is replayed for every objective/HVP.
"""
from contextlib import contextmanager
import math
import numpy as np
import torch
from torch import nn
from torch.func import functional_call


class NativeObjective(nn.Module):
    def __init__(self, model, backbone, batch=None, block='embedding'):
        super().__init__()
        self.model, self.backbone, self.batch, self.block = model, backbone, batch, block

    def forward(self):
        if self.backbone == 'ODSR':
            return self.model.compute_loss(*self.batch)
        return self.model.objectives()[0 if self.block == 'weight' else 1]


class SurvivingObjective:
    def __init__(self, original, retained, keep, backbone, original_batch=None,
                 retained_batch=None, seed=42, block='embedding'):
        self.original, self.retained = original, retained
        self.keep = torch.as_tensor(keep, device=next(retained.parameters()).device)
        self.user_key = 'embedding_dict.user_emb' if backbone == 'ODSR' else 'user_embeddings'
        self.original_user = dict(original.named_parameters())[self.user_key].detach().clone()
        self.old = NativeObjective(original, backbone, original_batch, block)
        self.new = NativeObjective(retained, backbone, retained_batch, block)
        names = dict(retained.named_parameters())
        if backbone == 'EGCL':
            selected = retained.weight_parameters() if block == 'weight' else retained.embedding_parameters()
            ids = {id(p) for p in selected}
            names = {n:p for n,p in names.items() if id(p) in ids}
        self.names = list(names)
        self.params = list(names.values())
        self.all_params = dict(retained.named_parameters())
        self.seed = seed
        self.device = self.params[0].device
        for p in original.parameters():p.requires_grad_(False)
        # Native ODSR dropout stays active, with a replayed fixed realization.
        original.train(); retained.train()

    def parameters_for(self, old):
        result = {}
        for name, p in self.all_params.items():
            value = p if name in self.names else p.detach()
            if old and name == self.user_key:
                value = self.original_user.index_copy(0, self.keep, value)
            result['model.'+name] = value
        return result

    def loss(self, old):
        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            if devices:torch.cuda.manual_seed(self.seed)
            return functional_call(self.old if old else self.new, self.parameters_for(old), ())

    def gradient(self, old, create_graph=False):
        loss=self.loss(old)
        grads=torch.autograd.grad(loss,self.params,create_graph=create_graph,allow_unused=True)
        grads=tuple(p*0 if g is None else g for p,g in zip(self.params,grads))
        return loss,grads

    def discrepancy(self):
        lo,go=self.gradient(True)
        ln,gn=self.gradient(False)
        b=tuple((a-c).detach() for a,c in zip(go,gn))
        return b,dict(original_objective=float(lo.detach()),retained_objective=float(ln.detach()),
                      original_gradient_norm=norm(go),retained_gradient_norm=norm(gn),
                      correction_gradient_norm=norm(b))

    def hvp(self, vector):
        _,grad=self.gradient(True,create_graph=True)
        dot=sum((g*v).sum() for g,v in zip(grad,vector))
        result=torch.autograd.grad(dot,self.params,allow_unused=True)
        return tuple(torch.zeros_like(p) if g is None else g.detach() for p,g in zip(self.params,result))


def dot(a,b):
    # Accumulate scalar products in float64 to reduce CG cancellation.
    return sum((x.double()*y.double()).sum() for x,y in zip(a,b))


def norm(a):return float(dot(a,a).sqrt().detach())


def cg(matvec, rhs, *, tolerance=1e-3, max_steps=50, callback=None):
    x=tuple(torch.zeros_like(b) for b in rhs)
    r=tuple(b.clone() for b in rhs);p=r
    scale=norm(rhs)
    if not math.isfinite(scale):raise FloatingPointError('Nonfinite deletion gradient')
    if scale==0:return x,dict(converged=True,iterations=0,relative_residual=0.,reason='zero_rhs')
    rr=dot(r,r);history=[]
    for step in range(1,max_steps+1):
        ap=matvec(p);curvature=dot(p,ap)
        if not torch.isfinite(curvature) or float(curvature)<=0:
            return x,dict(converged=False,iterations=step,relative_residual=norm(r)/scale,reason='nonpositive_curvature',history=history)
        alpha=float(rr/curvature)
        x=tuple(a+alpha*b for a,b in zip(x,p));r=tuple(a-alpha*b for a,b in zip(r,ap))
        rr_new=dot(r,r);relative=float(rr_new.sqrt())/scale
        history.append(dict(step=step,relative_residual=relative,curvature=float(curvature)))
        if callback:callback(history[-1])
        if not math.isfinite(relative):raise FloatingPointError('Nonfinite CG residual')
        if relative<=tolerance:
            actual=tuple(a-b for a,b in zip(matvec(x),rhs));verified=norm(actual)/scale
            if verified<=tolerance:
                return x,dict(converged=True,iterations=step,relative_residual=verified,reason='converged',history=history)
            # Restart from the true residual rather than claiming recurrence convergence.
            r=tuple(-a for a in actual);rr_new=dot(r,r);p=r;rr=rr_new
            continue
        beta=float(rr_new/rr);p=tuple(a+beta*b for a,b in zip(r,p));rr=rr_new
    actual=tuple(a-b for a,b in zip(matvec(x),rhs))
    return x,dict(converged=False,iterations=max_steps,relative_residual=norm(actual)/scale,reason='iteration_limit',history=history)


def solve_influence(objective, *, damping=.01, tolerance=1e-3, max_steps=50,
                    max_restarts=8, callback=None):
    b,diagnostics=objective.discrepancy();attempts=[]
    for attempt in range(max_restarts):
        mu=damping*10**attempt
        def operator(v):return tuple(h+mu*q for h,q in zip(objective.hvp(v),v))
        delta,report=cg(operator,b,tolerance=tolerance,max_steps=max_steps,callback=callback)
        report['damping']=mu;attempts.append(report)
        if report['converged']:
            diagnostics.update(solver=report,attempts=attempts,update_norm=norm(delta),
                               parameter_norm=norm(objective.params),normalization='native objective; no extra sample-size factor')
            return delta,diagnostics
        if callback:callback(dict(restart=attempt+1,reason=report['reason'],damping=mu))
    raise RuntimeError('Influence solve failed; no deployment produced: '+str(attempts))


def solve_shifted_krylov(objective, *, damping=.01, tolerance=1e-3,
                         max_steps=50, max_restarts=8, callback=None):
    """Reuse a matrix-free Krylov basis across damping levels.

    For symmetric H, K_k(H,b) is invariant to adding mu*I. Rather than
    repeat the expensive native HVPs for every shift, build an orthogonal
    basis once and solve systems of at most k by k in that basis. The full
    Hessian is never assembled. Every accepted result has a fresh true-HVP
    residual check; projected curvature must be positive after damping.
    """
    b,diagnostics=objective.discrepancy();scale=norm(b)
    if scale==0:
        zero=tuple(torch.zeros_like(p) for p in objective.params)
        report=dict(converged=True,iterations=0,relative_residual=0.,damping=damping,reason='zero_rhs')
        diagnostics.update(solver=report,attempts=[report],update_norm=0.,parameter_norm=norm(objective.params))
        return zero,diagnostics
    basis=[tuple(v/scale for v in b)]
    projected=torch.zeros((max_steps+1,max_steps),dtype=torch.double)
    hvp_calls=0
    for k in range(max_steps):
        w=objective.hvp(basis[k]);hvp_calls+=1
        if not all(torch.isfinite(v).all() for v in w):raise FloatingPointError('Nonfinite native Hessian-vector product')
        # Twice modified Gram-Schmidt keeps the basis usable with float32 HVPs.
        for _ in range(2):
            for j,q in enumerate(basis):
                value=float(dot(q,w));projected[j,k]+=value
                w=tuple(a-value*c for a,c in zip(w,q))
        beta=norm(w);projected[k+1,k]=beta
        if callback:callback(dict(krylov_step=k+1,hvp_calls=hvp_calls,next_basis_norm=beta))
        if beta<=1e-10 or k+1==max_steps:break
        basis.append(tuple(v/beta for v in w))
    size=k+1;t=projected[:size,:size];t=(t+t.T)/2
    eigmin=float(torch.linalg.eigvalsh(t).min());rhs=torch.zeros(size,dtype=torch.double);rhs[0]=scale
    attempts=[]
    for attempt in range(max_restarts):
        mu=damping*10**attempt
        report=dict(converged=False,iterations=size,damping=mu,projected_min_eigenvalue=eigmin)
        if eigmin+mu<=0:
            report.update(reason='nonpositive_projected_curvature',relative_residual=1.);attempts.append(report);continue
        coefficients=torch.linalg.solve(t+mu*torch.eye(size,dtype=torch.double),rhs)
        estimated=abs(float(projected[size,size-1]*coefficients[-1]))/scale
        if estimated>tolerance*1.05:
            report.update(reason='projected_residual',relative_residual=estimated);attempts.append(report);continue
        delta=tuple(sum(float(c)*q[j] for c,q in zip(coefficients,basis)) for j in range(len(b)))
        h=objective.hvp(delta);hvp_calls+=1
        actual=norm(tuple(a+mu*v-r for a,v,r in zip(h,delta,b)))/scale
        report.update(relative_residual=actual,converged=actual<=tolerance,reason='converged' if actual<=tolerance else 'true_residual_failed')
        attempts.append(report)
        if report['converged']:
            report.update(hvp_calls=hvp_calls,solver='shared_shift_Krylov_with_reorthogonalization')
            diagnostics.update(solver=report,attempts=attempts,update_norm=norm(delta),parameter_norm=norm(objective.params),normalization='native objective; no extra sample-size factor')
            return delta,diagnostics
    raise RuntimeError('No damped matrix-free Krylov system passed its actual residual check: '+str(attempts))
