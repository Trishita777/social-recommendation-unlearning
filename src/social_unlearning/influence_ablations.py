"""Controlled native-objective decomposition with shared curvature and damping.

J* changes examples/sampling to retained ones while keeping original graphs.
b_full = (g0-g*) + (g*-gR). Each component uses the SAME original Hessian.
No outcome metric selects damping, update size or an ablation checkpoint.
"""
from contextlib import nullcontext
import torch
from torch.func import functional_call
from .influence_spillover import cg, norm


def component_gradients(objective, intermediate):
    _,g0=objective.gradient(True)
    _,gr=objective.gradient(False)
    devices=[objective.device.index] if objective.device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(objective.seed)
        if devices:torch.cuda.manual_seed(objective.seed)
        loss=functional_call(intermediate,objective.parameters_for(True),())
    gm=torch.autograd.grad(loss,objective.params,allow_unused=True)
    gm=tuple(torch.zeros_like(p) if g is None else g for p,g in zip(objective.params,gm))
    rhs={
        'full':tuple((a-b).detach() for a,b in zip(g0,gr)),
        'examples_only':tuple((a-b).detach() for a,b in zip(g0,gm)),
        'graph_change_only':tuple((a-b).detach() for a,b in zip(gm,gr))}
    error=norm(tuple(a-b-c for a,b,c in zip(rhs['full'],rhs['examples_only'],rhs['graph_change_only'])))
    scale=max(norm(rhs['full']),1e-12)
    if error/scale>1e-5:raise AssertionError('Discrepancy decomposition failed')
    return rhs,dict(decomposition_relative_error=error/scale,gradient_norms={k:norm(v) for k,v in rhs.items()},
                    intermediate_objective=float(loss.detach()))


def shared_damping_solve(objective,rhs,initial_damping,callback=None,max_levels=8):
    attempts=[];calls=0
    for level in range(max_levels):
        mu=initial_damping*10**level;updates={};reports={}
        def operator(vector):
            nonlocal calls
            calls+=1
            return tuple(h+mu*v for h,v in zip(objective.hvp(vector),vector))
        for name,b in rhs.items():
            delta,report=cg(operator,b,tolerance=1e-3,max_steps=50)
            reports[name]=report
            if callback:callback(dict(component=name,damping=mu,report=report))
            if not report['converged']:break
            updates[name]=delta
        attempts.append(dict(damping=mu,solves=reports))
        if len(updates)==len(rhs):
            scale=norm(rhs['full'])
            # Controls step magnitude; it is a direction ablation, not a cheap
            # standalone algorithm (the reference norm requires the full solve).
            alpha=norm(updates['full'])/scale if scale else 0.
            updates['gradient_norm_matched']=tuple(alpha*b for b in rhs['full'])
            updates['graph_only']=tuple(torch.zeros_like(p) for p in objective.params)
            return updates,dict(shared_damping=mu,attempts=attempts,hvp_calls=calls,
                gradient_step_coefficient=alpha,update_norms={k:norm(v) for k,v in updates.items()},
                selection='shared damping chosen solely by all three true residual checks; no test/MIA selection')
    raise RuntimeError('No shared damping passed all component residual checks')
