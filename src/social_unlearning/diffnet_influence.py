"""Joint native DiffNet++ influence, including surviving edge attention values."""
import torch
from torch import nn
from torch.func import functional_call
from .diffnet_plus import parameter_map
from .influence_spillover import SurvivingObjective


class Loss(nn.Module):
    def __init__(self, model, batch):
        super().__init__()
        self.model, self.batch = model, batch

    def forward(self):
        return self.model.loss(*self.batch)


class DiffNetObjective(SurvivingObjective):
    def __init__(self, original, retained, original_data, retained_data, old_batch, new_batch, seed=42):
        self.old = Loss(original, old_batch)
        self.new = Loss(retained, new_batch)
        self.all_params = dict(retained.named_parameters())
        self.names = list(self.all_params)
        self.params = list(self.all_params.values())
        self.device = self.params[0].device
        self.seed = seed
        self.mapping = {n: i.to(self.device) for n, i in parameter_map(original_data, retained_data).items()}
        self.constants = {n: p.detach().clone() for n, p in original.named_parameters() if n in self.mapping}
        for p in original.parameters(): p.requires_grad_(False)

    def parameters_for(self, old):
        return {'model.' + name: self.constants[name].index_copy(0, self.mapping[name], p)
                if old and name in self.mapping else p for name, p in self.all_params.items()}


def solve(objective, damping=.01, tolerance=1e-3, max_steps=50, max_restarts=10, callback=None):
    """Residual-stopped damped CG; a fresh true HVP verifies every accepted solve."""
    from .influence_spillover import cg, norm
    rhs, diagnostics = objective.discrepancy()
    scale = norm(rhs)
    attempts = []
    calls = 0
    for attempt in range(max_restarts):
        mu = damping * 10**attempt
        def operator(v):
            nonlocal calls
            calls += 1
            return tuple(h+mu*x for h, x in zip(objective.hvp(v), v))
        delta, report = cg(operator, rhs, tolerance=tolerance, max_steps=max_steps, callback=callback)
        report['damping'] = mu
        if report['converged'] and scale:
            actual = norm(tuple(a-b for a, b in zip(operator(delta), rhs))) / scale
            report.update(relative_residual=actual, converged=actual <= tolerance,
                          reason='true_residual_pass' if actual <= tolerance else 'true_residual_fail')
        attempts.append(report)
        if callback: callback(dict(attempt=attempt+1, **report))
        if report['converged']:
            diagnostics.update(solver=report, attempts=attempts, hvp_calls=calls,
                               update_norm=norm(delta), parameter_norm=norm(objective.params),
                               normalization='native half-sum squared error; no extra sample-size factor',
                               approximation='fixed coupled 512-user minibatches; full-graph joint Hessian over surviving users, items, edge values and MLPs')
            return delta, diagnostics
    raise RuntimeError('No damped system passed its true residual check: ' + str(attempts))
