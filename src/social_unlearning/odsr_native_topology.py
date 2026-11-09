"""Finite-probe ODSR topology correction, followed by native retained learning.

No reverse BPR, contrastive ascent, Hessian inverse or unlearning certificate.
The correction uses deterministic evaluation-mode native gradients; dropout's
stochastic training objective is not estimated by this one-probe approximation.
"""
import numpy as np
import torch


def paired_probe(original_data, deleted, seed, size=1024):
    from run_unlearning import negatives
    if size < 2:
        raise ValueError('Probe must contain at least two original triplets')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(original_data.train), min(size, len(original_data.train)), replace=False)
    pairs = original_data.train[indices].copy()
    negative = negatives(original_data, pairs[:, 0], rng)
    retained = ~np.isin(pairs[:, 0], deleted)
    if retained.sum() < 2:
        raise ValueError('Insufficient retained triplets in fixed original probe')
    return dict(pairs=pairs, negative=negative, retained=retained,
                original_indices=indices, seed=seed,
                sampling='uniform original training interactions without replacement; no forced deleted-user coverage')


def probe_gradient(model, pairs, negative):
    from run_unlearning import loss_terms
    model.eval()
    users, items = torch.as_tensor(pairs, device=model.device).T
    negative = torch.as_tensor(negative, device=model.device)
    loss = loss_terms(model, users, items, negative, mode='original')
    named = list(model.named_parameters())
    grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    result = {name: torch.zeros_like(parameter) if grad is None else grad.detach()
              for (name, parameter), grad in zip(named, grads)}
    if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in result.values()):
        raise FloatingPointError('Nonfinite ODSR probe loss or gradient')
    return result, float(loss.detach())


def topology_correction(original, student, deleted, probe, relative_step=.001):
    if not 0 <= relative_step <= 1:
        raise ValueError('relative_step must lie in [0,1]')
    original_params = dict(original.named_parameters()); student_params = dict(student.named_parameters())
    if set(original_params) != set(student_params) or any(
            not torch.equal(original_params[name], student_params[name]) for name in original_params):
        raise ValueError('Native gradient difference requires identical parameter values')
    deleted = torch.as_tensor(deleted, dtype=torch.long, device=student.device)
    for adjacency in (student.adj_ui, student.adj_uu, student.sparse_norm_adj):
        indices = adjacency.coalesce().indices()
        # Item columns of adj_ui live in their own ID space; inspect user rows only.
        inspected = indices[0] if adjacency is student.adj_ui else indices
        if torch.isin(inspected, deleted).any():
            raise ValueError('Requested users must have no incident student graph edges')
    before, before_loss = probe_gradient(original, probe['pairs'], probe['negative'])
    mask = probe['retained']
    after, after_loss = probe_gradient(student, probe['pairs'][mask], probe['negative'][mask])
    difference = {name: after[name] - before[name] for name in student_params}
    difference['embedding_dict.user_emb'][deleted] = 0
    with torch.no_grad():
        gradient_norm = sum(g.square().sum() for g in difference.values()).sqrt()
        parameter_sq = sum(p.square().sum() for p in student_params.values())
        parameter_sq -= student.embedding_dict['user_emb'][deleted].square().sum()
        parameter_norm = parameter_sq.clamp_min(0).sqrt()
        scale = relative_step * parameter_norm / gradient_norm.clamp_min(1e-12)
        if float(gradient_norm) == 0:
            scale.zero_()
        for name, parameter in student_params.items():
            parameter.add_(difference[name], alpha=-float(scale))
    return dict(original_probe_loss=before_loss, altered_retained_probe_loss=after_loss,
                original_triplets=len(probe['pairs']), retained_triplets=int(mask.sum()),
                deleted_triplets=int((~mask).sum()),
                deleted_users_in_probe=int(len(np.unique(probe['pairs'][~mask, 0]))),
                gradient_difference_norm=float(gradient_norm), shared_parameter_norm=float(parameter_norm),
                correction_displacement_norm=float(scale * gradient_norm), relative_step=relative_step,
                probe_seed=probe['seed'], sampling=probe['sampling'],
                approximation='one paired eval-mode finite interaction probe; retained negatives reused; native duplicate-user CL candidate multiplicity preserved within each batch; no Hessian or certificate')
