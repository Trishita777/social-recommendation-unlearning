"""Explicit EGCL adaptations; no exact-unlearning or source-parity claim.

The candidate uses the difference of native gradients at the SAME parameters
on original and edge-deleted graphs, then native retained optimization.
GNNDelete uses the released DeletionLayer and node-embedding loss form, extended
to EGCL's recommendation/social propagation streams and attention pooling.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gnndelete_edges import deletion_class, neighborhoods


def native_gradients(model):
    """Two native block gradients, evaluated at a common unmodified parameter state."""
    model.train()  # enables exact chunk checkpointing of the native full InfoNCE
    weight_ids = {id(p) for p in model.weight_parameters()}
    named = dict(model.named_parameters())
    output = {}
    for weight_block in (True, False):
        selected = [(name, p) for name, p in named.items() if (id(p) in weight_ids) == weight_block]
        weight_loss, embedding_loss = model.objectives(include_cl=not weight_block)
        loss = weight_loss if weight_block else embedding_loss
        grads = torch.autograd.grad(loss, [p for _, p in selected], allow_unused=False)
        for (name, _), grad in zip(selected, grads):
            if not torch.isfinite(grad).all():
                raise FloatingPointError('Nonfinite native EGCL gradient')
            output[name] = grad.detach()
        del weight_loss, embedding_loss, loss, grads
    return output


def topology_correction(original, student, deleted, relative_step=.001):
    """One bounded first-order correction, not an inverse-Hessian estimate.

Each native optimizer block receives -[g_after(theta0)-g_before(theta0)],
normalized so its Euclidean displacement is relative_step times that block's
shared parameter norm. Deleted raw rows are excluded and set to zero, the
conditional optimum of their positive native L2 term after isolation.
"""
    if not 0 <= relative_step <= 1:
        raise ValueError('relative_step must be in [0,1]')
    before = dict(original.named_parameters()); after = dict(student.named_parameters())
    if set(before) != set(after) or any(not torch.equal(before[n], after[n]) for n in before):
        raise ValueError('Gradient difference requires identical original/student parameters')
    ids = torch.as_tensor(deleted, device=student.device, dtype=torch.long)
    for adj in (student.recadj, student.socadj):
        if torch.isin(adj.coalesce().indices(), ids).any():
            raise ValueError('Requested users must be fully isolated')
    old_grad = native_gradients(original)
    new_grad = native_gradients(student)
    delta = {name: new_grad[name] - old_grad[name] for name in after}
    delta['user_embeddings'][ids] = 0
    weight_ids = {id(p) for p in student.weight_parameters()}
    report = dict(relative_step=relative_step, blocks={},
                  approximation='normalized first-order native gradient difference, no Hessian inverse/certificate')
    with torch.no_grad():
        for block, weight_block in [('weight_network', True), ('embedding_attention', False)]:
            names = [n for n, p in after.items() if (id(p) in weight_ids) == weight_block]
            parameter_sq = sum(p.square().sum() for n, p in after.items() if n in names)
            if not weight_block:
                parameter_sq -= student.user_embeddings[ids].square().sum()
            gradient_norm = sum(delta[n].square().sum() for n in names).sqrt()
            parameter_norm = parameter_sq.clamp_min(0).sqrt()
            scale = relative_step * parameter_norm / gradient_norm.clamp_min(1e-12)
            if float(gradient_norm) == 0:
                scale.zero_()
            for name in names:
                after[name].add_(delta[name], alpha=-float(scale))
            report['blocks'][block] = dict(gradient_difference_norm=float(gradient_norm),
                                           shared_parameter_norm=float(parameter_norm),
                                           displacement_norm=float(scale * gradient_norm))
        student.user_embeddings[ids] = 0
    report['deleted_raw_rows'] = 'zeroed to isolated native L2 optimum; outputs already structurally zero'
    return report


class GNNDeleteEGCL(nn.Module):
    """Frozen EGCL, with per-layer source operators shared across native streams."""
    def __init__(self, body, masks, source, rec_operators=None, social_operators=None):
        super().__init__()
        self.body = body; self.device = body.device
        self.user_num = body.user_num; self.item_num = body.item_num
        for p in body.parameters():
            p.requires_grad_(False)
        layer = deletion_class(source)
        d = body.user_embeddings.shape[1]
        self.rec_operators = rec_operators if rec_operators is not None else nn.ModuleList([
            layer(d, None).to(self.device) for _ in range(body.layers)])
        self.social_operators = social_operators if social_operators is not None else self.rec_operators
        for k, mask in enumerate(masks):
            self.register_buffer(f'mask_{k}', torch.as_tensor(mask, dtype=torch.bool, device=self.device))

    def operator_parameters(self):
        return list({id(p): p for p in list(self.rec_operators.parameters())
                     + list(self.social_operators.parameters())}.values())

    def propagation(self, initial, adjacency, degree_time, operators, social, apply):
        aggregate = initial; states = []
        for k in range(self.body.layers):
            aggregate = torch.sparse.mm(adjacency, aggregate)
            if apply:
                mask = getattr(self, f'mask_{k}')
                aggregate = operators[k](aggregate, mask[:self.user_num] if social else mask)
            states.append(aggregate * torch.exp(-degree_time) * degree_time.pow(k) / math.factorial(k))
        return torch.stack(states, 1)

    def layer_states(self, apply=True):
        b = self.body
        rec = self.propagation(torch.cat([b.user_embeddings, b.item_embeddings]),
                               b.recadj, b.degree_time, self.rec_operators, False, apply)
        social = self.propagation(b.user_embeddings, b.socadj, b.social_time,
                                  self.social_operators, True, apply)
        return rec, social

    def forward(self):
        rec, _ = self.layer_states()
        b = self.body
        users = b.user_attention(rec[:self.user_num], b.user_embeddings)
        items = b.item_attention(rec[self.user_num:], b.item_embeddings)
        return users, items


def deletion_setup(raw, deleted, layers, seed, cap=4096):
    """Typed fixed nonedges; every deleted directed UI/social edge is eligible."""
    masks, forgotten, edges = neighborhoods(raw, deleted, layers)
    if not len(forgotten):
        raise ValueError('No incident training edges to forget')
    rng = np.random.default_rng(seed)
    forgotten = forgotten[rng.choice(len(forgotten), min(cap, len(forgotten)), replace=False)]
    occupied = set(map(tuple, edges)); occupied.update((int(v), int(u)) for u, v in edges)
    negative = []
    for _, v in forgotten:
        for _attempt in range(100000):
            x = int(rng.integers(raw.U))
            y = int(rng.integers(raw.U, raw.U + raw.I)) if v >= raw.U else int(rng.integers(raw.U))
            if x != y and (x, y) not in occupied:
                negative.append((x, y)); break
        else:
            raise ValueError('No valid typed negative found')
    # Locality excludes all request-edge endpoints, not just sampled witnesses.
    endpoints = np.unique(edges[np.isin(edges[:, 0], deleted) | np.isin(edges[:, 1], deleted)])
    nonendpoint = np.ones(raw.U + raw.I, dtype=bool); nonendpoint[endpoints] = False
    locality = [mask & nonendpoint for mask in masks]
    return masks, forgotten, np.asarray(negative, dtype=np.int64), locality


def gnndelete_loss(model, witness, teacher, forgotten, negative, locality, alpha=.5):
    """Released nodeemb MSE form, over both native EGCL propagation streams.

Request witnesses use the original graph; targets and locality use the altered
graph. Unlike the released GCN implementation, the full graph replaces its
local induced subgraph, and native attention remains frozen.
"""
    wr, ws = witness.layer_states(); rr, rs = model.layer_states()
    tr, ts = teacher
    rec_edges = forgotten[:, 1] >= model.user_num
    randomness = rr.new_zeros(()); preserve = rr.new_zeros(())
    for k, mask in enumerate(locality):
        current = []; target = []
        for choose, z, t in [(rec_edges, wr[:, k], tr[:, k]), (~rec_edges, ws[:, k], ts[:, k])]:
            if choose.any():
                edge, neg = forgotten[choose], negative[choose]
                current.extend([z[edge[:, 0]], z[edge[:, 1]]])
                target.extend([t[neg[:, 0]], t[neg[:, 1]]])
        randomness = randomness + F.mse_loss(torch.cat(current), torch.cat(target))
        losses = []
        if mask.any():
            losses.append(F.mse_loss(rr[:, k][mask], tr[:, k][mask]))
        if mask[:model.user_num].any():
            sm = mask[:model.user_num]
            losses.append(F.mse_loss(rs[:, k][sm], ts[:, k][sm]))
        if losses:
            preserve = preserve + sum(losses) / len(losses)
    loss = alpha * randomness + (1 - alpha) * preserve
    return loss, dict(randomness=float(randomness.detach()), locality=float(preserve.detach()),
                      total=float(loss.detach()))
