"""DiffNet++ released architecture, explicitly without side attributes.

Reference: PeiJieSun/diffnet cad45d58cc4667bcaa787c8601871eb18963f47a,
Diffnet++/diffnetplus.py (MIT; vendored under external/DiffNetPlus).
Only active prediction parameters are retained. Attribute fusion is absent as
in the paper's no-attribute setting. The released sigmoid squared-error sum,
two diffusion layers, trainable edge logits and double-exponential low-level
attention are preserved. Graph buffers are not learned checkpoint parameters.
"""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_graph(raw, keep):
    keep = np.asarray(keep, dtype=np.int64)
    mapping = np.full(raw.U, -1, dtype=np.int64)
    mapping[keep] = np.arange(len(keep))
    def mapped(pairs):
        out = pairs[mapping[pairs[:, 0]] >= 0].copy()
        out[:, 0] = mapping[out[:, 0]]
        return out
    train = mapped(raw.train)
    social = raw.social[(mapping[raw.social] >= 0).all(1)]
    social = mapping[social]
    social = np.unique(np.concatenate([social, social[:, ::-1]]), axis=0)
    ui = np.unique(train, axis=0)
    iu = np.unique(train[:, ::-1], axis=0)
    edges = dict(uu=social, ui=ui, iu=iu)
    # Global keys provide an exact original-to-retained edge-parameter map.
    keys = {}
    for name, pairs in edges.items():
        global_pairs = pairs.copy()
        if name in ('uu', 'ui'): global_pairs[:, 0] = keep[pairs[:, 0]]
        if name in ('uu', 'iu'): global_pairs[:, 1] = keep[pairs[:, 1]]
        keys[name] = global_pairs[:, 0] * (raw.I if name == 'ui' else raw.U) + global_pairs[:, 1]
    positives = [set() for _ in keep]
    for u, i in train: positives[u].add(int(i))
    return SimpleNamespace(user_num=len(keep), item_num=raw.I, keep=keep,
                           mapping=mapping, train=train, val=mapped(raw.val),
                           test=mapped(raw.test), positives=positives,
                           edges=edges, edge_keys=keys)


class WeightedAggregate(torch.autograd.Function):
    """Sparse forward with differentiable backward, including edge-value HVPs."""
    @staticmethod
    def forward(ctx, values, x, indices, nrows):
        ctx.save_for_backward(values, x, indices)
        return torch.sparse.mm(torch.sparse_coo_tensor(
            indices, values, (nrows, len(x)), device=x.device, check_invariants=False), x)

    @staticmethod
    def backward(ctx, grad):
        values, x, indices = ctx.saved_tensors
        row, col = indices
        # Chunking bounds the edge-by-dimension temporary in large graphs.
        value_grads = []
        xgrad = torch.zeros_like(x)
        for start in range(0, len(values), 65536):
            r, c = row[start:start+65536], col[start:start+65536]
            g = grad[r]
            value_grads.append((g * x[c]).sum(1))
            xgrad = xgrad.index_add(0, c, values[start:start+65536, None] * g)
        vg = torch.cat(value_grads) if value_grads else values * 0
        return vg, xgrad, None, None


class Attention(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.first = nn.Linear(dimension, 1)
        self.second = nn.Linear(1, 1)

    def forward(self, x):
        return F.leaky_relu(self.second(torch.tanh(self.first(x))), .2).exp()


class DiffNetPlus(nn.Module):
    def __init__(self, data, dimension=64, device='cpu'):
        super().__init__()
        self.dimension = dimension
        self.user_embeddings = nn.Parameter(torch.empty(data.user_num, dimension))
        self.item_embeddings = nn.Parameter(torch.empty(data.item_num, dimension))
        self.edge_logits = nn.ParameterDict()
        self.low = nn.ModuleDict()
        self.high = nn.ModuleDict()
        for relation, edges in data.edges.items():
            self.register_buffer('edges_' + relation, torch.as_tensor(edges.T.copy(), dtype=torch.long), persistent=False)
        for layer in range(2):
            for relation in ('uu', 'ui', 'iu'):
                key = f'l{layer}_{relation}'
                self.edge_logits[key] = nn.Parameter(torch.empty(len(data.edges[relation])))
                self.low[key] = nn.Linear(1, 1)
            for relation in ('uu', 'ui', 'self', 'iu'):
                self.high[f'l{layer}_{relation}'] = Attention(dimension * (2 if relation in ('uu', 'ui') else 1))
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        nn.init.normal_(self.user_embeddings, std=.01)
        nn.init.normal_(self.item_embeddings, std=.01)
        for key, p in self.edge_logits.items():
            nn.init.normal_(p, std=.01 if key == 'l1_iu' else 1.)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def aggregate(self, layer, relation, x, nrows):
        key = f'l{layer}_{relation}'
        indices = getattr(self, 'edges_' + relation)
        # TF sparse.softmax(exp(sigmoid(Dense(edge_variable)))).
        logits = torch.sigmoid(self.low[key](self.edge_logits[key][:, None])).flatten().exp()
        weights = logits.exp()
        denom = weights.new_zeros(nrows).index_add(0, indices[0], weights)
        weights = weights / denom[indices[0]]
        return WeightedAggregate.apply(weights, x, indices, nrows)

    def forward(self):
        u, i = self.user_embeddings, self.item_embeddings
        users, items = [], []
        for layer in range(2):
            ui = self.aggregate(layer, 'ui', i, len(u))
            uu = self.aggregate(layer, 'uu', u, len(u))
            iu = self.aggregate(layer, 'iu', u, len(i))
            a = self.high[f'l{layer}_ui'](torch.cat([u, ui], 1)) + .7
            b = self.high[f'l{layer}_uu'](torch.cat([u, uu], 1)) + .3
            next_u = .5*u + .5*(a*ui + b*uu)/(a+b)
            a = self.high[f'l{layer}_self'](i) + 1.
            b = self.high[f'l{layer}_iu'](iu) + 1.
            next_i = (a*i + b*iu)/(a+b)
            u, i = next_u, next_i
            users.append(u); items.append(i)
        return torch.cat(users + [self.user_embeddings], 1), torch.cat(items + [self.item_embeddings], 1)

    def loss(self, users, items, labels):
        u, i = self()
        prediction = (u[users] * i[items]).sum(1).sigmoid()
        return .5 * (labels - prediction).square().sum()


def parameter_map(original_data, retained_data):
    result = {'user_embeddings': torch.as_tensor(retained_data.keep, dtype=torch.long)}
    for relation, keys in retained_data.edge_keys.items():
        oldkeys = original_data.edge_keys[relation]
        order = np.argsort(oldkeys)
        positions = np.searchsorted(oldkeys[order], keys)
        if np.any(positions >= len(oldkeys)) or not np.array_equal(oldkeys[order[positions]], keys):
            raise ValueError('Retained edge absent from original graph')
        index = torch.as_tensor(order[positions], dtype=torch.long)
        for layer in range(2): result[f'edge_logits.l{layer}_{relation}'] = index
    return result


def parameter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def compact(original, original_data, retained_data):
    model = DiffNetPlus(retained_data, original.dimension, original.user_embeddings.device).to(original.user_embeddings.dtype)
    mapping = parameter_map(original_data, retained_data)
    state = parameter_state(original)
    for name, index in mapping.items(): state[name] = state[name][index]
    model.load_state_dict(state)
    return model


def training_examples(data, rng, negative_draws=8):
    """Released user-batch sampler: eight draws/positive, deduplicated per user."""
    result = {}
    for user, positive in enumerate(data.positives):
        pos = np.array(sorted(positive), dtype=np.int64)
        if not len(pos): continue
        if len(pos) == data.item_num: raise ValueError('User has no negative items')
        neg = rng.integers(data.item_num, size=len(pos)*negative_draws)
        invalid = np.isin(neg, pos)
        while invalid.any():
            neg[invalid] = rng.integers(data.item_num, size=int(invalid.sum()))
            invalid = np.isin(neg, pos)
        neg = np.unique(neg)
        result[user] = (np.concatenate([pos, neg]), np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.float32))
    return result


def user_batch(examples, users, device):
    users = [int(u) for u in users if int(u) in examples]
    items = np.concatenate([examples[u][0] for u in users])
    labels = np.concatenate([examples[u][1] for u in users])
    ids = np.concatenate([np.full(len(examples[u][0]), u, dtype=np.int64) for u in users])
    return (torch.as_tensor(ids, device=device), torch.as_tensor(items, device=device), torch.as_tensor(labels, device=device))
