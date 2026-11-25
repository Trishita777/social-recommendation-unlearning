"""Fixed-partition user-withdrawal benchmark on ODSR and DiffNet++."""
import argparse
import hashlib
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn import functional as F

from partition_odsr import ODSR, ROOT, atomic_json, digest, evaluate, graph, load_data
from partition_odsr import loss_terms, negatives, seed_everything
from social_unlearning.recflow import RecFlow
from social_unlearning.diffnet_plus import DiffNetPlus, make_graph, training_examples, user_batch


METHODS = ('receraser_social', 'ultrare_social', 'grapheraser_social', 'guide_social', 'eemu_partition', 'sorecremover_partition')
SOCIAL_WEIGHT = 0.5
LOCAL_SUPPORT_PRIOR = 4.0
MOE_METHOD = 'sorecremover_moe'
GRAPH_MOE_METHOD = 'sorecremover_graph_moe'
MOE_OWNER_PRIOR = 0.8
MOE_KL_WEIGHT = 0.01
GRAPH_MOE_LOAD_WEIGHT = 0.002


def capacity_assignment(scores, capacities):
    """Greedily harden affinity/transport scores subject to shard capacities."""
    owners = np.full(len(scores), -1, dtype=np.int64)
    remaining = capacities.copy()
    for flat in np.argsort(-scores, axis=None, kind='stable'):
        user, shard = divmod(int(flat), len(capacities))
        if owners[user] < 0 and remaining[shard]:
            owners[user] = shard
            remaining[shard] -= 1
    if np.any(owners < 0):
        raise ValueError('Partition capacities did not cover every user')
    return owners


def social_adjacency(raw):
    edges = raw.social
    adjacency = sp.csr_matrix((np.ones(len(edges)), edges.T), shape=(raw.U, raw.U))
    return adjacency.maximum(adjacency.T)


def guide_partition(raw, adjacency, shards, seed):
    """Sparse GPFB-Fast relaxation with training-activity strata."""
    activity = np.bincount(raw.train[:, 0], minlength=raw.U)
    cutoffs = np.unique(np.quantile(activity, [.2, .4, .6, .8]))
    _, strata = np.unique(np.searchsorted(cutoffs, activity, side='left'), return_inverse=True)
    fairness = np.eye(strata.max() + 1)[strata]
    sizes = fairness.sum(axis=0)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    np.power(degree, -.5, out=inverse, where=degree > 0)
    normalized = sp.diags(inverse) @ adjacency @ sp.diags(inverse)
    # Scale each stratum's penalty for the graph size; use the sparse F(F^T H) product.
    penalty = .01 / sizes.max()
    target = sizes[:, None] / math.sqrt(raw.U * shards)
    rng = np.random.default_rng(seed)
    embedding, _ = np.linalg.qr(rng.normal(size=(raw.U, shards)))
    for _ in range(10):
        update = (10 * embedding + normalized @ embedding
                  + penalty * fairness @ (target - fairness.T @ embedding))
        left, _, right = np.linalg.svd(update, full_matrices=False)
        embedding = left @ right
    centers = KMeans(n_clusters=shards, n_init=10, max_iter=50, random_state=seed)
    distances = centers.fit_transform(embedding)
    owners = np.empty(raw.U, dtype=np.int64)
    remainder_offset = 0
    for group in range(len(sizes)):
        users = np.flatnonzero(strata == group)
        capacity = np.bincount((np.arange(len(users)) + remainder_offset) % shards, minlength=shards)
        owners[users] = capacity_assignment(-distances[users], capacity)
        remainder_offset = (remainder_offset + len(users)) % shards
    return owners


def guide_repair(data, adjacency, users, active, backbone, device):
    """Fold mirror neighbors into self messages without copying other user vectors."""
    retained = adjacency[users][:, active]
    local = adjacency[users][:, users]
    missing = np.asarray(retained.sum(1) - local.sum(1)).ravel().astype(np.int64)
    if backbone == 'DiffNet++':
        mirror_rows = np.repeat(np.arange(len(users)), missing)
        loops = np.column_stack((mirror_rows, mirror_rows))
        data.edges['uu'] = np.concatenate((data.edges['uu'], loops))
        original_ids = users[mirror_rows]
        data.edge_keys['uu'] = np.concatenate((data.edge_keys['uu'], original_ids * adjacency.shape[0] + original_ids))
    else:
        local = local.astype(np.float32)
        local.setdiag(1 + missing)
        degrees = np.asarray(local.sum(1)).ravel()
        normalized = (sp.diags(1 / degrees) @ local).tocoo()
        coordinates = torch.as_tensor(np.stack((normalized.row, normalized.col)), device=device, dtype=torch.long)
        values = torch.as_tensor(normalized.data, device=device, dtype=data.adj_uu.dtype)
        data.adj_uu = torch.sparse_coo_tensor(coordinates, values, normalized.shape).coalesce()


def guide_affected_shards(adjacency, owners, deleted):
    neighbors = np.unique(adjacency[deleted].indices)
    return set(owners[np.concatenate((deleted, neighbors))])


def pyramid_histograms(adjacency, seed):
    dimensions = min(6, adjacency.shape[0])
    if adjacency.nnz:
        vectors, _, _ = randomized_svd(adjacency, n_components=dimensions, random_state=seed)
        magnitudes = np.abs(vectors) * math.sqrt(adjacency.shape[0])
        coordinates = magnitudes / (1 + magnitudes)
    else:
        coordinates = np.zeros((adjacency.shape[0], dimensions))
    coordinates = np.pad(coordinates, ((0, 0), (0, 6 - dimensions)))
    return [np.stack([np.bincount((column * bins).astype(int), minlength=bins)
                      for column in coordinates.T]) / len(coordinates)
            for bins in (1, 2, 4, 8)]


def guide_weights(adjacency, parts, active, seed):
    query = pyramid_histograms(adjacency[active][:, active], seed)
    scores = []
    for users, _, _ in parts:
        shard = pyramid_histograms(adjacency[users][:, users], seed)
        matches = [np.minimum(q, p).sum() for q, p in zip(query, shard)]
        scores.append(matches[-1] + sum((matches[i] - matches[i+1]) / 2**(3-i) for i in range(3)))
    return np.asarray(scores) / sum(scores)


def workload_balanced_assignment(scores, workload_order, shards):
    """Preserve quotas within narrow bands of estimated training work."""
    owner = np.empty(len(scores), dtype=np.int64)
    width = 64 * shards
    for start in range(0, len(scores), width):
        rows = workload_order[start:start + width]
        capacity = np.bincount(np.arange(len(rows)) % shards, minlength=shards)
        owner[rows] = capacity_assignment(scores[rows], capacity)
    return owner


def partition(raw, method, shards, seed, features=None, ablation=None):
    rng = np.random.default_rng(seed)
    if method == 'grapheraser_social':
        # Sparse, capacity-constrained label propagation on the social relation.
        edges = raw.social
        adj = sp.csr_matrix((np.ones(len(edges)), edges.T), shape=(raw.U, raw.U))
        adj = adj.maximum(adj.T)
        cap = math.ceil(raw.U / shards * 1.05)
        owner = np.empty(raw.U, dtype=np.int64)
        owner[rng.permutation(raw.U)] = np.arange(raw.U) % shards
        counts = np.bincount(owner, minlength=shards)
        for _ in range(30):
            votes = adj @ np.eye(shards)[owner]
            gain = votes.max(1) - votes[np.arange(raw.U), owner]
            moved = 0
            for user in np.argsort(-gain, kind='stable'):
                source = owner[user]
                target = int(votes[user].argmax())
                if gain[user] > 0 and counts[target] < cap and counts[source] > 1:
                    owner[user] = target
                    counts[source] -= 1
                    counts[target] += 1
                    moved += 1
            if moved == 0:
                break
        return owner

    cap = np.bincount(np.arange(raw.U) % shards, minlength=shards)
    social = None
    workload_order = None
    if method in ('sorecremover_partition', MOE_METHOD, GRAPH_MOE_METHOD):
        # Compare user preferences and the preferences exposed by social neighbors.
        edges = raw.social
        adjacency = sp.csr_matrix((np.ones(len(edges)), edges.T), shape=(raw.U, raw.U))
        adjacency = adjacency.maximum(adjacency.T)
        social = normalize(adjacency, norm='l1')
        preferences = normalize(features)
        context = normalize(social @ preferences)
        views = [preferences, math.sqrt(SOCIAL_WEIGHT) * context]
        if method == GRAPH_MOE_METHOD:
            if ablation != 'one_hop_partition':
                views.append(.5 * normalize(social @ (social @ preferences)))
            if ablation != 'user_balance_only':
                interaction_count = np.bincount(raw.train[:, 0], minlength=raw.U)
                social_degree = np.asarray(adjacency.sum(axis=1)).ravel()
                workload_order = np.argsort(interaction_count + social_degree, kind='stable')
        features = np.concatenate(views, axis=1)
    centers = features[rng.choice(raw.U, shards, replace=False)]
    owner = None
    for _ in range(20):
        cost = ((features[:, None] - centers[None]) ** 2).sum(2)
        if social is not None and owner is not None:
            # Marginal loss in retained social neighbors if this user moves.
            neighbor_share = social @ np.eye(shards)[owner]
            cost += SOCIAL_WEIGHT * (neighbor_share.max(axis=1, keepdims=True) - neighbor_share)
        scores = -cost
        if method in ('ultrare_social', 'sorecremover_partition', MOE_METHOD, GRAPH_MOE_METHOD):
            # Entropic transport with prescribed uniform user mass and shard capacity.
            scale = max(float(np.median(cost)) * 0.1, 1e-6)
            kernel = np.exp(-cost / scale).clip(min=1e-20)
            column = np.ones(shards)
            for _ in range(50):
                row = 1 / (kernel @ column)
                column = cap / (kernel.T @ row)
            scores = row[:, None] * kernel * column
        updated = (capacity_assignment(scores, cap) if workload_order is None
                   else workload_balanced_assignment(scores, workload_order, shards))
        if np.array_equal(owner, updated):
            break
        owner = updated
        centers = np.stack([features[owner == shard].mean(0) for shard in range(shards)])
    return owner


def eemu_edge_shards(raw, shards, seed):
    """Randomly balance each training profile across interaction shards."""
    rng = np.random.default_rng(seed)
    order = np.argsort(raw.train[:, 0], kind='stable')
    breaks = np.flatnonzero(np.diff(raw.train[order, 0])) + 1
    assignments = np.empty(len(raw.train), dtype=np.int64)
    for group in np.split(order, breaks):
        if len(group):
            assignments[rng.permutation(group)] = (np.arange(len(group)) + rng.integers(shards)) % shards

    # Assign undirected social pairs together to keep reverse links in one shard.
    _, pair_mapping = np.unique(np.sort(raw.social, axis=1), axis=0, return_inverse=True)
    n_pairs = int(pair_mapping.max()) + 1 if len(pair_mapping) else 0
    social_assignments = np.empty(n_pairs, dtype=np.int64)
    social_assignments[rng.permutation(n_pairs)] = np.arange(n_pairs) % shards
    return [SimpleNamespace(**{
        **vars(raw),
        'train': raw.train[assignments == shard],
        'social': raw.social[social_assignments[pair_mapping] == shard],
    }) for shard in range(shards)]


def edge_shard_users(raw_shard):
    return np.union1d(raw_shard.train[:, 0], raw_shard.social.ravel())


def edge_shard_affected(raw_shards, deleted):
    return {shard for shard, raw_shard in enumerate(raw_shards)
            if np.isin(raw_shard.train[:, 0], deleted).any() or np.isin(raw_shard.social, deleted).any()}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def backbone_graph(raw, users, backbone, device):
    if backbone == 'DiffNet++':
        return make_graph(raw, users)
    return graph(raw, users, device)


def odsr_batches(data, rng, batch_size, device):
    order = rng.permutation(len(data.train))
    for start in range(0, len(order), batch_size):
        batch = data.train[order[start:start + batch_size]]
        u, p = torch.as_tensor(batch, device=device).T
        n = torch.as_tensor(negatives(data, batch[:, 0], rng), device=device)
        yield u, p, n


def diffnet_batches(data, rng, batch_size, device):
    examples = training_examples(data, rng, negative_draws=8)
    order = rng.permutation(sorted(examples))
    for start in range(0, len(order), batch_size):
        yield user_batch(examples, order[start:start + batch_size], device)


def fit_shard(raw, users, cfg, args, device, guide=None, active=None, training_log=None):
    seed_everything(args.seed)
    data = backbone_graph(raw, users, args.backbone, device)
    if guide is not None:
        guide_repair(data, guide, users, active, args.backbone, device)
    # Give independent shards a common data-independent item coordinate system.
    item_generator = torch.Generator(device=device).manual_seed(args.seed + 1_000_000)
    if args.backbone == 'DiffNet++':
        model = DiffNetPlus(data, 64, device)
        nn.init.normal_(model.item_embeddings, std=.01, generator=item_generator)
        batches = diffnet_batches
        loss_fn = model.loss
        batch_size = 512
        lr = .0005
    elif args.backbone == 'RecFlow':
        model = RecFlow(data, device, args.seed, item_generator)
        batches = odsr_batches
        loss_fn = model.loss
        batch_size = 4096
        lr = args.recflow_lr
    else:
        model = ODSR(data, cfg, device).to(device)
        nn.init.xavier_uniform_(model.embedding_dict['item_emb'], generator=item_generator)
        batches = odsr_batches
        loss_fn = lambda u, p, n: loss_terms(model, u, p, n)
        batch_size = cfg['batch_size']
        lr = cfg['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001 if args.backbone == "RecFlow" else 0.)
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.96) if args.backbone == "RecFlow" else None
    rng = np.random.default_rng(args.seed)
    best_score = -1
    best = None
    best_epoch = None
    steps = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in batches(data, rng, batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(*batch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite shard loss')
            loss.backward()
            optimizer.step()
            steps += 1
        if decay is not None:
            decay.step()
        if epoch == 1 or epoch % args.validation_every == 0 or epoch == args.epochs:
            metrics, (users_emb, items_emb, _) = evaluate(model, data, data.val, device)
            score = metrics.get('recall@20', 0)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best = (data.keep, users_emb.clone(), items_emb.clone())
                patience = 0
            else:
                patience += 1
            if patience >= 10:
                break
    if training_log is not None:
        training_log.update(epochs=epoch, best_epoch=best_epoch, optimizer_steps=steps,
                            best_validation_recall_at_20=best_score,
                            stopped_on_patience=patience >= 10)
    return best


class Embeddings(nn.Module):
    def __init__(self, users, items):
        super().__init__()
        self.users = users
        self.items = items

    def forward(self):
        return self.users, self.items


def shard_item_support(data, shard_rows):
    owners = np.empty(len(data.keep), dtype=np.int64)
    for shard, rows in enumerate(shard_rows):
        owners[rows] = shard
    train_users, train_items = data.train.T
    counts = np.bincount(
        owners[train_users] * data.item_num + train_items,
        minlength=len(shard_rows) * data.item_num,
    ).reshape(len(shard_rows), data.item_num)
    return owners, counts


def support_weighted_item_catalogs(data, shard_rows, items, counts=None):
    """Shrink each local catalog toward a consensus weighted by item evidence."""
    if counts is None:
        _, counts = shard_item_support(data, shard_rows)
    counts = torch.as_tensor(counts, device=items.device, dtype=items.dtype)
    evidence = counts.log1p()
    total_evidence = evidence.sum(0)
    consensus = (evidence.unsqueeze(-1) * items).sum(0) / total_evidence.clamp_min(1e-8).unsqueeze(-1)
    consensus = torch.where(total_evidence[:, None] > 0, consensus, items.mean(0))
    local_fraction = (counts / (counts + LOCAL_SUPPORT_PRIOR)).unsqueeze(-1)
    return torch.lerp(consensus.unsqueeze(0), items, local_fraction)


def moe_router_features(data, owners, counts, multihop=False):
    """Measure each expert's support using the retained training graph."""
    n_users, n_experts = len(owners), len(counts)
    train_users, train_items = data.train.T
    activity = np.bincount(train_users, minlength=n_users)
    user_item = sp.csr_matrix((np.ones(len(data.train)), (train_users, train_items)),
                              shape=(n_users, data.item_num))
    overlap = (user_item @ (counts > 0).T) / np.maximum(activity[:, None], 1)
    links = data.social if hasattr(data, 'social') else data.edges['uu']
    adjacency = sp.csr_matrix((np.ones(len(links)), links.T), shape=(n_users, n_users))
    membership = np.eye(n_experts)[owners]
    transition = normalize(adjacency.maximum(adjacency.T), norm='l1')
    social_share = transition @ membership
    log_activity = np.broadcast_to(np.log1p(activity[:, None]), overlap.shape)
    features = [overlap, social_share, membership, log_activity]
    if multihop:
        collaborative_share = user_item @ normalize(user_item.T @ membership, norm='l1')
        collaborative_share /= np.maximum(activity[:, None], 1)
        features.extend((transition @ social_share, collaborative_share))
    return np.stack(features, axis=-1)


def expert_inputs(parts, data, items):
    rows = [data.mapping[ids] for ids, _, _ in parts]
    owners, counts = shard_item_support(data, rows)
    users = items.new_empty((len(owners), items.shape[-1]))
    for user_rows, (_, embeddings, _) in zip(rows, parts):
        users[user_rows] = embeddings
    return rows, owners, counts, users


def fit_expert_router(features, users, catalogs, data, args, validation, load_weight=0.0):
    seed_everything(args.seed)
    router = nn.Sequential(nn.Linear(features.shape[-1], 16), nn.Tanh(), nn.Linear(16, 1)).to(features.device)
    nn.init.zeros_(router[-1].weight)
    nn.init.zeros_(router[-1].bias)
    owner_bias = math.log(MOE_OWNER_PRIOR * (len(catalogs) - 1) / (1 - MOE_OWNER_PRIOR))
    prior_logits = owner_bias * features[..., 2]
    log_prior = prior_logits.log_softmax(dim=1)
    optimizer = torch.optim.Adam(router.parameters(), lr=.01)
    rng = np.random.default_rng(args.seed)
    known_positives = [seen.copy() for seen in data.positives]
    for user, item in data.val:
        known_positives[user].add(int(item))
    negative_data = SimpleNamespace(item_num=data.item_num, positives=known_positives)
    eligible = np.array([len(seen) < data.item_num for seen in known_positives])
    validation = validation[eligible[validation[:, 0]]]

    for _ in range(args.aggregation_epochs):
        order = rng.permutation(len(validation))
        for start in range(0, len(order), 2048):
            batch = validation[order[start:start + 2048]]
            u, p = torch.as_tensor(batch, device=features.device).T
            n = torch.as_tensor(negatives(negative_data, batch[:, 0], rng), device=features.device)
            log_weights = (prior_logits[u] + router(features[u]).squeeze(-1)).log_softmax(dim=1)
            weights = log_weights.exp()
            expert_margins = ((catalogs[:, p] - catalogs[:, n]) * users[u]).sum(-1).T
            ranking = F.softplus(-(weights * expert_margins).sum(dim=1)).mean()
            regularizer = (weights * (log_weights - log_prior[u])).sum(dim=1).mean()
            usage = weights.mean(dim=0)
            load_penalty = (usage * len(catalogs) - 1).square().mean()
            optimizer.zero_grad(set_to_none=True)
            (ranking + MOE_KL_WEIGHT * regularizer + load_weight * load_penalty).backward()
            optimizer.step()

    with torch.no_grad():
        weights = (prior_logits + router(features).squeeze(-1)).softmax(dim=1)
    return weights, len(validation)


def combine_moe(parts, data, items, args, device):
    """Fit a user-to-expert router and preserve exact full-catalog dot products."""
    rows, owners, counts, users = expert_inputs(parts, data, items)
    catalogs = support_weighted_item_catalogs(data, rows, items, counts)
    if len(parts) == 1:
        return Embeddings(users, catalogs[0])

    features = torch.as_tensor(moe_router_features(data, owners, counts), device=device, dtype=items.dtype)
    weights, n_validation = fit_expert_router(features, users, catalogs, data, args, data.val)
    user_factors = (weights.unsqueeze(-1) * users.unsqueeze(1)).flatten(1)
    item_factors = catalogs.transpose(0, 1).reshape(data.item_num, -1)
    deployment = Embeddings(user_factors, item_factors)
    deployment.routing_diagnostics = {
        'mean_owner_weight': float((weights * features[..., 2]).sum(1).mean()),
        'mean_expert_weights': weights.mean(0).tolist(),
        'validation_pairs': n_validation,
    }
    return deployment


def align_expert_spaces(items, users, counts, owners):
    """Align supported shared items to a reference shard by orthogonal Procrustes."""
    reference = int(counts.sum(axis=1).argmax())
    dimension = items.shape[-1]
    identity = torch.eye(dimension, device=items.device, dtype=items.dtype)
    rotations = []
    for expert, expert_items in enumerate(items):
        overlap = np.minimum(counts[reference], counts[expert])
        anchors = np.flatnonzero(overlap)
        if expert == reference or len(anchors) < dimension:
            rotations.append(identity)
            continue
        weights = torch.as_tensor(np.log1p(overlap[anchors]), device=items.device, dtype=items.dtype)
        source = F.normalize(expert_items[anchors], dim=1)
        target = F.normalize(items[reference, anchors], dim=1)
        cross = (source.T * weights) @ target
        ridge = .05 * torch.linalg.matrix_norm(cross) / math.sqrt(dimension)
        left, _, right = torch.linalg.svd(cross + ridge * identity, full_matrices=False)
        rotations.append(left @ right)
    rotated_users = torch.empty_like(users)
    for expert, rotation in enumerate(rotations):
        selected = owners == expert
        rotated_users[selected] = users[selected] @ rotation
    return items @ torch.stack(rotations), rotated_users


class ExpertPair(nn.Module):
    def __init__(self, users, first, second, weights):
        super().__init__()
        self.users = (weights.unsqueeze(-1) * users.unsqueeze(1)).flatten(1)
        self.first = first
        self.second = second

    def forward(self):
        return self.users, torch.cat((self.first, self.second), dim=1)


class RoutedExperts(list):
    def __init__(self, groups, diagnostics):
        super().__init__(groups)
        self.routing_diagnostics = diagnostics


def group_expert_pairs(users, catalogs, owners, secondary, secondary_weights, strength):
    if strength == 0:
        groups = []
        for expert, catalog in enumerate(catalogs):
            rows = np.flatnonzero(owners == expert)
            groups.append((rows, Embeddings(users[rows], catalog)))
        return groups

    groups = []
    sorted_pairs = np.sort(np.column_stack((owners, secondary)), axis=1)
    pairs, membership = np.unique(sorted_pairs, axis=0, return_inverse=True)
    for group, (first, second) in enumerate(pairs):
        rows = np.flatnonzero(membership == group)
        secondary_weight = strength * secondary_weights[rows]
        owner_first = torch.as_tensor(owners[rows] == first, device=users.device)
        first_weight = torch.where(owner_first, 1 - secondary_weight, secondary_weight)
        weights = torch.stack((first_weight, 1 - first_weight), dim=1)
        groups.append((rows, ExpertPair(users[rows], catalogs[first], catalogs[second], weights)))
    return groups


def combine_graph_moe(parts, data, items, args, device):
    rows, owners, counts, users = expert_inputs(parts, data, items)
    if len(parts) == 1:
        return Embeddings(users, items[0])
    items, users = align_expert_spaces(items, users, counts, owners)
    catalogs = support_weighted_item_catalogs(data, rows, items, counts)
    ablation = getattr(args, 'ablation', None)
    if ablation == 'owner_only':
        return [(part_rows, Embeddings(users[part_rows], catalogs[expert]))
                for expert, part_rows in enumerate(rows)]
    features = torch.as_tensor(moe_router_features(data, owners, counts,
                                                  multihop=ablation != 'local_router'),
                               device=device, dtype=items.dtype)
    rng = np.random.default_rng(args.seed + 31000)
    val_users = rng.permutation(np.unique(data.val[:, 0]))
    if len(val_users) < 2:
        return [(part_rows, Embeddings(users[part_rows], catalogs[expert]))
                for expert, part_rows in enumerate(rows)]
    calibration_mask = np.isin(data.val[:, 0], val_users[:max(1, len(val_users) // 5)])
    fit_data, calibration = data.val[~calibration_mask], data.val[calibration_mask]
    load_weight = 0.0 if ablation == 'no_load_regularizer' else GRAPH_MOE_LOAD_WEIGHT
    weights, validation_pairs = fit_expert_router(
        features, users, catalogs, data, args, fit_data, load_weight=load_weight)
    alternate_weight, secondary = weights.masked_fill(features[..., 2].bool(), -1).max(dim=1)
    home_weight = (weights * features[..., 2]).sum(dim=1)
    secondary_weights = alternate_weight / (alternate_weight + home_weight)
    secondary = secondary.cpu().numpy()
    candidates = {}
    for strength in (() if ablation == 'no_calibration' else (0.0, 0.5, 1.0)):
        groups = group_expert_pairs(users, catalogs, owners, secondary, secondary_weights, strength)
        metrics = evaluate_deployment(groups, data, device, pairs=calibration)
        candidates[strength] = metrics['recall@20']
    strength = max(candidates, key=lambda value: (candidates[value], -value)) if candidates else 1.0
    groups = group_expert_pairs(users, catalogs, owners, secondary, secondary_weights, strength)
    return RoutedExperts(groups, {
        'mixing_strength': strength,
        'mean_owner_weight': float((1 - strength * secondary_weights).mean()),
        'router_training_pairs': validation_pairs,
        'calibration_users': len(np.unique(calibration[:, 0])),
        'calibration_recall_at_20': candidates,
        'max_active_experts': 1 if strength == 0 else 2,
    })


def evaluate_deployment(model, data, device, pairs=None):
    pairs = data.test if pairs is None else pairs
    if isinstance(model, Embeddings):
        metrics, _ = evaluate(model, data, pairs, device)
        return metrics

    totals = {}
    evaluated = 0
    for rows, shard_model in model:
        mapping = np.full(len(data.keep), -1, dtype=np.int64)
        mapping[rows] = np.arange(len(rows))
        shard_test = pairs[mapping[pairs[:, 0]] >= 0].copy()
        shard_test[:, 0] = mapping[shard_test[:, 0]]
        shard_data = SimpleNamespace(
            item_num=data.item_num,
            positives=[data.positives[row] for row in rows],
        )
        metrics, _ = evaluate(shard_model, shard_data, shard_test, device)
        count = metrics.pop('evaluated_users')
        evaluated += count
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + count * value
    return {**{name: value / evaluated for name, value in totals.items()}, 'evaluated_users': evaluated}


def combine(parts, data, method, args, device, guide=None):
    items = torch.stack([part[2] for part in parts])
    if method == GRAPH_MOE_METHOD:
        return combine_graph_moe(parts, data, items, args, device)
    if method == MOE_METHOD:
        return combine_moe(parts, data, items, args, device)
    if method == 'eemu_partition':
        # A dot product in concatenated shard coordinates gives mean shard scores.
        users = items.new_zeros((len(data.keep), len(parts), items.shape[-1]))
        for shard, (ids, embedding, _) in enumerate(parts):
            users[data.mapping[ids], shard] = embedding
        combined_items = items.transpose(0, 1).reshape(data.item_num, -1) / args.shards
        return Embeddings(users.flatten(1), combined_items)

    if method == 'sorecremover_partition':
        rows = [data.mapping[ids] for ids, _, _ in parts]
        catalogs = support_weighted_item_catalogs(data, rows, items)
        return [(part_rows, Embeddings(part[1], catalog))
                for part_rows, part, catalog in zip(rows, parts, catalogs)]

    users = torch.empty((len(data.keep), items.shape[-1]), device=device)
    for ids, embedding, _ in parts:
        users[data.mapping[ids]] = embedding

    if method == 'guide_social':
        weights = torch.as_tensor(guide_weights(guide, parts, data.keep, args.seed), device=device, dtype=items.dtype)
        return Embeddings(users, (weights[:, None, None] * items).sum(0))

    if method == 'ultrare_social' or len(parts) == 1:
        return Embeddings(users, items.mean(0))

    seed_everything(args.seed)
    if method == 'receraser_social':
        attention = nn.Sequential(nn.Linear(items.shape[-1], 32), nn.ReLU(), nn.Linear(32, 1)).to(device)

        def item_vectors(ids):
            vectors = items[:, ids]
            weights = attention(vectors).softmax(dim=0)
            return (weights * vectors).sum(0)
    else:
        attention = nn.Embedding(1, len(parts)).to(device)
        nn.init.zeros_(attention.weight)

        def item_vectors(ids):
            weights = attention.weight.flatten().softmax(dim=0)
            return (weights[:, None, None] * items[:, ids]).sum(0)

    optimizer = torch.optim.Adam(attention.parameters(), lr=0.01)
    rng = np.random.default_rng(args.seed)
    val_seen = [positive.copy() for positive in data.positives]
    for user, item in data.val:
        val_seen[user].add(int(item))
    sampler_data = SimpleNamespace(item_num=data.item_num, positives=val_seen)
    for _ in range(args.aggregation_epochs):
        order = rng.permutation(len(data.val))
        for start in range(0, len(order), 2048):
            batch = data.val[order[start:start + 2048]]
            u, p = torch.as_tensor(batch, device=device).T
            n = torch.as_tensor(negatives(sampler_data, batch[:, 0], rng), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.softplus(-(users[u] * (item_vectors(p) - item_vectors(n))).sum(1)).mean()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return Embeddings(users, item_vectors(slice(None)))


def run(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if min(args.epochs, args.aggregation_epochs, args.validation_every) < 1:
        raise ValueError('Epoch budgets and validation interval must be positive')
    ablation = getattr(args, 'ablation', None)
    router_ablations = getattr(args, 'router_ablations', False)
    if (ablation or router_ablations) and args.methods != [GRAPH_MOE_METHOD]:
        raise ValueError('Ablations require GraphMoE as the sole method')
    raw = load_data(args.dataset, args.seed)
    n_delete = max(1, round(raw.U * args.fraction))
    if not 0 < args.fraction < 1 or n_delete >= raw.U or not 2 <= args.shards <= raw.U:
        raise ValueError('Need a nonempty retained graph and at least two valid shards')
    deleted = np.sort(np.random.default_rng(args.seed + 10000).permutation(raw.U)[:n_delete])
    keep = np.setdiff1d(np.arange(raw.U), deleted)
    dataset_config = {

        'yelp2018': dict(eps=.54, ssl_reg=.43, dropout=.17),
        'douban-book': dict(eps=.57, ssl_reg=.58, dropout=.01),
    }
    cfg = dict(seed=args.seed, lr=.001, emb_size=64, n_layer=3, tau=.2, reg=0.,
               batch_size=2048, **(dataset_config[args.dataset] if args.backbone == 'ODSR' else {}))
    results = dict(backbone=args.backbone, dataset=args.dataset, seed=args.seed, fraction=args.fraction,
                   shards=args.shards, epochs=args.epochs, aggregation_epochs=args.aggregation_epochs,
                   validation_every=args.validation_every, split_hash=raw.meta['split_hash'],
                   deleted_users=len(deleted), deletion_hash=digest(deleted), social_hash=raw.meta['social_hash'],
                   gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   device=str(device),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   completion='partial', methods={})
    if args.backbone == 'RecFlow':
        results['backbone_source_sha256'] = hashlib.sha256((ROOT/'src/social_unlearning/recflow.py').read_bytes()).hexdigest()
        results['backbone_recipe'] = dict(implementation='paper-based reproduction', dimension=64, layers=2, learning_rate=args.recflow_lr, decay=.96, batch_size=4096, regularization=.001, time_dimension=8)
    if ablation:
        results['ablation'] = ablation
    if args.backbone == 'DiffNet++':
        implementation = ROOT / 'src/social_unlearning/diffnet_plus.py'
        results['backbone_source_sha256'] = hashlib.sha256(implementation.read_bytes()).hexdigest()
        results['backbone_recipe'] = dict(dimension=64, learning_rate=.0005, users_per_batch=512, negative_draws=8)
    for method in ('retrain', *METHODS, MOE_METHOD, GRAPH_MOE_METHOD):
        if method not in args.methods:
            continue
        print('START', args.dataset, args.seed, method, flush=True)
        synchronize(device)
        setup = time.perf_counter()
        before = None
        original_training = []
        update_training = []
        guide = None
        raw_shards = None
        if method == 'retrain':
            owner = np.zeros(raw.U, dtype=np.int64)
            parts = 1
            partition_seconds = 0.0
        elif method == 'eemu_partition':
            raw_shards = eemu_edge_shards(raw, args.shards, args.seed)
            parts = args.shards
            partition_seconds = time.perf_counter() - setup
        elif method == 'guide_social':
            guide = social_adjacency(raw)
            parts = args.shards
            owner = guide_partition(raw, guide, parts, args.seed)
            partition_seconds = time.perf_counter() - setup
        else:
            features = None
            if method != 'grapheraser_social':
                ui = sp.csr_matrix((np.ones(len(raw.train)), raw.train.T), shape=(raw.U, raw.I))
                features = TruncatedSVD(n_components=min(32, raw.I - 1), random_state=args.seed).fit_transform(ui)
            owner = partition(raw, method, args.shards, args.seed, features, ablation)
            parts = args.shards
            partition_seconds = time.perf_counter() - setup
        if raw_shards is None:
            original_users = [np.flatnonzero(owner == shard) for shard in range(parts)]
            raw_shards = [raw] * parts
        else:
            original_users = [edge_shard_users(raw_shard) for raw_shard in raw_shards]
        if method != 'retrain':
            all_users = np.arange(raw.U)
            before = []
            for shard, (raw_shard, ids) in enumerate(zip(raw_shards, original_users)):
                if not len(ids):
                    before.append(None)
                    continue
                training = {'shard': shard, 'users': len(ids)}
                before.append(fit_shard(raw_shard, ids, cfg, args, device, guide, all_users, training))
                original_training.append(training)
            combine([part for part in before if part is not None],
                    backbone_graph(raw, all_users, args.backbone, device), method, args, device, guide)
        synchronize(device)
        setup_seconds = time.perf_counter() - setup if before is not None else None
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        elapsed_start = time.perf_counter()
        if method == 'eemu_partition':
            affected = edge_shard_affected(raw_shards, deleted)
        elif guide is not None:
            affected = guide_affected_shards(guide, owner, deleted)
        else:
            affected = set(owner[deleted])
        after = []
        for shard, (users, raw_shard) in enumerate(zip(original_users, raw_shards)):
            retained = np.setdiff1d(users, deleted)
            if method == 'eemu_partition' and shard in affected:
                train = raw_shard.train[~np.isin(raw_shard.train[:, 0], deleted)]
                social = raw_shard.social[~np.isin(raw_shard.social, deleted).any(1)]
                raw_shard = SimpleNamespace(**{**vars(raw_shard), 'train': train, 'social': social})
                retained = edge_shard_users(raw_shard)
            if len(retained):
                if shard in affected:
                    training = {'shard': shard, 'users': len(retained)}
                    after.append(fit_shard(raw_shard, retained, cfg, args, device, guide, keep, training))
                    update_training.append(training)
                else:
                    after.append(before[shard])
        retained_data = backbone_graph(raw, keep, args.backbone, device)
        if router_ablations:
            synchronize(device)
            expert_update_seconds = time.perf_counter() - elapsed_start
        model = combine(after, retained_data, method, args, device, guide)
        synchronize(device)
        update_seconds = time.perf_counter() - elapsed_start
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        start = time.perf_counter()
        metrics = evaluate_deployment(model, retained_data, device)
        row = dict(utility=metrics, partition_seconds=partition_seconds, original_setup_seconds=setup_seconds,
                   update_seconds=update_seconds, test_seconds=time.perf_counter() - start,
                   peak_gpu_bytes=peak, affected_shards=len(affected),
                   shard_users=[len(ids) for ids in original_users],
                   original_training=original_training, update_training=update_training)
        if method == 'eemu_partition':
            row['shard_interactions'] = [len(raw_shard.train) for raw_shard in raw_shards]
            row['shard_social_edges'] = [len(raw_shard.social) for raw_shard in raw_shards]
            row['method_config'] = dict(partitioner='random per-user interaction edge sharding',
                                        social='disjoint random undirected pairs',
                                        aggregation='mean shard scores; absent shard-user contributes zero')
        else:
            row['within_shard_social_edge_fraction'] = float(np.mean(owner[raw.social[:, 0]] == owner[raw.social[:, 1]]))
        if guide is not None:
            row['method_config'] = dict(partitioner='sparse GPFB-Fast with activity quotas',
                                        repair='mirror social neighbors',
                                        aggregation='normalized pyramid match on retained social graphs')
        if method in ('sorecremover_partition', MOE_METHOD, GRAPH_MOE_METHOD):
            row['method_config'] = dict(social_weight=SOCIAL_WEIGHT, local_support_prior=LOCAL_SUPPORT_PRIOR)
        if method in (MOE_METHOD, GRAPH_MOE_METHOD):
            row['method_config'].update(owner_prior=MOE_OWNER_PRIOR, router_kl_weight=MOE_KL_WEIGHT)
            row['routing'] = getattr(model, 'routing_diagnostics', {})
        if method == GRAPH_MOE_METHOD:
            row['method_config'].update(router_load_weight=GRAPH_MOE_LOAD_WEIGHT)
            row['shard_interactions'] = [int(np.count_nonzero(owner[raw.train[:, 0]] == shard))
                                         for shard in range(parts)]
        if router_ablations:
            variants = ('owner_only', 'local_router', 'no_load_regularizer', 'no_calibration')
            row['expert_update_seconds'] = expert_update_seconds
            row['router_ablations'] = {}
            for variant in variants:
                start = time.perf_counter()
                variant_args = SimpleNamespace(**{**vars(args), 'ablation': variant})
                variant_model = combine(after, retained_data, method, variant_args, device, guide)
                synchronize(device)
                aggregation_seconds = time.perf_counter() - start
                row['router_ablations'][variant] = dict(
                    utility=evaluate_deployment(variant_model, retained_data, device),
                    aggregation_seconds=aggregation_seconds,
                    update_seconds=expert_update_seconds + aggregation_seconds,
                    routing=getattr(variant_model, 'routing_diagnostics', {}),
                )
        results['methods'][method] = row
        atomic_json(args.output, results)
        print(method, row, flush=True)
    results['completion'] = 'complete'
    atomic_json(args.output, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['ODSR', 'RecFlow'], default='ODSR')
    parser.add_argument('--dataset', choices=['yelp2018', 'douban-book', 'recflow-yelp'], required=True)
    parser.add_argument('--recflow-lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fraction', type=float, default=.01)
    parser.add_argument('--shards', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--aggregation-epochs', type=int, default=10)
    parser.add_argument('--validation-every', type=int, default=1)
    parser.add_argument('--methods', nargs='+', choices=('retrain', *METHODS, MOE_METHOD, GRAPH_MOE_METHOD),
                        default=('retrain', *METHODS))
    parser.add_argument('--ablation', choices=('one_hop_partition', 'user_balance_only'))
    parser.add_argument('--router-ablations', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
