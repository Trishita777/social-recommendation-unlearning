"""Identity protocol and metrics for a controlled user-membership MIO target."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def target_partition(user_count, manifest, seed, fraction=.01):
    """Use only identities entirely excluded from the already fitted shadow bank."""
    if not 0 < fraction < .5:
        raise ValueError('fraction must be between zero and one half')
    excluded = np.asarray(manifest['excluded_target_users'], dtype=np.int64)
    if len(np.unique(excluded)) != len(excluded) or np.any(excluded < 0) or np.any(excluded >= user_count):
        raise ValueError('Invalid shadow excluded-user universe')
    forbidden = np.concatenate([np.asarray(ids, dtype=np.int64) for ids in manifest['roles'].values()]
                               + [np.asarray(manifest['shadow_member_users'], dtype=np.int64)])
    if np.intersect1d(excluded, forbidden).size:
        raise ValueError('Shadow exclusion overlaps fitting/evaluation or graph members')
    count = max(1, round(user_count * fraction))
    if len(excluded) < 2 * count:
        raise ValueError('Shadow-excluded pool too small; provision a new disjoint shadow bank')
    order = np.random.default_rng(seed + 20000).permutation(excluded)
    deleted, nonmembers = np.sort(order[:count]), np.sort(order[count:2 * count])
    return deleted, nonmembers


def validate_graph_membership(data, deleted, nonmembers, *, before_deletion):
    """Assert full row space, actual member edges, and both-direction exclusion."""
    absent = np.asarray(nonmembers) if before_deletion else np.union1d(deleted, nonmembers)
    if not np.array_equal(data.keep, np.arange(data.user_num)):
        raise ValueError('Controlled target must preserve the full ID universe')
    if np.isin(data.train[:, 0], absent).any() or np.isin(data.social, absent).any():
        raise ValueError('Never-member/deleted incident edges remain')
    if before_deletion and not np.isin(deleted, np.unique(data.train[:, 0])).all():
        raise ValueError('Every requested user must be an actual original training member')
    for adjacency in (data.adj_uu, data.norm_adj):
        indices = adjacency.coalesce().indices().detach().cpu().numpy()
        if np.isin(indices, absent).any():
            raise ValueError('Absent users have normalized graph edges or synthetic self loops')


def attack_metrics(labels, probabilities, threshold=.5):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError('A target audit requires both actual membership classes')
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite attack probabilities')
    predicted = probabilities >= threshold
    return dict(auc=float(roc_auc_score(labels, probabilities)),
                balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
                tpr=float(predicted[labels == 1].mean()), fpr=float(predicted[labels == 0].mean()),
                threshold=float(threshold), members=int(labels.sum()), nonmembers=int((labels == 0).sum()),
                mean_member_probability=float(probabilities[labels == 1].mean()),
                mean_nonmember_probability=float(probabilities[labels == 0].mean()))


def bootstrap_attack(labels, probabilities, threshold=.5, seed=1, samples=1000):
    """Stratified whole-user bootstrap; never resample individual interactions."""
    labels = np.asarray(labels); probabilities = np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == c) for c in (0, 1)]
    if any(len(ids) == 0 for ids in groups):
        raise ValueError('Both membership classes required for bootstrap')
    values = []
    for _ in range(samples):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        result = attack_metrics(labels[ids], probabilities[ids], threshold)
        values.append([result['auc'], result['balanced_accuracy']])
    intervals = np.quantile(values, [.025, .975], axis=0)
    return dict(auc_95ci=intervals[:, 0].tolist(), balanced_accuracy_95ci=intervals[:, 1].tolist(),
                bootstrap='stratified whole-user', bootstrap_samples=samples)
