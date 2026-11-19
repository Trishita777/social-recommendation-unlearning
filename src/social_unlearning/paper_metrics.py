"""DiffNet++ sampled HR/NDCG protocol, separate from full-catalog metrics.

HR uses hits/min(number of positives,K), exactly the released evaluator's
definition. Five independent 1000-negative candidate draws are shared by all
methods. Our negatives are distinct, exclude ALL observed interactions, and
all users are evaluated: explicitly documented corrections to released code's
replacement sampler / incomplete exclusions / Python-2 batching.
"""
from collections import defaultdict
import numpy as np
import torch


def ranking_metrics(positive_scores, negative_scores, cutoffs=(5, 10, 15)):
    scores = np.concatenate([positive_scores, negative_scores])
    # Deterministic tie handling matching the source evaluator's numpy argsort.
    ranked = np.argsort(scores)[::-1]
    hit = ranked < len(positive_scores)
    result = {}
    for k in cutoffs:
        denominator = min(len(positive_scores), k)
        result[f'hr@{k}'] = float(hit[:k].sum()/denominator)
        result[f'ndcg@{k}'] = float((hit[:k]/np.log2(np.arange(2, len(hit[:k])+2))).sum()
                                  / (1/np.log2(np.arange(2, denominator+2))).sum())
    return result


def candidate_bank(raw, pairs, keep, seed, repetitions=5, negatives=1000):
    observed = defaultdict(set)
    for split in (raw.train, raw.val, raw.test):
        for u, i in split: observed[int(u)].add(int(i))
    truth = defaultdict(set)
    for u, i in pairs: truth[int(u)].add(int(i))
    keep = set(map(int, keep))
    users = sorted(set(truth) & keep)
    positive = {u: np.array(sorted(truth[u]), dtype=np.int64) for u in users}
    banks = []
    for repetition in range(repetitions):
        bank = {}
        for u in users:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 93871, repetition, u]))
            if raw.I-len(observed[u]) < negatives:
                raise ValueError(f'User {u} has fewer than {negatives} unobserved items')
            # Rejection sampling avoids building the entire catalog for each user.
            chosen, seen = [], set(observed[u])
            while len(chosen) < negatives:
                for i in rng.integers(raw.I, size=max(64, int((negatives-len(chosen))*1.1))):
                    i = int(i)
                    if i not in seen:
                        seen.add(i); chosen.append(i)
                        if len(chosen) == negatives: break
            bank[u] = np.asarray(chosen, dtype=np.int64)
        banks.append(bank)
    return users, positive, banks


@torch.no_grad()
def evaluate_sampled(embeddings, data, bank):
    users, positive, banks = bank
    uvec, ivec = embeddings
    repetitions = []
    for draw in banks:
        totals = defaultdict(float)
        for start in range(0, len(users), 128):
            batch = users[start:start+128]
            ids = np.stack([draw[u] for u in batch])
            uv = uvec[torch.as_tensor(data.mapping[batch], device=uvec.device)]
            # Raw dot and sigmoid induce identical order except numerical saturation;
            # use sigmoid as released code does, including deterministic ties.
            ns = (uv[:, None, :] * ivec[torch.as_tensor(ids, device=ivec.device)]).sum(-1).sigmoid().cpu().numpy()
            for j, u in enumerate(batch):
                ps = (uv[j] * ivec[torch.as_tensor(positive[u], device=ivec.device)]).sum(-1).sigmoid().cpu().numpy()
                for key, value in ranking_metrics(ps, ns[j]).items(): totals[key] += value
        repetitions.append({k: v/len(users) for k, v in totals.items()})
    return dict(metrics={k: float(np.mean([r[k] for r in repetitions])) for k in repetitions[0]},
                repetition_metrics=repetitions, evaluated_users=len(users),
                protocol='DiffNet++ HR/NDCG@5,10,15; five fixed draws of 1000 distinct all-unobserved negatives/user; released HR denominator; same candidates across methods')
