"""Read-only diagnostics for a completed shadow MIO bank; no target tuning."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

from run_unlearning import ODSR, atomic_json, load_data
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.benchmark_mio import BenchmarkMIO, embedding_features


def run(bank, dataset, device):
    torch.set_num_threads(4)
    manifest = json.loads((bank/'manifest.json').read_text())
    raw = load_data(dataset, 42)
    members = np.array(manifest['shadow_member_users'])
    absent = np.setdiff1d(np.arange(raw.U), members)
    roles = {k: np.array(v) for k, v in manifest['roles'].items()}
    for name, ids in roles.items():
        assert not np.intersect1d(ids, manifest['excluded_target_users']).size
        for other, other_ids in roles.items():
            if name != other:
                assert not np.intersect1d(ids, other_ids).size
    data = edge_graph(raw, absent, device)
    assert not np.isin(data.train[:, 0], absent).any()
    assert not np.isin(data.social, absent).any()
    ck = torch.load(bank/'shadow_original/deployment.pt', map_location='cpu', weights_only=False)
    model = ODSR(data, ck['config'], device).to(device)
    model.load_state_dict(ck['model']); model.eval()
    with torch.no_grad():
        users, items = model(); users = users.cpu(); items = items.cpu()
    history = defaultdict(list)
    for u, i in np.concatenate([raw.train, raw.val]):
        history[int(u)].append(int(i))
    attacker_ck = torch.load(bank/'attacker.pt', map_location='cpu', weights_only=False)
    attacker = BenchmarkMIO(users.shape[1]*2)
    attacker.load_state_dict(attacker_ck['model']); attacker.eval()
    scores, labels = {}, {}
    report = {'scope': 'diagnostic only; original frozen attacker; no target tuning',
              'identity_and_graph_checks': 'passed', 'roles': {}}
    for role, ids in roles.items():
        x = embedding_features(users, items, ids, {u: u for u in range(raw.U)}, history)
        scores[role] = attacker.member_probability(x).numpy()
        labels[role] = np.isin(ids, members).astype(int)
        y, p = labels[role], scores[role]
        report['roles'][role] = {
            'auc': float(roc_auc_score(y, p)),
            'balanced_accuracy_at_0.5': float(balanced_accuracy_score(y, p >= .5)),
            'classes': {str(c): {
                'score_quantiles': np.quantile(p[y == c], [0, .25, .5, .75, 1]).tolist(),
                'mean_score': float(p[y == c].mean()),
                'mean_user_embedding_norm': float(users[ids[y == c]].norm(dim=1).mean()),
            } for c in [0, 1]},
        }
    fpr, tpr, thresholds = roc_curve(labels['validation'], scores['validation'])
    threshold = float(thresholds[np.argmax(tpr-fpr)])
    report['validation_selected_threshold'] = threshold
    report['test_balanced_accuracy_at_validation_threshold'] = float(
        balanced_accuracy_score(labels['test'], scores['test'] >= threshold))
    report['training_best_epoch'] = attacker_ck['training']['best_epoch']
    report['final_validation_loss'] = attacker_ck['training']['history'][-1]['validation_loss']
    report['constant_half_probability_loss'] = float(np.log(2))
    atomic_json(bank/'diagnostics.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run(args.bank, args.dataset, torch.device(args.device))
