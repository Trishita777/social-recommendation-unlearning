"""Publish the main comparison and stage-wise ablations when all runs finish."""
import json
import time

from run_partition_epoch_sweep import ROOT, SEEDS, save

RESULTS = ROOT / 'results/partition_social'
MAIN = RESULTS / 'douban_yelp_epoch30_shards10'
ABLATIONS = RESULTS / 'graph_moe_ablation30_shards10'
PAPER_CASES = (('ODSR', 'yelp2018'),)
METHOD = 'sorecremover_graph_moe'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def collect():
    main = read(MAIN / 'summary.json') or read(MAIN / 'partial_summary.json')
    ablations = {}
    for backbone, dataset in PAPER_CASES:
        for seed in SEEDS:
            labels = ['one_hop_partition', 'user_balance_only']
            if seed == 42:
                labels.extend(('shards20', 'shards50'))
            key = f'{backbone}/{dataset}/{seed}'
            full = main.get('runs', {}).get(f'{key}/{METHOD}')
            entries = {}
            if full:
                entries['full'] = full['methods'][METHOD]
            for label in labels:
                path = ABLATIONS / dataset / f'seed{seed}_{label}.json'
                run = read(path)
                if run.get('completion') != 'complete':
                    continue
                expected_shards = int(label.removeprefix('shards')) if label.startswith('shards') else 10
                expected = (backbone, dataset, seed, expected_shards, 30, 2, .01)
                fields = ('backbone', 'dataset', 'seed', 'shards', 'epochs', 'validation_every', 'fraction')
                if tuple(run.get(field) for field in fields) != expected:
                    raise ValueError(f'Unexpected ablation protocol in {path}')
                if full and any(run[field] != full[field] for field in ('split_hash', 'deletion_hash', 'social_hash')):
                    raise ValueError(f'Ablation and full model use different data in {path}')
                entries[label] = run['methods'][METHOD]
            ablations[key] = entries

    component_runs = sum(len(set(rows) - {'full'}) for rows in ablations.values())
    router_runs = sum(bool(rows.get('full', {}).get('router_ablations')) for rows in ablations.values())
    complete = bool(main.get('complete')) and component_runs == 8 and router_runs == 3
    report = dict(complete=complete, main=main, ablations=ablations,
                  additional_ablation_runs=component_runs, router_ablation_cases=router_runs,
                  updated=time.time())
    save(MAIN / ('all_results.json' if complete else 'all_results_partial.json'), report)
    print(f"Main {main.get('completed', 0)}/72, ablation runs {component_runs}/8, router cases {router_runs}/3",
          flush=True)
    return complete


if __name__ == '__main__':
    while not collect():
        time.sleep(45)
