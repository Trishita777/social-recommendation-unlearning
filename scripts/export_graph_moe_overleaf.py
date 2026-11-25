"""Render the verified 30-epoch partition benchmark as a manuscript input."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/partition_social/douban_yelp_epoch30_shards10'
METHODS = {
    'retrain': 'Full retraining',
    'receraser_social': 'RecEraser-social',
    'ultrare_social': 'UltraRE-social',
    'grapheraser_social': 'GraphEraser-social',
    'guide_social': 'GUIDE-Fast-social',
    'eemu_partition': 'EEMU partition control',
    'sorecremover_partition': 'Partitioned SoRecRemover',
    'sorecremover_graph_moe': 'SoRecRemover-GraphMoE',
}
VARIANTS = {
    'one_hop_partition': 'One-hop partitioning',
    'user_balance_only': 'Equal user capacity',
    'owner_only': 'Owner-only scoring',
    'local_router': 'Local router features',
    'no_load_regularizer': 'Without load regularization',
    'no_calibration': 'Fixed secondary mixing',
}
DATASETS = {'douban-book': 'Douban-Book', 'yelp2018': 'Yelp2018'}
GRAPH_MOE = 'sorecremover_graph_moe'
RANKING = ('recall@20', 'ndcg@20')


def cell(values, digits):
    center = f'{mean(values):.{digits}f}'
    if len(values) == 1:
        return f'${center}$'
    return f'${center} \\pm {stdev(values):.{digits}f}$'


def row(*values):
    return ' & '.join(map(str, values)) + r' \\'


def table(caption, columns, rows):
    return '\n'.join((
        r'\begin{table}[htbp]', r'\centering\small', f'\\caption{{{caption}}}',
        f'\\begin{{tabular}}{{l{"r" * (len(columns) - 1)}}}', r'\toprule',
        row(*columns), r'\midrule', *rows, r'\bottomrule',
        r'\end{tabular}', r'\end{table}',
    ))


def render(report):
    main = report['main']
    parts = [
        r'\section{Partition-Based Unlearning with GraphMoE}',
        r'We evaluate 1\% user withdrawal on Douban-Book and Yelp2018 with ODSR and DiffNet++. '
        r'All partition methods use 10 shards. Each fit uses up to 30 epochs, validation at '
        r'epoch 1 and every two epochs thereafter, and the checkpoint with the best validation '
        r'Recall@20. Full retraining uses a single model. This study measures retained-user '
        r'ranking and update cost. '
        r'Update time excludes initial model fitting and final test evaluation. Timings were '
        r'measured on shared H100 NVL GPUs and may reflect host and device contention.',
        f'The verified snapshot contains {main["completed"]} of {main["total"]} benchmark runs, '
        f'{report["additional_ablation_runs"]} of 16 additional ablation runs, and '
        f'{report["router_ablation_cases"]} of 6 router cases. '
        r'Each row reports its seed count. Entries with multiple seeds give the mean and sample '
        r'standard deviation. The three benchmark seeds are 42, 43, and 44.',
        r'\subsection{Main comparison}',
    ]
    for backbone, datasets in main['aggregate'].items():
        for dataset, methods in datasets.items():
            rows = []
            for method, label in METHODS.items():
                scores = methods.get(method)
                if scores is None:
                    continue
                values = []
                for metric, digits in (('recall@10', 4), ('ndcg@10', 4), ('recall@20', 4),
                                       ('ndcg@20', 4), ('update_seconds', 1)):
                    entry = scores[metric]
                    center = f'{entry["mean"]:.{digits}f}'
                    sd = entry['sample_sd']
                    values.append(f'${center}\\pm {sd:.{digits}f}$' if sd is not None else f'${center}$')
                rows.append(row(label, scores['n'], *values))
            parts.append(table(
                f'{backbone} on {DATASETS[dataset]} after withdrawal.',
                ('Method', '$n$', 'Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20', 'Update (s)'),
                rows,
            ))
    parts.append(r'\clearpage\subsection{Partitioning and routing ablations}')
    for backbone, dataset in (('ODSR', 'yelp2018'), ('DiffNet++', 'douban-book')):
        cases = [report['ablations'][f'{backbone}/{dataset}/{seed}'] for seed in (42, 43, 44)]
        rows = []
        for variant, label in {'full': 'Full GraphMoE', **VARIANTS}.items():
            pairs = []
            for case in cases:
                full = case.get('full')
                if full is None:
                    continue
                control = case.get(variant) or full.get('router_ablations', {}).get(variant)
                if control is not None:
                    pairs.append((full, control))
            if pairs:
                rows.append(row(
                    label, len(pairs),
                    *(cell([b['utility'][key] for _, b in pairs], 4) for key in RANKING),
                    *(cell([b['utility'][key] - a['utility'][key] for a, b in pairs], 4) for key in RANKING),
                    cell([b['update_seconds'] for _, b in pairs], 1),
                ))
        parts.append(table(
            f'{backbone} on {DATASETS[dataset]}. Deltas are paired against the full variant. '
            'Routing controls share the trained experts.',
            ('Variant', '$n$', 'Recall@20', 'NDCG@20', r'$\Delta$ Recall', r'$\Delta$ NDCG', 'Update (s)'),
            rows,
        ))
    parts.append(r'\subsection{Shard-count sensitivity}')
    rows = []
    for backbone, dataset in (('ODSR', 'yelp2018'), ('DiffNet++', 'douban-book')):
        case = report['ablations'][f'{backbone}/{dataset}/42']
        for shards, variant in ((10, 'full'), (20, 'shards20'), (50, 'shards50')):
            run = case.get(variant)
            if run is not None:
                rows.append(row(
                    f'{backbone} / {DATASETS[dataset]}', shards,
                    *(f'{run["utility"][key]:.4f}' for key in RANKING),
                    f'{run["update_seconds"]:.1f}', run['affected_shards'],
                    f'{run["within_shard_social_edge_fraction"]:.3f}',
                ))
    parts.append(table(
        'Shard-count sensitivity for seed 42. Social retention is the fraction of social edges within shards.',
        ('Backbone / dataset', 'Shards', 'Recall@20', 'NDCG@20', 'Update (s)', 'Affected', 'Social retention'),
        rows,
    ))
    parts.append(r'\clearpage\subsection{Per-seed benchmark results}')
    for backbone, datasets in main['aggregate'].items():
        for dataset in datasets:
            rows = []
            for method, label in METHODS.items():
                for seed in main['seed_ids']:
                    run = main['runs'].get(f'{backbone}/{dataset}/{seed}/{method}')
                    if run is None:
                        continue
                    result = run['methods'][method]
                    setup = result['original_setup_seconds']
                    rows.append(row(
                        label, seed,
                        *(f'{result["utility"][key]:.4f}' for key in RANKING),
                        f'{result["update_seconds"]:.1f}',
                        f'{setup:.1f}' if setup is not None else '---',
                    ))
            parts.append(table(
                f'{backbone} on {DATASETS[dataset]} by seed. Setup is the initial ensemble fitting time.',
                ('Method', 'Seed', 'Recall@20', 'NDCG@20', 'Update (s)', 'Setup (s)'),
                rows,
            ))
    return '\n\n'.join(parts) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    source = args.source or RESULTS / 'all_results.json'
    if args.source is None and not source.exists() and args.allow_partial:
        source = RESULTS / 'all_results_partial.json'
    report = json.loads(source.read_text())
    if not report['complete'] and not args.allow_partial:
        parser.error('The verified benchmark is incomplete')
    args.output.write_text(render(report))


if __name__ == '__main__':
    main()
