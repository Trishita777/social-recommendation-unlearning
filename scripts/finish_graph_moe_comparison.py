"""Complete GraphMoE cells against the existing five-epoch partition benchmark."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from statistics import mean, stdev

from run_partition_epoch_sweep import CASES, METRICS, ROOT, result_path, save

DATA = ROOT / 'results/partition_social'
OUT = DATA / 'graph_moe_matched5'
GRAPH = 'sorecremover_graph_moe'
REQUIRED = ('retrain', 'sorecremover_partition', 'receraser_social', 'ultrare_social',
            'grapheraser_social', 'guide_social', 'eemu_partition')
FIELDS = ('split_hash', 'deletion_hash', 'social_hash', 'aggregation_epochs',
          'backbone_source_sha256', 'deleted_users')
VALUES = (*METRICS, 'update_seconds', 'original_setup_seconds')


def inventory():
    files = [*sorted(DATA.glob('*.json'))]
    for name in ('three_seed', 'diffnet_three_seed', 'guide_three_seed',
                 'eemu_three_seed', 'graph_moe_validation', OUT.name):
        files += sorted((DATA / name).glob('*seed*.json'))
    records = {case: {} for case in CASES}
    for path in files:
        run = json.loads(path.read_text())
        case = (run.get('backbone', 'ODSR'), run.get('dataset'), run.get('seed'))
        if (case not in records or run.get('completion') != 'complete'
                or tuple(run.get(k) for k in ('epochs', 'validation_every', 'fraction', 'shards'))
                != (5, 2, .01, 5)):
            continue
        for method, row in run['methods'].items():
            if method in (*REQUIRED, GRAPH):
                records[case][method] = (path, run, row)
    for case, methods in records.items():
        missing = set(REQUIRED) - methods.keys()
        if missing:
            raise ValueError(f'Missing reusable baselines for {case}: {sorted(missing)}')
        for field in FIELDS:
            recorded = {run[field] for _, run, _ in methods.values() if field in run}
            if len(recorded) > 1:
                raise ValueError(f'Mismatched {field} for {case}')
    return records


def report(records):
    groups = {}
    cases = {}
    for (backbone, dataset, seed), methods in records.items():
        results = {}
        for method, (path, run, row) in methods.items():
            values = {key: row['utility'].get(key, row.get(key)) for key in VALUES}
            results[method] = dict(source=str(path.relative_to(ROOT)), gpu=run.get('gpu'), **values)
            groups.setdefault((backbone, dataset, method), []).append(values)
        cases[f'{backbone}/{dataset}/{seed}'] = results
    aggregate = []
    for (backbone, dataset, method), rows in sorted(groups.items()):
        entry = dict(backbone=backbone, dataset=dataset, method=method, seeds=len(rows))
        for key in VALUES:
            vals = [row[key] for row in rows if row[key] is not None]
            entry[f'{key}_mean'] = mean(vals) if vals else None
            entry[f'{key}_sd'] = stdev(vals) if len(vals) > 1 else None
        aggregate.append(entry)
    complete = sum(GRAPH in methods for methods in records.values())
    save(OUT / 'comparison.json', dict(epochs=5, graph_moe_completed=complete,
                                      graph_moe_total=len(CASES), cases=cases, aggregate=aggregate))
    temp = OUT / 'comparison.csv.tmp'
    with temp.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=aggregate[0].keys())
        writer.writeheader()
        writer.writerows(aggregate)
    temp.replace(OUT / 'comparison.csv')
    print(f'GraphMoE: {complete}/{len(CASES)}; all {len(REQUIRED) * len(CASES)} baseline cells reused', flush=True)


def refresh():
    with (OUT / 'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = inventory()
        report(records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--datasets', nargs='+', choices=('epinions', 'yelp2018'))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    records = refresh()
    if args.gpu is None:
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    for case in CASES:
        backbone, dataset, seed = case
        if dataset not in args.datasets or GRAPH in records[case]:
            continue
        output = result_path(OUT, case, GRAPH)
        checkpoint = OUT / (output.stem + '.pending')
        command = [sys.executable, '-u', str(ROOT / 'scripts/run_partition_social.py'),
                   '--backbone', backbone, '--dataset', dataset, '--seed', str(seed),
                   '--device', 'cuda:0', '--epochs', '5', '--validation-every', '2',
                   '--methods', GRAPH, '--output', str(checkpoint)]
        with (ROOT / f'results/runtime_acceleration/gpu{args.gpu}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(f'Running {case} on GPU {args.gpu}', flush=True)
            with output.with_suffix('.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            record = json.loads(checkpoint.read_text())
            if record.get('completion') != 'complete':
                raise RuntimeError(f'Incomplete output in {checkpoint}')
            record['execution'] = dict(host=os.uname().nodename, physical_gpu=args.gpu)
            save(output, record)
            checkpoint.unlink()
        records = refresh()


if __name__ == '__main__':
    main()
