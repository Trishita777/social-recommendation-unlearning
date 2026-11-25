"""Run or collect the resumable, matched 30-epoch partition benchmark."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
from statistics import mean, stdev
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ('ODSR', 'RecFlow')
DATASETS = ('douban-book', 'yelp2018')
SEEDS = (42, 43, 44)
PRIMARY_METHODS = ('retrain', 'sorecremover_partition', 'sorecremover_graph_moe')
OTHER_METHODS = ('receraser_social', 'ultrare_social',
                 'grapheraser_social', 'guide_social', 'eemu_partition')
METHODS = PRIMARY_METHODS + OTHER_METHODS
EPOCHS = 30
VALIDATION_EVERY = 2
CASES = [(b, d, s) for b, d in (('ODSR','yelp2018'), ('ODSR','douban-book'), ('RecFlow','recflow-yelp')) for s in SEEDS]
METRICS = ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20')


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def result_path(out, case, method):
    backbone, dataset, seed = case
    tag = 'odsr' if backbone == 'ODSR' else 'recflow'
    return out / f'{tag}_{dataset}_seed{seed}_{method}.json'


def read_result(path, case, method, source, allow_unstamped=False, shards=5):
    run = json.loads(path.read_text()) if path.exists() else {}
    if run.get('completion') != 'complete':
        return None
    actual = (run['backbone'], run['dataset'], run['seed'], run['epochs'],
              run['validation_every'], run['fraction'], run['shards'])
    expected = (*case, EPOCHS, VALIDATION_EVERY, .01, shards)
    if actual != expected or run['source_sha256'] != source or set(run['methods']) != {method}:
        raise ValueError(f'Protocol or source mismatch in {path}')
    if not allow_unstamped and 'execution' not in run:
        return None
    row = run['methods'][method]
    if not row['update_training'] or any(
        not 1 <= entry['best_epoch'] <= entry['epochs'] <= EPOCHS
        for entry in row['original_training'] + row['update_training']
    ):
        raise ValueError(f'Invalid checkpoint metadata in {path}')
    return run


def benchmark_result(args, case, method, source, allow_unstamped=False):
    path = result_path(args.output, case, method)
    run = read_result(path, case, method, source, allow_unstamped, args.shards)
    if run is None and method == 'retrain' and args.retrain_from:
        path = result_path(args.retrain_from, case, method)
        run = read_result(path, case, method, args.reference_source)
        if run is not None:
            run['reused_from'] = str(path)
    return run


def worker(args, source):
    cases = []
    for value in args.cases:
        backbone, dataset, seed = value.split(':')
        case = (backbone, dataset, int(seed))
        if case not in CASES or case in cases:
            raise ValueError(f'Invalid or duplicate case {value!r}')
        cases.append(case)
    jobs = [(case, method) for methods in (PRIMARY_METHODS, OTHER_METHODS)
            for case in cases for method in methods if method in args.methods]
    execution = dict(host=socket.gethostname(), physical_gpu=args.gpu)
    python_paths = [str(ROOT/'scripts')]
    if os.environ.get('PYTHONPATH'):
        python_paths.append(os.environ['PYTHONPATH'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
               PYTHONPATH=os.pathsep.join(python_paths))
    completed = 0
    status_path = args.output / f'worker_{args.worker}.json'

    def status(state, **fields):
        save(status_path, dict(state=state, pid=os.getpid(), execution=execution,
                               completed=completed, total=len(jobs), updated=time.time(), **fields))

    with (args.output / f'worker_{args.worker}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for case, method in jobs:
                path = result_path(args.output, case, method)
                run = benchmark_result(args, case, method, source, allow_unstamped=True)
                if run is None:
                    status('waiting_for_gpu_lock', active=[*case, method])
                    manager = args.gpu_lock.open('a') if args.gpu_lock else nullcontext()
                    with manager as gpu:
                        if gpu is not None:
                            fcntl.flock(gpu, fcntl.LOCK_EX)
                        if hashlib.sha256(args.runner.read_bytes()).hexdigest() != source:
                            raise RuntimeError('The runner changed during the sweep')
                        cmd = [sys.executable, '-u', str(args.runner), '--backbone', case[0],
                               '--dataset', case[1], '--seed', str(case[2]), '--device', 'cuda:0',
                               '--epochs', str(EPOCHS), '--validation-every', str(VALIDATION_EVERY),
                               '--shards', str(args.shards), '--methods', method, '--output', str(path)]
                        if args.router_ablations and method == 'sorecremover_graph_moe' and case[:2] in (
                            ('ODSR', 'yelp2018'),
                        ):
                            cmd.append('--router-ablations')
                        with path.with_suffix('.log').open('a') as log:
                            job = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                            status('running', active=[*case, method], child_pid=job.pid)
                            if job.wait():
                                raise RuntimeError(f'{case} {method} failed. See {path.with_suffix(".log")}')
                    run = benchmark_result(args, case, method, source, allow_unstamped=True)
                    if run is None:
                        raise RuntimeError(f'Missing complete runner output {path}')
                if 'execution' not in run:
                    run['execution'] = execution
                    save(path, run)
                elif run['execution'] != execution:
                    raise RuntimeError(f'Execution device changed for {path}')
                completed += 1
                status('case_complete', active=[*case, method])
            status('complete')
        except Exception as error:
            status('failed', error=str(error))
            raise


def collect(args, source):
    out = args.output
    groups = {}
    runs = {}
    progress = {}
    for case in CASES:
        backbone, dataset, seed = case
        recorded = {}
        for method in args.methods:
            run = benchmark_result(args, case, method, source)
            if run is not None:
                recorded[method] = run
        progress[f'{backbone}/{dataset}/{seed}'] = list(recorded)
        reference = recorded.get('retrain')
        for method, run in recorded.items():
            row = run['methods'][method]
            if reference:
                for field in ('split_hash', 'deletion_hash', 'social_hash'):
                    if run[field] != reference[field]:
                        raise ValueError(f'Paired benchmark mismatch for {case} {method}: {field}')
            if reference and run['execution'] == reference['execution']:
                row['update_over_retrain'] = row['update_seconds'] / reference['methods']['retrain']['update_seconds']
            runs[f'{backbone}/{dataset}/{seed}/{method}'] = run
            groups.setdefault(backbone, {}).setdefault(dataset, {}).setdefault(method, []).append(row)

    aggregate = {}
    for backbone, datasets in groups.items():
        for dataset, methods in datasets.items():
            for method, rows in methods.items():
                metrics = {m: [r['utility'][m] for r in rows] for m in METRICS}
                for key in ('update_seconds', 'update_over_retrain', 'original_setup_seconds', 'test_seconds'):
                    if all(r.get(key) is not None for r in rows):
                        metrics[key] = [r[key] for r in rows]
                mean_sd = {key: {'mean': mean(values), 'sample_sd': stdev(values) if len(values)>1 else None}
                           for key, values in metrics.items()}
                aggregate.setdefault(backbone, {}).setdefault(dataset, {})[method] = {
                    'n': len(rows), **mean_sd,
                }
    total = len(CASES) * len(args.methods)
    save(out/'progress.json', dict(completed=len(runs), total=total, by_case=progress, updated=time.time()))
    output = dict(epochs=EPOCHS, shards=args.shards, validation_every=VALIDATION_EVERY, seed_ids=SEEDS,
                  complete=len(runs)==total, completed=len(runs), total=total,
                  aggregate=aggregate, runs=runs)
    save(out/('summary.json' if output['complete'] else 'partial_summary.json'), output)
    print(f"{len(runs)}/{total} complete", flush=True)
    return output['complete']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner', type=Path, default=ROOT/'scripts/run_partition_social.py')
    parser.add_argument('--worker')
    parser.add_argument('--shards', type=int, default=10)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--retrain-from', type=Path)
    parser.add_argument('--router-ablations', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu-lock', type=Path)
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    if bool(args.worker) != bool(args.cases):
        parser.error('--worker and --cases are required together')
    args.output = args.output.resolve()
    args.runner = args.runner.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    source = hashlib.sha256(args.runner.read_bytes()).hexdigest()
    if args.retrain_from:
        args.reference_source = hashlib.sha256((args.retrain_from / 'runner.snapshot').read_bytes()).hexdigest()
    if args.worker:
        worker(args, source)
    else:
        collect(args, source)
