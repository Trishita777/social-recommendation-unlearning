"""Sequential, fail-fast main comparisons, deletion sweeps, and ODSR ablations."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(args):
    for dataset in args.datasets:
        for seed in args.seeds:
            common = ['--dataset', dataset, '--seed', str(seed), '--device', args.device]
            yield ['scripts/train_original.py', '--backbone', args.backbone, *common]
            for fraction in args.fractions:
                if args.backbone == 'ODSR':
                    deletion = [*common, '--fraction', str(fraction)]
                    # A matched scratch reference is required by all evaluation paths.
                    yield ['scripts/run_edge_user_odsr.py', *deletion, '--method', 'retrain']
                    for method in args.methods:
                        if method in ('retrain', 'sorecremover'):
                            continue
                        yield ['scripts/run_edge_user_odsr.py', *deletion, '--method', method]
                    if 'sorecremover' in args.methods:
                        for variant in args.variants:
                            yield ['scripts/run_odsr_native_topology.py', *deletion, '--variant', variant]
                else:
                    yield ['scripts/run_egcl_edge_controls.py', *common, '--method', 'retrain']
                    for method in args.methods:
                        if method == 'retrain':
                            continue
                        if method in ('graph_only', 'retain_finetune'):
                            yield ['scripts/run_egcl_edge_controls.py', *common, '--method', method]
                        else:
                            adapter = 'native_topology' if method == 'sorecremover' else 'gnndelete_native'
                            yield ['scripts/run_egcl_unlearning_adapters.py', *common, '--method', adapter]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', choices=['ODSR', 'EGCL'], default='ODSR')
    p.add_argument('--datasets', nargs='+', choices=['epinions', 'yelp2018'], default=['epinions', 'yelp2018'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--fractions', nargs='+', type=float, default=[.01])
    p.add_argument('--methods', nargs='+', choices=['retrain', 'graph_only', 'retain_finetune', 'scif', 'gnndelete', 'da2', 'sorecremover'])
    p.add_argument('--variants', nargs='+', choices=['full', 'no_topology', 'uniform_weights', 'no_reset'], default=['full'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if any(not 0 < f < 1 for f in a.fractions):
        p.error('Deletion fractions must be between zero and one')
    if a.methods is None:
        a.methods = ['retrain', 'graph_only', 'retain_finetune', 'gnndelete', 'sorecremover']
        if a.backbone == 'ODSR':
            a.methods += ['scif', 'da2']
    if a.backbone == 'EGCL' and (a.fractions != [.01] or a.variants != ['full'] or any(m in a.methods for m in ['scif', 'da2'])):
        p.error('EGCL release supports 1% deletion, full variant, and retrain/graph_only/retain_finetune/gnndelete/sorecremover')
    for command in commands(a):
        command = [sys.executable, '-u', *command]
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
