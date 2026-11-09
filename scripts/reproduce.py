"""Sequential, fail-fast main comparisons, deletion sweeps, and ODSR ablations."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from _method_cli import METHODS, canonical_method

ROOT = Path(__file__).resolve().parents[1]


def commands(args):
    for dataset in args.datasets:
        for seed in args.seeds:
            common = ['--dataset', dataset, '--seed', str(seed), '--device', args.device]
            yield [f'scripts/train_{args.backbone.lower()}.py', *common]
            for fraction in args.fractions:
                deletion = ['--backbone', args.backbone, *common, '--fraction', str(fraction)]
                # A matched scratch reference is required by all evaluation paths.
                yield ['scripts/run_retraining.py', *deletion]
                for method in args.methods:
                    if method == 'retraining':
                        continue
                    if method == 'sorecremover':
                        for variant in args.variants:
                            yield ['scripts/run_sorecremover.py', *deletion, '--variant', variant]
                    else:
                        yield [f'scripts/run_{method}.py', *deletion]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbone', choices=['ODSR', 'EGCL'], default='ODSR')
    p.add_argument('--datasets', nargs='+', choices=['epinions', 'yelp2018'], default=['epinions', 'yelp2018'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--fractions', nargs='+', type=float, default=[.01])
    p.add_argument('--methods', nargs='+', type=canonical_method, choices=list(METHODS))
    p.add_argument('--variants', nargs='+', choices=['full', 'no_topology', 'uniform_weights', 'no_reset'], default=['full'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if any(not 0 < f < 1 for f in a.fractions):
        p.error('Deletion fractions must be between zero and one')
    if a.methods is None:
        a.methods = ['retraining', 'graph_pruning', 'retain_finetuning', 'gnndelete', 'sorecremover']
        if a.backbone == 'ODSR':
            a.methods += ['scif', 'da2unlearn']
    if a.backbone == 'EGCL' and (a.fractions != [.01] or a.variants != ['full'] or any(m in a.methods for m in ['scif', 'da2unlearn'])):
        p.error('EGCL supports 1% deletion, full variant, and retraining/graph_pruning/retain_finetuning/gnndelete/sorecremover')
    for command in commands(a):
        command = [sys.executable, '-u', *command]
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
