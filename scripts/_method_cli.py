"""Named public experiment commands mapped to the frozen implementations."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'retraining': ('Scratch retraining', 'retrain'),
    'graph_pruning': ('Graph pruning', 'graph_only'),
    'retain_finetuning': ('Retained fine-tuning', 'retain_finetune'),
    'scif': ('SCIF', 'scif'),
    'gnndelete': ('GNNDelete', 'gnndelete'),
    'da2unlearn': ('DA2Unlearn', 'da2'),
    'sorecremover': ('SoRecRemover', None),
}
ALIASES = {'retrain': 'retraining', 'graph_only': 'graph_pruning',
           'retain_finetune': 'retain_finetuning', 'da2': 'da2unlearn'}


def canonical_method(value):
    return ALIASES.get(value, value)


def main(method):
    title, internal = METHODS[method]
    p = argparse.ArgumentParser(description=f'Run {title} with the frozen study protocol.')
    supported = ['ODSR'] if method in ('scif', 'da2unlearn') else ['ODSR', 'EGCL']
    p.add_argument('--backbone', choices=supported, default='ODSR')
    p.add_argument('--dataset', choices=['epinions', 'yelp2018'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fraction', type=float, default=.01)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    if method == 'sorecremover':
        p.add_argument('--variant', choices=['full', 'no_topology', 'uniform_weights', 'no_reset'], default='full')
        p.add_argument('--bank', type=Path, help='Optional ODSR positive-only MIA bank; use controlled MIA for target AUC')
    a = p.parse_args()
    if not 0 < a.fraction < 1:
        p.error('Deletion fraction must be between zero and one')
    if a.backbone == 'EGCL':
        if a.fraction != .01:
            p.error('The released matched EGCL controls support 1% deletion only')
        if method == 'sorecremover' and (a.variant != 'full' or a.bank is not None):
            p.error('EGCL supports the full variant; its MIA uses the separate controlled evaluator')
    common = ['--dataset', a.dataset, '--seed', str(a.seed), '--device', a.device]
    if a.backbone == 'ODSR':
        common += ['--fraction', str(a.fraction)]
        if method == 'sorecremover':
            command = ['scripts/run_odsr_native_topology.py', *common, '--variant', a.variant]
            if a.bank is not None:
                command += ['--bank', str(a.bank.resolve())]
        else:
            command = ['scripts/run_edge_user_odsr.py', *common, '--method', internal]
    elif method in ('sorecremover', 'gnndelete'):
        adapter = 'native_topology' if method == 'sorecremover' else 'gnndelete_native'
        command = ['scripts/run_egcl_unlearning_adapters.py', *common, '--method', adapter]
    else:
        command = ['scripts/run_egcl_edge_controls.py', *common, '--method', internal]
    command = [sys.executable, '-u', *command]
    if a.dry_run:
        print(shlex.join(command))
    else:
        subprocess.run(command, cwd=ROOT, check=True)
