"""Run both backbones and corrected methods on synthetic data in a temporary copy.

This checks executable integration, not full-data convergence or paper metrics.
The temporary copy is removed on exit; real experiment artifacts are untouched.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    import torch
    if not a.device.startswith('cuda') or not torch.cuda.is_available():
        p.error('This integration smoke test requires CUDA; use pytest for CPU checks')
    with tempfile.TemporaryDirectory(prefix='sorecremover-smoke-') as directory:
        root = Path(directory)
        for folder in ['scripts', 'src', 'configs', 'external/ODSR', 'external/EGCL/model/social_recommender']:
            for source in (ROOT / folder).rglob('*'):
                if source.is_file() and source.suffix in ['.py', '.json'] and '__pycache__' not in source.parts:
                    target = root / source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        data = root / 'external/ODSR/dataset/epinions'
        data.mkdir(parents=True)
        train = [(u, (u * 7 + j * 3) % 60) for u in range(40) for j in range(8)]
        test = [(u, (u * 7 + 1) % 60) for u in range(40)]
        social = [(u, (u + 1) % 40) for u in range(40)]
        for name, pairs in [('train.txt', train), ('test.txt', test), ('trust.txt', social)]:
            (data / name).write_text(''.join(f'{u} {i}\n' for u, i in pairs))
        common = ['--dataset', 'epinions', '--seed', '42', '--device', a.device]
        commands = []
        for backbone in ['ODSR', 'EGCL']:
            commands.append(['scripts/train_original.py', '--backbone', backbone, '--epochs', '2', *common])
            control = 'scripts/run_edge_user_odsr.py' if backbone == 'ODSR' else 'scripts/run_egcl_edge_controls.py'
            for method in ['retrain', 'graph_only', 'retain_finetune']:
                commands.append([control, *common, '--method', method])
            if backbone == 'ODSR':
                commands.append(['scripts/run_odsr_native_topology.py', *common, '--variant', 'full'])
            else:
                commands.append(['scripts/run_egcl_unlearning_adapters.py', *common, '--method', 'native_topology'])
        for command in commands:
            print('RUN', ' '.join(command), flush=True)
            result = subprocess.run([sys.executable, '-u', *command], cwd=root, capture_output=True, text=True)
            if result.returncode:
                print(result.stdout[-8000:])
                print(result.stderr[-8000:])
                raise SystemExit(result.returncode)
        reports = list((root / 'checkpoints').rglob('results.json'))
        checked = 0
        for path in reports:
            report = json.loads(path.read_text())
            metrics = report.get('metrics', {})
            for key in ['recall@20', 'ndcg@20']:
                if key in metrics:
                    if not math.isfinite(metrics[key]):
                        raise RuntimeError(f'Nonfinite metric: {path}: {key}')
                    checked += 1
        if checked < 16:
            raise RuntimeError(f'Expected utility reports from both pipelines; checked {checked} metrics')
        print(f'PASS: {len(commands)} GPU commands, {len(reports)} result files, {checked} finite utility metrics.')


if __name__ == '__main__':
    main()
