"""Fetch pinned baseline sources and optionally verified official ODSR datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def download(url):
    with urlopen(url, timeout=180) as response:
        return response.read()


def checkout(record):
    path = ROOT / record['path']
    if path.exists():
        if not (path / '.git').is_dir():
            raise RuntimeError(f'{path}: existing directory is not a source checkout')
        current = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip()
        if current != record['commit'] or dirty:
            raise RuntimeError(f'{path}: existing checkout differs; use a fresh directory')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    subprocess.run(['git', 'init', str(path)], check=True, env=env)
    subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', record['url']], check=True, env=env)
    subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', record['commit']], check=True, env=env)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', 'FETCH_HEAD'], check=True, env=env)


def dataset_files(dataset, revision):
    records = []
    for candidates in [('train.txt',), ('test.txt',), ('trust.txt', 'links.txt')]:
        for name in candidates:
            rel = f'dataset/{dataset}/{name}'
            url = f'https://raw.githubusercontent.com/Rongfeng-Guo/ODSR-KDD-2026/{revision}/{rel}'
            try:
                pointer = download(url)
                break
            except HTTPError as exc:
                if exc.code != 404 or name == candidates[-1]:
                    raise
        if pointer.startswith(b'version https://git-lfs.github.com/spec/v1'):
            lines = pointer.decode().splitlines()
            expected = next(line.split('sha256:')[1] for line in lines if line.startswith('oid '))
            size = int(next(line.split()[1] for line in lines if line.startswith('size ')))
            source = f'https://media.githubusercontent.com/media/Rongfeng-Guo/ODSR-KDD-2026/{revision}/{rel}'
        else:
            expected, size, source = hashlib.sha256(pointer).hexdigest(), len(pointer), url
        dest = ROOT / 'external/ODSR' / rel
        if dest.exists():
            content = dest.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected or len(content) != size:
                raise RuntimeError(f'{dest}: existing data differs from pinned source')
        else:
            content = download(source) if source != url else pointer
            if len(content) != size or hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f'{rel}: downloaded size/hash mismatch')
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix('.download')
            temp.write_bytes(content)
            temp.replace(dest)
        records.append(dict(path=str(dest.relative_to(ROOT)), sha256=expected, size=size, source=source))
        print(f'Verified {rel}', flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='*', choices=['GNNDelete', 'DA2Unlearn', 'EGCL'], default=['GNNDelete', 'DA2Unlearn', 'EGCL'])
    p.add_argument('--datasets', nargs='*', choices=['epinions', 'yelp2018'], default=[])
    a = p.parse_args()
    pinned = json.loads((ROOT / 'configs/upstream_sources.json').read_text())
    for name in a.sources:
        checkout(pinned[name])
    for dataset in a.datasets:
        records = dataset_files(dataset, pinned['ODSR']['commit'])
        dest = ROOT / 'results/data_manifests' / f'{dataset}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
