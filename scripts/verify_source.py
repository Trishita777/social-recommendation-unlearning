"""Verify the frozen implementation and release source inventory."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifests = [json.loads((ROOT / 'configs/frozen_corrected_method.json').read_text())['sha256'],
                 json.loads((ROOT / 'configs/source_integrity.json').read_text())]
    checked = set()
    for manifest in manifests:
        for name, expected in manifest.items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f'Source mismatch: {name}')
            checked.add(name)
    print(f'Verified {len(checked)} source files, including all frozen method files.')


if __name__ == '__main__':
    main()
