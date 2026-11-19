"""Ensure the execution-only optimization preserves original evaluation."""
from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_unlearning import evaluate
from social_unlearning.fast_evaluation import evaluate_fast


@pytest.mark.parametrize('catalog', [1, 7, 64])
def test_exact_rankings_metrics_groups_and_reference(catalog):
    generator = torch.Generator().manual_seed(52)
    class Fixed:
        def __init__(self):
            self.u = torch.randn(259, 16, generator=generator)
            self.i = torch.randn(catalog, 16, generator=generator)
        def eval(self):
            return self
        def __call__(self):
            return self.u, self.i
    model = Fixed()
    data = SimpleNamespace(item_num=catalog, positives=[set(range(min(u % 7, catalog))) for u in range(259)])
    pairs = np.array([(u, u % catalog) for u in range(258, -1, -1)] + [(2, 0), (2, catalog-1)])
    groups = {u: str(u % 3) for u in range(259)}
    rng = torch.get_rng_state().clone()
    expected, reference = evaluate(model, data, pairs, 'cpu', groups=groups)
    actual, output = evaluate_fast(model, data, pairs, 'cpu', groups=groups)
    assert actual == expected
    for u in reference[2]:
        np.testing.assert_array_equal(reference[2][u], output[2][u])
    assert torch.equal(rng, torch.get_rng_state())
    expected, _ = evaluate(model, data, pairs, 'cpu', reference=reference, groups=groups)
    actual, _ = evaluate_fast(model, data, pairs, 'cpu', reference=reference, groups=groups)
    assert actual == expected
    empty = np.empty((0, 2), dtype=np.int64)
    assert evaluate(model, data, empty, 'cpu')[0] == evaluate_fast(model, data, empty, 'cpu')[0]
