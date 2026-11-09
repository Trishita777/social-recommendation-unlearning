import tempfile
from pathlib import Path
import numpy as np
import torch
from social_unlearning.checkpoint import save_checkpoint, load_checkpoint


def test_exact_resume_and_reject_mismatch():
    torch.manual_seed(8)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    x = torch.ones(3, 2)
    def step():
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
    step()
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory) / 'last.pt'
        cfg = {'seed': 8}
        provenance = {'source': 'test', 'split': 'a', 'deletion': 'b'}
        save_checkpoint(p, model, optimizer, config=cfg, provenance=provenance, epoch=1, step=1)
        expected_random = (torch.rand(2), np.random.rand(2))
        step()
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        load_checkpoint(p, model, optimizer, expected_provenance=provenance, expected_config=cfg)
        assert torch.equal(torch.rand(2), expected_random[0])
        assert np.array_equal(np.random.rand(2), expected_random[1])
        step()
        assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items())
        try:
            load_checkpoint(p, model, optimizer, expected_provenance={'source': 'different'}, expected_config=cfg)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched provenance accepted')

if __name__ == '__main__':
    test_exact_resume_and_reject_mismatch()
    print('PASS: exact optimizer/RNG resume and provenance rejection')
