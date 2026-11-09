"""CURE4Rec-style representation MIO components, not a complete audit protocol.

Paper: https://papers.neurips.cc/paper_files/paper/2024/file/
b364953e402d7d92e13830383677efb5-Paper-Datasets_and_Benchmarks_Track.pdf
Features follow section 3.2. Training follows section 4.5: SGD, .001, 100 epochs.
The paper's '(64,16,4)' architecture does not specify binary output wiring;
we interpret these as hidden widths and add two output logits. Record this
interpretation, input dimension, and validation selection in every report.
Missing native user rows require a declared query adapter; never silently use
stale original embeddings, zero padding, or copied item-history embeddings.
"""
import copy
import torch
from torch import nn


class BenchmarkMIO(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 4), nn.ReLU(), nn.Linear(4, 2))

    def forward(self, features):
        return self.layers(features)

    @torch.no_grad()
    def member_probability(self, features):
        self.eval()
        return self(features).softmax(1)[:, 1]


@torch.no_grad()
def embedding_features(user_embeddings, item_embeddings, queries, user_to_row, histories):
    """Concatenate native user embedding and mean interacted-item embedding.

    histories is attacker-known data, separate from the deployed model. Its
    availability must be declared. No graph or recommendation model is modified.
    A missing user raises before producing any feature rows.
    """
    missing = [int(user) for user in queries if int(user) not in user_to_row]
    if missing:
        raise ValueError(f'{len(missing)} queried users lack native embeddings; a declared query adapter is required')
    rows = []
    for user in queries:
        user = int(user)
        items = torch.as_tensor(histories[user], dtype=torch.long, device=item_embeddings.device)
        if len(items) == 0:
            raise ValueError('MIO needs a nonempty known interaction history')
        rows.append(torch.cat([user_embeddings[user_to_row[user]], item_embeddings[items].mean(0)]))
    return torch.stack(rows).detach()


def train_mio(train_x, train_y, val_x, val_y, *, seed=1, epochs=100, batch_size=512):
    """Fit on shadow train identities; select on disjoint shadow validation only.

    Caller verifies identity disjointness and feature provenance. Test arrays
    deliberately are not accepted. Validation selection is a declared addition
    to the paper's training description. No feature scaling is introduced.
    """
    for x, y in [(train_x, train_y), (val_x, val_y)]:
        if x.ndim != 2 or len(x) != len(y) or not torch.isfinite(x).all():
            raise ValueError('Invalid MIO features')
        if set(y.detach().cpu().tolist()) != {0, 1}:
            raise ValueError('Both membership classes are required')
    torch.manual_seed(seed)
    model = BenchmarkMIO(train_x.shape[1]).to(train_x.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(seed)
    best_loss = float('inf'); best_state = None; history = []
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator).to(train_x.device)
        for ids in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x[ids]), train_y[ids].long())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite MIO training loss')
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.cross_entropy(model(val_x), val_y.long()))
        history.append(dict(epoch=epoch, validation_loss=validation_loss))
        if validation_loss < best_loss:
            best_loss = validation_loss; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch
    model.load_state_dict(best_state)
    model.requires_grad_(False); model.eval()
    return model, dict(epochs=epochs, best_epoch=best_epoch, learning_rate=.001,
        optimizer='SGD', batch_size=batch_size, hidden_widths=[64,16,4], output_classes=2,
        architecture_status='paper_widths_interpreted_as_hidden_widths_with_binary_output_head',
        selection='shadow_validation_cross_entropy', history=history)
