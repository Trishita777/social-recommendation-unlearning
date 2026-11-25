"""Paper-based RecFlow reproduction with a conditional vector field."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RecFlow(nn.Module):
    def __init__(self, data, device, seed, item_generator):
        super().__init__()
        dim = 64
        self.seed = seed
        self.users = nn.Parameter(torch.empty(data.user_num, dim, device=device))
        self.items = nn.Parameter(torch.empty(data.item_num, dim, device=device))
        self.social = nn.Parameter(torch.empty_like(self.users))
        nn.init.xavier_uniform_(self.users)
        nn.init.xavier_uniform_(self.social)
        nn.init.xavier_uniform_(self.items, generator=item_generator)
        self.field = nn.Sequential(nn.Linear(2*dim+8, dim), nn.GELU(), nn.Linear(dim, dim)).to(device)
        self.interaction_graph = data.norm_adj.coalesce().to_sparse_csr()
        # Recover the unnormalized adjacency up to a constant from the row-normalized
        # matrix. Off-diagonal edge weights are one, including after GUIDE repairs.
        adj = data.adj_uu.coalesce()
        row, col = adj.indices()
        off = row != col
        degree = torch.zeros(len(self.users), device=device)
        degree[row[off]] = adj.values()[off].reciprocal()
        weight = adj.values() * degree[row]
        weight = torch.where(off, weight, (weight - 1).clamp_min(0))
        degrees = torch.zeros_like(degree).index_add_(0, row, weight)
        inv = torch.where(degrees > 0, degrees.rsqrt(), 0)
        self.friends = torch.sparse_coo_tensor(adj.indices(), inv[row]*weight*inv[col],
                                               adj.shape).coalesce().to_sparse_csr()

    @staticmethod
    def propagate(graph, embeddings):
        summed = embeddings
        for _ in range(2):
            embeddings = embeddings + graph @ embeddings
            summed = summed + embeddings
        return summed

    def encode(self):
        collaborative = self.propagate(self.interaction_graph, torch.cat((self.users, self.items)))
        u, i = collaborative.split((len(self.users), len(self.items)))
        return u, i, self.propagate(self.friends, self.social)

    def velocity(self, x, t, context):
        phase = t * t.new_tensor([1, 2, 4, 8]) * math.pi
        return self.field(torch.cat((x, context, phase.sin(), phase.cos()), dim=-1))

    def reconstruct(self, social, generator=None):
        t = torch.rand((len(social), 1), device=social.device, generator=generator)
        noise = torch.randn(social.shape, device=social.device, generator=generator)
        x = (1-t)*social + t*noise
        velocity = self.velocity(x, t, social)
        predicted = x - t*velocity
        corrected = x - .5*t*(velocity+self.velocity(predicted, torch.zeros_like(t), social))
        return corrected, F.mse_loss(velocity, noise-social)

    def loss(self, user, positive, negative):
        u, i, s = self.encode()
        correction, flow_loss = self.reconstruct(s[user])
        margins = ((u[user]+correction)*(i[positive]-i[negative])).sum(1)
        return F.softplus(-margins).mean() + flow_loss

    def forward(self):
        u, i, s = self.encode()
        generator = torch.Generator(device=s.device).manual_seed(self.seed + 13571)
        correction, _ = self.reconstruct(s, generator)
        return u+correction, i
