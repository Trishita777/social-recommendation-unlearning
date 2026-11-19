"""Sequential exact gradients for the two-relation adapter."""
import torch
from torch.func import functional_call
from torch.nn import functional as F
from .dual_relation_unlearning import DualRelationUnlearning, masked_target_kl


class SequentialDualRelationUnlearning(DualRelationUnlearning):

    def relation_loss(self, index):
        """One deterministic relation forward, for sequential gradient release.

        Full candidate denominators and fixed target/anchor identities match
        losses(). Recompute propagation rather than retaining both loss graphs.
        """
        if index not in (0, 1):
            raise ValueError('Expected interaction (0) or social (1)')
        self.original.eval()
        params = {'model.' + name: value for name, value in self.student.named_parameters()}
        a, b = functional_call(self.wrapper, params, ())[index]
        ids = self.anchors[index]; ta, tb = self.targets[index]
        a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)
        value = a.new_zeros(())
        for start in range(0, len(ids), self.chunk):
            ix = ids[start:start+self.chunk]
            value = value + len(ix) / len(ids) * masked_target_kl(
                a[ix], b, ta[start:start+self.chunk], tb, ix, self.retained,
                self.temperature, self.soft_mix)
        return value


def staged_update_gradients(retain, objective, params, coefficients):
    """Accumulate exact native + two relation gradients without saved overlap.

    Native retention already has its graph; consume it before relation forwards.
    This uses first-order gradients and does not change the loss or candidates.
    """
    if len(coefficients) != 2:
        raise ValueError('Expected two relation coefficients')
    if not torch.isfinite(retain):
        raise FloatingPointError('Nonfinite native retention')
    grad = torch.autograd.grad(retain, params, allow_unused=True)
    for p, g in zip(params, grad):
        p.grad = torch.zeros_like(p) if g is None else g.detach()
    components = {'native_retention': float(retain.detach())}
    for index, (name, coefficient) in enumerate(zip(('interaction_contrast', 'social_contrast'), coefficients)):
        loss = objective.relation_loss(index)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite relation contrast')
        components[name] = float(loss.detach())
        if coefficient:
            delta = torch.autograd.grad(coefficient * loss, params, allow_unused=True)
            for p, g in zip(params, delta):
                if g is not None:
                    p.grad.add_(g)
        del loss
    return components
