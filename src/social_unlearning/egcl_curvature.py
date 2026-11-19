"""Native-equivalent EGCL normalization with finite Hessians at isolated zero views.

F.normalize(x, eps=e) is x / max(norm(x), e). Writing its denominator as
sqrt(max(sum(x*x), e*e)) preserves the native value and first derivative,
while avoiding an internal 0/0 in double backward through norm at zero.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .egcl import EGCL


def finite_normalize(x,eps=1e-6):
    return x/x.square().sum(dim=1,keepdim=True).clamp_min(eps*eps).sqrt()


class CurvatureEGCL(EGCL):
    def info_nce(self,a,b,anchor_weights=None):
        a=finite_normalize(a);b=finite_normalize(b)
        loss=a.new_zeros(());size=self.cfg.get('contrastive_chunk',512);temp=self.cfg.get('temperature',.05)
        for start in range(0,len(a),size):
            end=min(start+size,len(a));labels=torch.arange(start,end,device=a.device)
            def chunk(x,y,labels):return F.cross_entropy(x@y.T/temp,labels,reduction='none')
            values=checkpoint(chunk,a[start:end],b,labels,use_reentrant=False) if self.training and torch.is_grad_enabled() else chunk(a[start:end],b,labels)
            loss=loss+(values if anchor_weights is None else values*anchor_weights[start:end].detach()).sum()
        return loss


class _StreamedCrossEntropy(torch.autograd.Function):
    """Exact full-candidate InfoNCE with checkpointed differentiable gradients.

    The native checkpointed loss still saves quadratic softmax activations when
    its first gradient is requested with create_graph=True. Checkpointing the
    analytic gradient itself prevents those activations accumulating over rows.
    """
    @staticmethod
    def forward(ctx,a,b,weights,temperature,chunk_size):
        ctx.save_for_backward(a,b,weights);ctx.temperature=temperature;ctx.chunk_size=chunk_size
        loss=a.new_zeros(())
        for start in range(0,len(a),chunk_size):
            end=min(start+chunk_size,len(a));labels=torch.arange(start,end,device=a.device)
            loss=loss+(F.cross_entropy(a[start:end]@b.T/temperature,labels,reduction='none')*weights[start:end]).sum()
        return loss

    @staticmethod
    def backward(ctx,upstream):
        a,b,weights=ctx.saved_tensors;temperature=ctx.temperature;parts=[];gb=torch.zeros_like(b)
        def derivatives(x,y,labels,w):
            prob=(x@y.T/temperature).softmax(1)
            # Out-of-place subtraction keeps higher-order differentiation valid.
            residual=(prob-F.one_hot(labels,num_classes=len(y)).to(prob.dtype))*w[:,None]/temperature
            return residual@y,residual.T@x
        for start in range(0,len(a),ctx.chunk_size):
            end=min(start+ctx.chunk_size,len(a));labels=torch.arange(start,end,device=a.device)
            args=(a[start:end],b,labels,weights[start:end])
            ga,addition=checkpoint(derivatives,*args,use_reentrant=False) if torch.is_grad_enabled() else derivatives(*args)
            parts.append(ga);gb=gb+addition
        return torch.cat(parts)*upstream,gb*upstream,None,None,None


def streamed_info_nce(a,b,temperature,chunk_size,weights=None):
    if weights is None:weights=a.new_ones(len(a))
    return _StreamedCrossEntropy.apply(a,b,weights.detach(),temperature,chunk_size)


class StreamedCurvatureEGCL(CurvatureEGCL):
    def info_nce(self,a,b,anchor_weights=None):
        return streamed_info_nce(finite_normalize(a),finite_normalize(b),
                                 self.cfg.get('temperature',.05),self.cfg.get('contrastive_chunk',512),anchor_weights)
