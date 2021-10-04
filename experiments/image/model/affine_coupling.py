from denseflow.transforms import Bijection, BatchNormBijection2d, Conv1x1
from denseflow.utils import sum_except_batch
from denseflow.nn.layers import ElementwiseParams2d
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from denseflow.nn.nets import DenseNetMultihead

import torch.nn as nn
import torch

checkpoint = lambda func, inputs: cp.checkpoint(func, inputs, preserve_rng_state=True)

def _checkpoint_fb(t):
    def func(x):
        return t(x) + torch.cat((x, x), dim=1)
    return func

# Intra-unit coupling net
class DNMH(nn.Module):
    def __init__(self, inChannels, midChannel, outChannel, checkpointing=False):
        super(DNMH, self).__init__()
        self.dn = DenseNetMultihead(in_channels=inChannels,
                 out_channels=outChannel,
                 num_blocks=1,
                 mid_channels=midChannel,
                 depth=7,
                 # depth=4,
                 growth=64,
                 dropout=0.0,
                 gated_conv=True,
                 zero_init=True, checkpointing=checkpointing)

    def forward(self, x):
        return self.dn(x)

class AffineCouplingBijection(Bijection):
    def __init__(self, coupling_net, split_dim=1, num_condition=None):
        super(AffineCouplingBijection, self).__init__()
        assert split_dim >= 1
        self.coupling_net = coupling_net
        self.split_dim = split_dim
        self.num_condition = num_condition

    def split_input(self, input):
        if self.num_condition:
            split_proportions = (self.num_condition, input.shape[self.split_dim] - self.num_condition)
            return torch.split(input, split_proportions, dim=self.split_dim)
        else:
            return torch.chunk(input, 2, dim=self.split_dim)

    def forward(self, x):
        if not x.requires_grad:
            x.requires_grad = True
        x1, x2 = self.split_input(x)
        elementwise_params = self.coupling_net(x1)
        z2, ldj = self._elementwise_forward(x2, elementwise_params)

        z = torch.cat([x1, z2], dim=self.split_dim)
        return z, ldj

    def inverse(self, z):
        with torch.no_grad():
            z1, z2 = self.split_input(z)
            x1 = z1

            elementwise_params = self.coupling_net(x1)
            x2 = self._elementwise_inverse(z2, elementwise_params)

            x = torch.cat([x1, x2], dim=self.split_dim)
        return x

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, x, elementwise_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, z, elementwise_params):
        raise NotImplementedError()


class AdvancedAffineCouplingBijection(AffineCouplingBijection):

    def __init__(self, coupling_net, split_dim=1, num_condition=None, scale_fn=lambda s: torch.exp(s)):
        super(AdvancedAffineCouplingBijection, self).__init__(coupling_net=coupling_net, split_dim=split_dim, num_condition=num_condition)
        assert callable(scale_fn)
        self.scale_fn = scale_fn

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        scale = self.scale_fn(unconstrained_scale)
        z = scale * x + shift
        ldj = sum_except_batch(torch.log(scale))
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        scale = self.scale_fn(unconstrained_scale)
        x = (z - shift) / scale
        return x

    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift


class SingleAffineCoupling(AdvancedAffineCouplingBijection):

    def __init__(self, in_channels, mid_chnls=16, checkpointing=False):
        # assert in_channels % 2 == 0
        out_c = in_channels // 2
        in_c = in_channels - out_c
        net = nn.Sequential(DNMH(
                                      inChannels=in_c,
                                      midChannel=mid_chnls,
                                      outChannel=out_c*2,
                                      checkpointing=checkpointing),
                             ElementwiseParams2d(2, mode='sequential'))
        super(AdvancedAffineCouplingBijection, self).__init__(coupling_net=net)

    def _elementwise_forward(self, x, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        z = shift + torch.exp(log_scale) * x
        ldj = sum_except_batch(log_scale)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        x = (z - shift) * torch.exp(-log_scale)
        return x
