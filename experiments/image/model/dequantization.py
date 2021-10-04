import torch
import torch.nn as nn
from denseflow.utils import sum_except_batch
from denseflow.transforms import AffineCouplingBijection, ConditionalAffineCouplingBijection, Conv1x1, Unsqueeze2d, Sigmoid, Squeeze2d
from denseflow.nn.layers import ElementwiseParams2d, LambdaLayer
from denseflow.nn.nets import DenseNet
from denseflow.nn.blocks import DenseBlock
from denseflow.distributions import ConvNormal2d
from denseflow.flows import ConditionalInverseFlow

class ConditionalCoupling(ConditionalAffineCouplingBijection):

    def __init__(self, in_channels, num_context, num_blocks, mid_channels, depth, growth, dropout, gated_conv, checkpointing=False):
        assert in_channels % 2 == 0

        net = nn.Sequential(DenseNet(in_channels=in_channels // 2 + num_context,
                                     out_channels=in_channels,
                                     num_blocks=num_blocks,
                                     mid_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=True),
                            ElementwiseParams2d(2, mode='sequential'))
        super(ConditionalCoupling, self).__init__(coupling_net=net)

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


class DequantizationFlow(ConditionalInverseFlow):

    def __init__(self, data_shape, num_bits=8, num_steps=4, num_context=64,
                 num_blocks=1, mid_channels=64, depth=4, growth=64, dropout=0.0, gated_conv=True, checkpointing=False):
        context_net = nn.Sequential(LambdaLayer(lambda x: 2 * x.float() / (2 ** num_bits - 1) - 1),
                                    DenseBlock(in_channels=data_shape[0],
                                               out_channels=mid_channels,
                                               depth=4,
                                               growth=16,
                                               dropout=dropout,
                                               gated_conv=gated_conv,
                                               zero_init=False),
                                    nn.Conv2d(mid_channels, mid_channels, kernel_size=2, stride=2, padding=0),
                                    DenseBlock(in_channels=mid_channels,
                                               out_channels=num_context,
                                               depth=4,
                                               growth=16,
                                               dropout=dropout,
                                               gated_conv=gated_conv,
                                               zero_init=False))

        transforms = []
        sample_shape = (data_shape[0] * 4, data_shape[1] // 2, data_shape[2] // 2)
        for i in range(num_steps):
            transforms.extend([
                Conv1x1(sample_shape[0]),
                ConditionalCoupling(in_channels=sample_shape[0],
                                    num_context=num_context,
                                    num_blocks=num_blocks,
                                    mid_channels=mid_channels,
                                    depth=depth,
                                    growth=growth,
                                    dropout=dropout,
                                    gated_conv=gated_conv, checkpointing=checkpointing)
            ])

        # Final shuffle of channels, squeeze and sigmoid
        transforms.extend([Conv1x1(sample_shape[0]),
                           Unsqueeze2d(),
                           Sigmoid()
                           ])

        super(DequantizationFlow, self).__init__(base_dist=ConvNormal2d(sample_shape),
                                                 transforms=transforms,
                                                 context_init=context_net)
