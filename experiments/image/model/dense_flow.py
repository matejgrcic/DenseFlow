import torch
import torch.nn as nn
from denseflow.flows import Flow
from denseflow.transforms import UniformDequantization, VariationalDequantization, ScalarAffineBijection, Squeeze2d, Conv1x1, Slice, SimpleMaxPoolSurjection2d, ActNormBijection2d, WaveletSqueeze2d
from denseflow.distributions import StandardNormal, ConvNormal2d
from .flow_modules import InvertibleDenseBlock, InvertibleTransition
from .dequantization import DequantizationFlow


def parameter_count(module):
    trainable, non_trainable = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            non_trainable += p.numel()
    return trainable, non_trainable

def dim_from_shape(x):
    return x[0] * x[1] * x[2]

class DenseFlow(Flow):

    def __init__(self, data_shape=(3, 32, 32), block_config=[2, 4, 3], layers_config=[2, 2, 2],  layer_mid_chnls=[6, 12, 20], growth_rate=None, num_bits=8, checkpointing=True):

        transforms = []
        current_shape = data_shape

        # Change range from [0,1]^D to [-0.5, 0.5]^D
        transforms.append(ScalarAffineBijection(shift=-0.5))

        # Initial squeeze
        transforms.append(Squeeze2d())
        current_shape = (current_shape[0] * 4,
                         current_shape[1] // 2,
                         current_shape[2] // 2)

        # scale = 1
        dim_initial = dim_from_shape(data_shape)
        dim_output = 0
        for i, num_layers in enumerate(block_config):
            idbt = InvertibleDenseBlock(current_shape[0], num_layers, layers_config[i], layer_mid_chnls[i],
                                        growth_rate=growth_rate, checkpointing=checkpointing)
            transforms.append(idbt)

            chnls = current_shape[0] + growth_rate * (num_layers - 1)
            current_shape = (chnls,
                             current_shape[1],
                             current_shape[2])

            if i != len(block_config) - 1:
                transforms.append(InvertibleTransition(current_shape[0]))

                d0 = dim_from_shape(current_shape)

                current_shape = (current_shape[0] * 2,
                                 current_shape[1] // 2,
                                 current_shape[2] // 2)
                d1 = dim_from_shape(current_shape)
                dim_output += (d0 - d1)

        dim_output += dim_from_shape(current_shape)
        coef = 1.
        # transforms = [UniformDequantization(num_bits=num_bits, coef=coef), *transforms]
        transforms = [VariationalDequantization(encoder=DequantizationFlow(data_shape, num_bits=num_bits), num_bits=num_bits, coef=coef), *transforms]

        super(DenseFlow, self).__init__(base_dist=ConvNormal2d(current_shape),
                                       transforms=transforms, coef=coef)
        self.out_shape = current_shape

