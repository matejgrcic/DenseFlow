import torch
import torch.nn as nn
from denseflow.nn.blocks import ResidualDenseBlock, MultiHeadDenseBlock, DenseBlock
import torch.utils.checkpoint as cp

checkpoint = lambda func, inputs: cp.checkpoint(func, inputs, preserve_rng_state=True)

def _checkpoint_dn(t):
    def func(x):
        return t(x)
    return func


class DenseNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks,
                 mid_channels, depth, growth, dropout,
                 gated_conv=False, zero_init=False, checkpointing=False):
        super(DenseNet, self).__init__()

        layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)] +\
                 [ResidualDenseBlock(in_channels=mid_channels,
                                     out_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=False) for _ in range(num_blocks)] +\
                 [nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)]

        if zero_init:
            nn.init.zeros_(layers[-1].weight)
            if hasattr(layers[-1], 'bias'):
                nn.init.zeros_(layers[-1].bias)

        self.transform = nn.Sequential(*layers)

        # super(DenseNet, self).__init__(*layers)
        self.checkpointing = checkpointing
        self.cp_func = _checkpoint_dn(self.transform)

    def forward(self, x):
        if self.training and self.checkpointing:
            return checkpoint(self.cp_func, x)
        else:
            return self.cp_func(x)

class PureDenseNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks,
                 mid_channels, depth, growth, dropout,
                 gated_conv=False, zero_init=False, checkpointing=False):
        super(PureDenseNet, self).__init__()

        layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)] +\
                 [DenseBlock(in_channels=mid_channels,
                                     out_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=False) for _ in range(num_blocks)] +\
                 [nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)]

        if zero_init:
            nn.init.zeros_(layers[-1].weight)
            if hasattr(layers[-1], 'bias'):
                nn.init.zeros_(layers[-1].bias)

        self.transform = nn.Sequential(*layers)

        # super(DenseNet, self).__init__(*layers)
        self.checkpointing = checkpointing
        self.cp_func = _checkpoint_dn(self.transform)

    def forward(self, x):
        if self.training and self.checkpointing:
            return checkpoint(self.cp_func, x)
        else:
            return self.cp_func(x)


class DenseNetMultihead(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks,
                 mid_channels, depth, growth, dropout,
                 gated_conv=False, zero_init=False, checkpointing=False):
        super(DenseNetMultihead, self).__init__()

        layers = [nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0)] +\
                 [MultiHeadDenseBlock(in_channels=mid_channels,
                                     out_channels=mid_channels,
                                     depth=depth,
                                     growth=growth,
                                     dropout=dropout,
                                     gated_conv=gated_conv,
                                     zero_init=False) for _ in range(num_blocks)] +\
                 [nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0)]

        if zero_init:
            nn.init.zeros_(layers[-1].weight)
            if hasattr(layers[-1], 'bias'):
                nn.init.zeros_(layers[-1].bias)

        self.transform = nn.Sequential(*layers)

        # super(DenseNet, self).__init__(*layers)
        self.checkpointing = checkpointing
        self.cp_func = _checkpoint_dn(self.transform)

    def forward(self, x):
        if self.training and self.checkpointing:
            return checkpoint(self.cp_func, x)
        else:
            return self.cp_func(x)


