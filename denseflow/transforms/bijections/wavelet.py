import torch
import torch.nn.functional as F
from denseflow.transforms.bijections import Bijection


def _create_kernel():
    w = torch.ones(4, 1, 2, 2)
    w[1, 0, 0, 1] = -1
    w[1, 0, 1, 1] = -1

    w[2, 0, 1, 0] = -1
    w[2, 0, 1, 1] = -1

    w[3, 0, 1, 0] = -1
    w[3, 0, 0, 1] = -1
    w *= 0.5
    return w

class WaveletSqueeze2d(Bijection):
    def __init__(self, in_channels):
        super(WaveletSqueeze2d, self).__init__()

        w = _create_kernel()
        self.in_channels = in_channels
        w = torch.cat([w] * self.in_channels, 0)
        self.register_buffer('weight', w)

    def forward(self, x):
        z = F.conv2d(x, self.weight, bias=None, stride=2, groups=self.in_channels)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = F.conv_transpose2d(z, self.weight, bias=None, stride=2, groups=self.in_channels)
        return x
