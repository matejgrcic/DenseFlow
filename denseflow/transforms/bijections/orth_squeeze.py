import torch
import torch.nn.functional as F
from denseflow.transforms.bijections import Bijection


class OrthogonalSqueeze2d(Bijection):
    def __init__(self, in_channels):
        super(OrthogonalSqueeze2d, self).__init__()
        self.in_channels = in_channels

        self.weight = torch.nn.Parameter(self._initialize_kernels())

    def _initialize_kernels(self):
        kernels = []
        for _ in range(self.in_channels):
            w = torch.empty(4, 4)
            torch.nn.init.orthogonal_(w)
            kernels.append(w.reshape(4, 1, 2, 2))
        return torch.cat(kernels, dim=0)

    def forward(self, x):
        z = F.conv2d(x, self.weight, bias=None, stride=2, groups=self.in_channels)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return z, ldj

    def inverse(self, z):
        x = F.conv_transpose2d(z, self.weight, bias=None, stride=2, groups=self.in_channels)
        return x

    def compute_regularization(self):
        total = 0.
        kernels = torch.chunk(self.weight, self.in_channels, dim=0)
        for kernel in kernels:
            w = kernel.reshape(4, 4)
            total += (torch.matmul(w, w.T) - torch.eye(4).to(w)).abs().mean()
        return total / len(kernels)