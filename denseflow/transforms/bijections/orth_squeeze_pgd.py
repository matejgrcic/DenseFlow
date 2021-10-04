import torch
import torch.nn.functional as F
from denseflow.transforms.bijections import Bijection
from denseflow.utils import orthogonalize_tensor

class OrthogonalSqueeze2dPgd(Bijection):
    def __init__(self, in_channels):
        super(OrthogonalSqueeze2dPgd, self).__init__()
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

    def orthogonalize_kernel(self):
        with torch.no_grad():
            updated_kernels = []
            kernels = torch.chunk(self.weight, self.in_channels, dim=0)
            for kernel in kernels:
                # a = kernel.reshape(4, 4)
                # print((torch.matmul()))
                w = orthogonalize_tensor(kernel.reshape(4, 4))
                updated_kernels.append(w.reshape(4, 1, 2, 2))

            self.weight.data = torch.cat(updated_kernels, dim=0).to(self.weight)