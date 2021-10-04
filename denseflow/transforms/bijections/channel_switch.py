import torch
from denseflow.transforms.bijections import Bijection

class SwitchChannels(Bijection):

    def __init__(self):
        super(SwitchChannels, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        ldj = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return torch.cat([x2, x1], dim=1), ldj

    def inverse(self, z):
        x2, x1 = z.chunk(2, dim=1)
        return torch.cat([x1, x2], dim=1)
