import torch
from torchvision.datasets import SVHN
from denseflow.data import DATA_PATH


class SVHNDataset(SVHN):
    def __init__(self, root=DATA_PATH, split='train', transform=None, download=False):
        super(SVHNDataset, self).__init__(root,
                                               split=split,
                                               transform=transform,
                                               download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return super(SVHNDataset, self).__getitem__(index)
