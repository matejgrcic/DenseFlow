import torch
from torchvision.datasets import CIFAR100
from denseflow.data import DATA_PATH


class CIFAR100Dataset(CIFAR100):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        super(CIFAR100Dataset, self).__init__(root,
                                                  train=train,
                                                  transform=transform,
                                                  download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return super(CIFAR100Dataset, self).__getitem__(index)
