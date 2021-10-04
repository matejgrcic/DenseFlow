import torch
from torchvision.datasets import CIFAR10
from denseflow.data import DATA_PATH
import numpy as np
import torchvision.transforms as tf


class CIFAR10Dataset(CIFAR10):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        super(CIFAR10Dataset, self).__init__(root,
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
        return super(CIFAR10Dataset, self).__getitem__(index)

class CIFAR10PerturbedDataset(CIFAR10):
    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        super(CIFAR10PerturbedDataset, self).__init__(root,
                                                  train=train,
                                                  transform=transform,
                                                  download=download)
        self.perturbations = tf.Compose([
            tf.ToPILImage(),
            tf.ColorJitter(brightness=(0.25, 0.25), contrast=(0.25, 2), saturation=(0.25, 2), hue=0.25),
            tf.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x, _ = super(CIFAR10PerturbedDataset, self).__getitem__(index)
        return x, self.perturbations(x)


class CIFAR10SemiSupDataset(CIFAR10):
    def __init__(self, num_samples, root=DATA_PATH, train=True, transform=None, download=False):
        super(CIFAR10SemiSupDataset, self).__init__(root,
                                                  train=train,
                                                  transform=transform,
                                                  download=download)

        self.num_classes = 10
        assert num_samples % self.num_classes == 0
        self.num_samples = num_samples

        self.total_num_samples = super(CIFAR10SemiSupDataset, self).__len__()
        self.indices_map = self._load_samples_subset()
        self.bucket_size = int(self.num_samples / self.num_classes)


    def _load_samples_subset(self):
        items = {}
        for i in range(1, self.num_classes+1):
            items[i] = []
        for i in range(self.total_num_samples):
            _, y = super(CIFAR10SemiSupDataset, self).__getitem__(i)
            items[int(y) + 1].append(i)
        return items

    def __len__(self):
        return self.num_samples


    def __getitem__(self, i):
        class_id = i // self.bucket_size
        sample_id = i % self.bucket_size
        index = self.indices_map[class_id + 1][sample_id]
        x, y = super(CIFAR10SemiSupDataset, self).__getitem__(index)
        # unsup_index = np.random.randint(self.total_num_samples)
        # x_unsup = super(CIFAR10SemiSupDataset, self).__getitem__(unsup_index)[0]
        # return x, y, x_unsup
        return x, y
