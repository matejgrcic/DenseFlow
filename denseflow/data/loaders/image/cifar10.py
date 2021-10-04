from denseflow.data.datasets.image import UnsupervisedCIFAR10, CIFAR10Dataset, CIFAR10SemiSupDataset, CIFAR10PerturbedDataset
from torchvision.transforms import Compose, ToTensor
from denseflow.data.transforms import Quantize
from denseflow.data import TrainTestLoader, DATA_PATH


class CIFAR10(TrainTestLoader):
    '''
    The CIFAR10 dataset of (Krizhevsky & Hinton, 2009):
    https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[]):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor(), Quantize(num_bits)]
        trans_test = [ToTensor(), Quantize(num_bits)]

        # Load data
        self.train = UnsupervisedCIFAR10(root, train=True, transform=Compose(trans_train), download=download)
        self.test = UnsupervisedCIFAR10(root, train=False, transform=Compose(trans_test))

class CIFAR10Supervised(TrainTestLoader):
    '''
    The CIFAR10 dataset of (Krizhevsky & Hinton, 2009):
    https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''

    def __init__(self, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[]):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor()]
        trans_test = [ToTensor()]

        # Load data
        self.train = CIFAR10Dataset(root, train=True, transform=Compose(trans_train), download=download)
        self.test = CIFAR10Dataset(root, train=False, transform=Compose(trans_test))


class CIFAR10SemiSup(TrainTestLoader):
    '''
    The CIFAR10 dataset of (Krizhevsky & Hinton, 2009):
    https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    '''

    def __init__(self, num_samples, root=DATA_PATH, download=True, num_bits=8, pil_transforms=[]):

        self.root = root

        # Define transformations
        trans_train = pil_transforms + [ToTensor()]
        trans_test = [ToTensor()]

        # Load data
        self.train_sup = CIFAR10SemiSupDataset(num_samples, root, train=True, transform=Compose(trans_train), download=download)
        self.train_unsup = CIFAR10Dataset(root, train=True, transform=Compose(trans_train))
        self.test = CIFAR10Dataset(root, train=False, transform=Compose(trans_test))