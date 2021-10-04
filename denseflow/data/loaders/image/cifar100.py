from denseflow.data.datasets.image import CIFAR100Dataset
from torchvision.transforms import Compose, ToTensor
from denseflow.data.transforms import Quantize
from denseflow.data import TrainTestLoader, DATA_PATH


class CIFAR100Supervised(TrainTestLoader):
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
        self.train = CIFAR100Dataset(root, train=True, transform=Compose(trans_train), download=download)
        self.test = CIFAR100Dataset(root, train=False, transform=Compose(trans_test))