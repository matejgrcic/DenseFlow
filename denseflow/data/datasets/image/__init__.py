from .imagenet32 import ImageNet32Dataset
from .imagenet64 import ImageNet64Dataset
from .celeba import CelebADataset
from .cifar10 import CIFAR10Dataset, CIFAR10SemiSupDataset, CIFAR10PerturbedDataset
from .svhn import SVHNDataset
from .cifar100 import CIFAR100Dataset

from .fixed_binarized_mnist import FixedBinaryMNISTDataset
from .omniglot import OMNIGLOTDataset

from .unsupervised_wrappers import *
