from pathlib import Path
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST

DATA_ROOT = Path('/scratch1/chofer/data/')\

RESULTS_ROOT = Path('./results')

DS_PATH_CFG = {
    'cifar100_train':
        (CIFAR100, {'root': DATA_ROOT / 'cifar100', 'train': True}),
    'cifar100_test':
        (CIFAR100, {'root': DATA_ROOT / 'cifar100', 'train': False}),

}


# DS_SPLIT_CFG = {
#     'cifar100_train': [500, 1000, 2500, 5000]
# }
