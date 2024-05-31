import torchvision
import torchvision.transforms as transforms
import numpy as np
from core.data.custom_dataset import CustomDataset
import torch
DATA_DESC = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465],
    'std': [0.2023, 0.1994, 0.2010],
    'is_augmented': False,
}

def load_cifar10(train_transform = None, subsampling=0, prune_base=False):
    """Load cifar10 dataset"""
    test_transform = transforms.Compose([transforms.ToTensor()])

    if train_transform == None:
        train_transform = test_transform

    cifar_train = torchvision.datasets.CIFAR10("./data_files", train=True, transform=train_transform, download=True)
    train_dataset = CustomDataset(cifar_train.data[1024:], cifar_train.targets[1024:], train_transform)

    if subsampling != 0 and prune_base:
        indx = np.arange(0, train_dataset.data.shape[0])
        indx = np.random.choice(indx, size=int((1-subsampling) * train_dataset.data.shape[0]))
        indx = np.array(indx)
        print(indx)
        train_dataset = CustomDataset(np.array(train_dataset.data)[indx], np.array(train_dataset.targets)[indx], train_transform)

    # Validation split as in Better Diffusion paper
    # We separate first 1024 images of training set as a fixed validation set.
    val_dataset = CustomDataset(cifar_train.data[0:1024], cifar_train.targets[0:1024], test_transform)
    cifar_test = torchvision.datasets.CIFAR10("./data_files", train=False, transform=test_transform, download=True)

    return train_dataset, cifar_test, val_dataset

