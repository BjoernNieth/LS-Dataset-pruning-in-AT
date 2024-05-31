import numpy as np
import os
from core.data.custom_dataset import CustomDataset

# Meta data of the dataset
DATA_DESC = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    # mean and std calculated for 1m dataset
    'mean': [0.50142212, 0.48873451, 0.45627161],
    'std':  [0.25449478, 0.25029943, 0.26565006],
    'is_augmented': True,
}


def load_cifar10_1m(train_transform = None, subsampling=0, prune_base=False):
    """Load 2m cifar10 generated images"""

    augmented = np.load(os.path.join("./data_files","1m_random.npz"))

    x = augmented["image"]
    y = augmented["label"]

    if subsampling != 0:
        indx = np.arange(0, x.shape[0])
        indx = np.random.choice(indx, size=int((1 - subsampling) * x.shape[0]))
        indx = np.array(indx)
        return CustomDataset(x[indx], y[indx], train_transform), None, None

    return CustomDataset(x, y, train_transform), None, None

def load_cifar10_50k_synthetic(train_transform = None, subsampling=0, prune_base=False):
    """Load 50k cifar10 generated images without the 100k set"""

    augmented = np.load(os.path.join("./data_files","50k_random_without100k.npz"))

    x = augmented["image"]
    y = augmented["label"]

    if subsampling != 0:
        indx = np.arange(0, x.shape[0])
        indx = np.random.choice(indx, size=int((1 - subsampling) * x.shape[0]))
        indx = np.array(indx)
        return CustomDataset(x[indx], y[indx], train_transform), None, None

    return CustomDataset(x, y, train_transform), None, None

def load_cifar10_2m(train_transform = None, subsampling=0, prune_base=False):
    """Load 1m cifar10 generated images"""

    augmented = np.load(os.path.join("./data_files","2m_random.npz"))

    x = augmented["image"]
    y = augmented["label"]

    if subsampling != 0:
        indx = np.arange(0, x.shape[0])
        indx = np.random.choice(indx, size=int((1 - subsampling) * x.shape[0]))
        indx = np.array(indx)
        return CustomDataset(x[indx], y[indx], train_transform), None, None

    return CustomDataset(x, y, train_transform), None, None

def load_cifar10_5m(train_transform = None, subsampling=0, prune_base=False):
    """Load 1m cifar10 generated images"""

    augmented = np.load(os.path.join("./data_files","5m_random.npz"))

    x = augmented["image"]
    y = augmented["label"]

    if subsampling != 0:
        indx = np.arange(0, x.shape[0])
        indx = np.random.choice(indx, size=int((1 - subsampling) * x.shape[0]))
        indx = np.array(indx)
        return CustomDataset(x[indx], y[indx], train_transform), None, None

    return CustomDataset(x, y, train_transform), None, None

def load_cifar10_20m(train_transform = None, subsampling=0, prune_base=False):
    """Load 1m cifar10 generated images"""

    augmented = np.load(os.path.join("./data_files","20m_random.npz"))

    x = augmented["image"]
    y = augmented["label"]

    if subsampling != 0:
        indx = np.arange(0, x.shape[0])
        indx = np.random.choice(indx, size=int((1 - subsampling) * x.shape[0]))
        indx = np.array(indx)
        return CustomDataset(x[indx], y[indx], train_transform), None, None

    return CustomDataset(x, y, train_transform), None, None

def load_cifar10_1m_Certainty (train_transform = None, subsampling=0, prune_base=False):
    """Load 1m cifar10 generated images"""

    augmented = np.load(os.path.join("./data_files","1m_certainty.npz"))

    x = augmented["image"]
    y = augmented["label"]

    if subsampling != 0:
        indx = np.arange(0, x.shape[0])
        indx = np.random.choice(indx, size=int((1 - subsampling) * x.shape[0]))
        indx = np.array(indx)
        return CustomDataset(x[indx], y[indx], train_transform), None, None

    return CustomDataset(x, y, train_transform), None, None

def load_cifar10_500k_Certainty (train_transform = None, subsampling=0, prune_base=False):
    """Load 1m cifar10 generated images"""

    augmented = np.load(os.path.join("./data_files","500k_certainty.npz"))

    x = augmented["image"]
    y = augmented["label"]

    if subsampling != 0:
        indx = np.arange(0, x.shape[0])
        indx = np.random.choice(indx, size=int((1 - subsampling) * x.shape[0]))
        indx = np.array(indx)
        return CustomDataset(x[indx], y[indx], train_transform), None, None

    return CustomDataset(x, y, train_transform), None, None

def load_cifar10_100k(train_transform = None, subsampling=0, prune_base=False):
    """Load 100k cifar10 generated images"""

    augmented = np.load(os.path.join("./data_files","100k_random.npz"))

    x = augmented["image"]
    y = augmented["label"]

    if subsampling != 0:
        indx = np.arange(0, x.shape[0])
        indx = np.random.choice(indx, size=int((1-subsampling) * x.shape[0]))
        indx = np.array(indx)
        return CustomDataset(x[indx], y[indx], train_transform), None, None

    return CustomDataset(x, y, train_transform), None, None