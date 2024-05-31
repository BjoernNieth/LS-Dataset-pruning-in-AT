import numpy as np
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
from PIL import Image


class CustomDataset(Dataset):
    """Custom dataset for the pruned data"""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self._toPil = transforms.ToPILImage()

    def __getitem__(self, index):
        """Get a sample of the dataset by its index If this function is used for training, make sure that the
           transforms contain torchvision.transforms.ToTensor """
        # Transform the image to a PIL image to apply transforms
        img = self._toPil(self.data[index])
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[index]
        return img, label



    def __len__(self):
        return len(self.data)


class AugmentedDatasetSampler(Sampler):
    """Get a sampler for the AugmentedCustomDataset class
       This samples returns the same number of batches as a sampler on the original dataset would do"""

    def __init__(self, len_orig, len_augmented, batchsize, augmented_percentage):
        # Store the ids of the original and augmented data
        self.orginal_inds = np.arange(0, len_orig)
        self.augmented_inds = np.arange(len_orig, len_augmented)

        # Percentage of augmented data in each batch
        self.augmented_percentage = augmented_percentage

        # Number of total returned batches
        self.batches = int(np.ceil(len_orig / batchsize))
        # Number of batches containing a full batchsize batch
        # This value is either self.batches - 1 or equal to self.batches
        self.batches_full = np.floor(len_orig / batchsize)
        self.batchsize = batchsize

    def __iter__(self):
        """Iterator function yielding the next batch"""

        # Shuffle the indices to get new permutations each epoch
        np.random.shuffle(self.orginal_inds)
        np.random.shuffle(self.augmented_inds)

        returned_batches = 0
        # Calculate the augmented and original samples per batch
        # Use ceil and floor to make sure batch contains exactly batchsize samples
        original_samples = int(np.ceil(self.batchsize * (1 - self.augmented_percentage)))
        augmented_samples = int(np.floor(self.batchsize * self.augmented_percentage))
        # Return batches until no further full batch can be returned
        while returned_batches < self.batches_full:
            # Take the first n new elements of the shuffled indices as indices for the batch
            batch_original = self.orginal_inds[original_samples * returned_batches:
                                               original_samples * (returned_batches + 1)]
            batch_augmented = self.augmented_inds[augmented_samples * returned_batches:
                                                  augmented_samples * (returned_batches + 1)]
            batch = np.concatenate((batch_original, batch_augmented))
            # Shuffle the batch to avoid first part always being original, second being augmented
            np.random.shuffle(batch)
            yield batch.tolist()

            returned_batches += 1

        # Check if a last, not full-sized batch should be returned
        if self.batches - self.batches_full == 1:
            # Calculate size of last batch
            samples_last_batch = self.orginal_inds.shape[0] - self.batchsize * returned_batches
            # Calculate samples for augmented and original according to last batch size
            original_samples_last = int(np.ceil(samples_last_batch * (1 - self.augmented_percentage)))
            augmented_samples_last = int(np.floor(samples_last_batch * self.augmented_percentage))

            batch_original = self.orginal_inds[original_samples * returned_batches:
                                               original_samples * returned_batches + original_samples_last]
            batch_augmented = self.orginal_inds[augmented_samples * returned_batches:
                                                augmented_samples * returned_batches + augmented_samples_last]
            batch = np.concatenate((batch_original, batch_augmented))
            np.random.shuffle(batch)
            yield batch.tolist()
            returned_batches += 1

    def __len__(self):
        return self.batches


class AugmentedCustomDataset(Dataset):
    """Dataset class to store augmented and original data
       This class should be used with a custom batch sampler"""

    def __init__(self, data_orgi, targets_orgi, data_augmented, targets_augmented, transform=None):
        self.transform = transform
        self._toPil = transforms.ToPILImage()

        # Store the length of the augmented and original data
        self.orgi_len = data_orgi.shape[0]
        self.augmented_len = data_augmented.shape[0]

        # The augmented and original data and targets are stored each as one array
        self.data = np.concatenate((data_orgi, data_augmented))
        self.targets = np.concatenate((targets_orgi, targets_augmented))

    def __getitem__(self, index):
        """Get a sample of the dataset by its index If this function is used for training, make sure that the
           transforms contain torchvision.transforms.ToTensor """
        # Transform the image to a PIL image to apply transforms
        img = self._toPil(self.data[index])
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[index]
        return img, label

    def __len__(self):
        return self.orgi_len
