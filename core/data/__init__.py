## Code took insperation from:
## Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., and Yan, S.
## Better diffusion models further improve adversarial training, June 2023
## Code available at https://github.com/wzekai99/DM-Improves-AT

import os
import shutil

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#import mlflow
from .cifar10 import load_cifar10
from .cifar10_augmented import load_cifar10_1m, load_cifar10_2m, load_cifar10_100k, load_cifar10_1m_Certainty, \
    load_cifar10_5m, load_cifar10_500k_Certainty, load_cifar10_20m, load_cifar10_50k_synthetic
from .custom_dataset import CustomDataset, AugmentedCustomDataset, AugmentedDatasetSampler

from core.data.dynamic_uncertainty import pruneDataset

from .deduplication import get_deduplicate_dataset

# Dict containing isAugmented and load-function for dataset
# In augmented case, base dataset is provided as tuple with load function of augmented dataset
DATASETS = {
    "cifar10": (False, load_cifar10),
    "cifar10_generated50k_random": (True, ("cifar10", load_cifar10_50k_synthetic)),
    "cifar10_generated1m_random": (True, ("cifar10", load_cifar10_1m)),
    "cifar10_generated2m_random": (True, ("cifar10", load_cifar10_2m)),
    "cifar10_generated5m_random": (True, ("cifar10", load_cifar10_5m)),
    "cifar10_generated20m_random": (True, ("cifar10", load_cifar10_20m)),
    "cifar10_generated100k": (True, ("cifar10", load_cifar10_100k)),
    "cifar10_generated1m_certainty": (True, ("cifar10", load_cifar10_1m_Certainty)),
    "cifar10_generated500k_certainty": (True, ("cifar10", load_cifar10_500k_Certainty))
}
dataset_folder = "./data_files"
num_workers=4

def get_data_info(dataset):
    """
    Returns dataset information.
    Arguments:
        dataset (str): name of the dataset
    """
    # if 'cifar100' in data_dir:
    #    from .cifar100 import DATA_DESC
    if 'cifar10' == dataset:
        from .cifar10 import DATA_DESC
    elif 'cifar10_generated50k_random' == dataset:
        from .cifar10_augmented import DATA_DESC
    elif 'cifar10_generated1m_random' == dataset:
        from .cifar10_augmented import DATA_DESC
    elif 'cifar10_generated2m_random' == dataset:
        from .cifar10_augmented import DATA_DESC
    elif 'cifar10_generated5m_random' == dataset:
        from .cifar10_augmented import DATA_DESC
    elif 'cifar10_generated20m_random' == dataset:
        from .cifar10_augmented import DATA_DESC
    elif 'cifar10_generated1m_certainty' == dataset:
        from .cifar10_augmented import DATA_DESC
    elif 'cifar10_generated100k' == dataset:
        from .cifar10_augmented import DATA_DESC
    elif 'cifar10_generated500k_certainty' == dataset:
        from .cifar10_augmented import DATA_DESC
    # elif 'svhn' in data_dir:
    #    from .svhn import DATA_DESC
    # elif 'tiny-imagenet' in data_dir:
    #    from .tiny_imagenet import DATA_DESC
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    DATA_DESC['data'] = dataset
    return DATA_DESC


def load_data(data_set, predictions_path=None, deduplication_threshold=None, subsampling=0, prune_percentage=None,
              batch_size=128, batch_size_validation=128, generated_fraction=0.8, prune_base=False, keep_class_distribution=False,
              use_normal_predictions=False, use_adversarial_predictions=False, only_generated=False, adversarial=False,
              no_overfitted_epochs=False, fundamental_frequency_pruning=False):
    """Load and prune the specified datasets and augment the trainings-set if specified"""

    isAugmented, read_data = DATASETS[data_set]

    # Common transform from better diffusion
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])

    # Distinguish between augmented and non-augmented case
    if not isAugmented:
        # Get meta information about the dataset
        data_desc = get_data_info(data_set)
        # Check for SSCD Pruning/SSCD deduplication
        if deduplication_threshold is not None:
            # Configure output path for pruned data based on dataset name and deduplication threshold
            output_path = os.path.join(dataset_folder, data_set + "_deduplication_" + str(deduplication_threshold))

            train_dataset, test_dataset, pruned_ids = get_deduplicate_dataset(dataset_folder, data_set,
                                                                              (read_data(None, subsampling, True)),
                                                                              data_desc, deduplication_threshold,
                                                                              train_transform, output_path)
            #mlflow.log_param("Number duplicates", len(pruned_ids))
            _, _, val_dataset = read_data(train_transform, subsampling, prune_base)
        else:
            # Specify output path if needed for Dynamic Uncertainty pruning
            output_path = os.path.join(dataset_folder, data_set)
            train_dataset, test_dataset, val_dataset = read_data(train_transform, subsampling, True)
            pruned_ids = []

        # Only prune if predictions for dynamic uncertainty calculation are availabe
        if prune_percentage is not None:
            assert predictions_path is not None

            # Output path as combination of previous applied pruning and current pruning
            model_string = os.path.basename(os.path.normpath(predictions_path))
            if fundamental_frequency_pruning:
                output_path = output_path + "_Fundamental_Frequency_" + str(prune_percentage) + "_no_overfitting_epochs_" + \
                              str(no_overfitted_epochs) + "_" + "use_adversarial_predictions_" + \
                              str(use_adversarial_predictions) + "_" + model_string
            else:
                output_path = output_path + "_Dyn_Un_" + str(prune_percentage) + "_no_overfitting_epochs_" + \
                              str(no_overfitted_epochs) + "_" + "use_adversarial_predictions_" + \
                              str(use_adversarial_predictions) + "_" + model_string
            if use_normal_predictions:
                output_path = output_path + "_use_normal_predictions"
            train_dataset = pruneDataset(train_dataset, pruned_ids, predictions_path,
                                         prune_percentage, output_path, train_transform, use_adversarial_predictions,
                                         adversarial, keep_class_distribution, no_overfitted_epochs, fundamental_frequency_pruning)

        # Use standard dataloader from torch
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_validation, shuffle=False, num_workers=num_workers)

        return train_dataset, test_dataset, val_dataset, train_loader, test_loader, val_loader
    else:
        # Get original and generated dataset as trainings set

        # In augmented case meta information for both datasets need to be retrieved
        base_name, read_augmented = read_data
        _, read_base = DATASETS[base_name]

        # Get meta information about both datasets
        data_desc_base = get_data_info(base_name)
        data_desc_augmented = get_data_info(data_set)

        if deduplication_threshold is not None:
            if prune_base:
                output_path_base = os.path.join(dataset_folder,
                                                base_name + "_deduplication_" + str(deduplication_threshold))
                train_dataset_base, test_dataset_base, duplicates_base = get_deduplicate_dataset(dataset_folder,
                                                                                                 base_name,
                                                                                                 (read_base(None, subsampling, prune_base)),
                                                                                                 data_desc_base,
                                                                                                 deduplication_threshold,
                                                                                                 train_transform,
                                                                                                 output_path_base)
            else:
                output_path_base = os.path.join(dataset_folder, base_name)
                train_dataset_base, test_dataset_base, val_dataset = read_base(train_transform, subsampling, prune_base)

            output_path_augmented = os.path.join(dataset_folder,
                                                 data_set + "_deduplication_" + str(deduplication_threshold))
            # Augmented data is not used for testing
            train_dataset_augmented, _, duplicates_augmented = get_deduplicate_dataset(dataset_folder, data_set,
                                                                                       (read_augmented(None, subsampling, prune_base)),
                                                                                       data_desc_augmented,
                                                                                       deduplication_threshold,
                                                                                       train_transform,
                                                                                       output_path_augmented)
            #mlflow.log_param("Number duplicates", len(duplicates_augmented))
        else:
            output_path_base = os.path.join(dataset_folder, base_name)
            train_dataset_base, test_dataset_base, val_dataset = read_base(train_transform, subsampling, prune_base)

            output_path_augmented = os.path.join(dataset_folder, data_set)
            # Augmented dataset is not used for testing
            train_dataset_augmented, _, _ = read_augmented(train_transform, subsampling, prune_base)
            duplicates_augmented, duplicates_augmented = ([], [])

        if prune_percentage is not None:
            assert predictions_path is not None
            model_string = os.path.basename(os.path.normpath(predictions_path))
            if fundamental_frequency_pruning:
                output_path_augmented = output_path_augmented + "_Fundamental_Frequency_" + str(
                    prune_percentage) + "_keep_class_balance_" + str(keep_class_distribution) + "_no_overfitting_epochs_" + \
                              str(no_overfitted_epochs) + "_" + "use_adversarial_predictions_" + \
                              str(use_adversarial_predictions) + "_" + model_string
            else:
                output_path_augmented = output_path_augmented + "_Dyn_Un_" + str(prune_percentage)  + "_keep_class_balance_"\
                            + str(keep_class_distribution) + "_no_overfitting_epochs_" + \
                              str(no_overfitted_epochs) + "_" + "use_adversarial_predictions_" + \
                              str(use_adversarial_predictions) + "_" + model_string
            train_dataset_augmented = pruneDataset(train_dataset_augmented, duplicates_augmented, predictions_path,
                                                   prune_percentage, output_path_augmented, train_transform, adversarial,
                                                   use_adversarial_predictions, keep_class_distribution, no_overfitted_epochs,
                                                   fundamental_frequency_pruning)
            if prune_base:
                predictions_path_base = predictions_path.replace(data_set, base_name)
                if fundamental_frequency_pruning:
                    output_path_base = output_path_base + "_Fundamental_Frequency_" + str(prune_percentage) + "_keep_class_balance_" + \
                                       str(keep_class_distribution) + "_" + model_string
                else:
                    output_path_base = output_path_base + "_Dyn_Un_" + str(prune_percentage) + "_keep_class_balance_" + \
                                            str(keep_class_distribution) + "_" + model_string
                train_dataset_base = pruneDataset(train_dataset_base, duplicates_base, predictions_path_base,
                                                  prune_percentage, output_path_base, train_transform, adversarial,
                                                   use_adversarial_predictions, keep_class_distribution, no_overfitted_epochs,
                                                  fundamental_frequency_pruning)

        # Use custom dataset class dealing with augmented and original data
        augmented_train_dataset = AugmentedCustomDataset(train_dataset_base.data, train_dataset_base.targets,
                                                         train_dataset_augmented.data, train_dataset_augmented.targets,
                                                         train_transform)

        # Use custom sampler to retrieve for augmented and original data
        augmented_sampler = AugmentedDatasetSampler(train_dataset_base.data.shape[0],
                                                    train_dataset_augmented.data.shape[0],
                                                    batch_size, generated_fraction)
        train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_sampler=augmented_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset_base, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_validation, shuffle=False, num_workers=num_workers)

        if only_generated:
            train_loader_generated = DataLoader(train_dataset_augmented, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            return train_dataset_augmented, test_dataset_base, val_dataset, train_loader_generated, test_loader, val_loader
        else:
            return augmented_train_dataset, test_dataset_base, val_dataset, train_loader, test_loader, val_loader