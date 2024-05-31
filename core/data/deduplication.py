import os.path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from core.utils.graph import Graph
from core.data.custom_dataset import CustomDataset
import sys
import numpy as np
import pickle
import torch.nn as nn
from .custom_dataset import CustomDataset
import itertools
from npy_append_array import NpyAppendArray
import random
from core.sscd.model import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_sscd_score(data, sscd_score_path, mean, std):
    """Calculate SSCD scores using external SSCD repo"""

    # Skew transform and normalization for better results from SSCD
    skew_320 = transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std,),
    ])

    # Read data and apply transforms
    train_data = data[0]
    train_data = CustomDataset(train_data.data, train_data.targets, skew_320)
    # Use pretrained SSCD model
    model = Model("CV_RESNET50", 512, 3.0)
    weights = torch.load("./sscd/sscd_disc_mixup.classy.pt")
    model.load_state_dict(weights)
    model.eval()
    model = nn.DataParallel(model)
    model = model.to(device)

    # Use non-shuffled dataloader to track results of each x
    data_loader = DataLoader(train_data, batch_size=100, shuffle=False)
    sscd = []
    os.mkdir(sscd_score_path)
    with torch.no_grad():
        with NpyAppendArray(sscd_score_path + "/sscd_scores.npy", delete_if_exists=True) as npaa:
            # Distinguish between GPU and CPU calculation because samples from GPU
            # need to be returned to CPU for later use
            if device == "cuda":
                print("GPU")
                # Use enumerate on non-shuffled dataloader to get original index of sample
                for (idx, (x, _)) in enumerate(data_loader):
                    sscd.append(model(x.cuda()).cpu().numpy())

                    if (idx+1) % 500 == 0:
                        results = np.concatenate(sscd)
                        sscd = []
                        npaa.append(results)
            else:
                print("CPU")
                for (idx, (x, _)) in enumerate(data_loader):
                    sscd.append(model(x.cuda()).numpy())

                    if (idx+1) % 500 == 0:
                        results = np.concatenate(sscd)
                        sscd = []
                        npaa.append(results)

    results = np.concatenate(sscd)
    npaa.append(results)

def deduplicate_dataset(data, deduplication_data_path, sscd_score_path, duplication_threshold):
    """Calculate the deduplicated dataset based on SSCD scores and deduplication threshold and save results to disk"""
    train_data, test_data, val_data = data

    # Load previous calculated SSCD scores
    sscd_scores = np.load(sscd_score_path + "/sscd_scores.npy")
    samples = sscd_scores.shape[0]

    # Get transform numpy dataset to torch dataset
    dataset = torch.from_numpy(sscd_scores.astype(np.float16)).to(device)
    batchsize = 1024

    # Use unshuffled dataloader to retrieve index of sample
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    print("Calculate cosine scores")
    # Save all duplicates in a graph to later only use one sample per component
    duplicates = Graph()

    # Calculates highest similarity score for each sample
    highest_cosine = torch.zeros(samples).half().to(device)

    # Offset to keep track of starting position of current batch in original data
    offset = 0

    # Save duplicate ids and their similarity score
    duplicates_idx = []
    duplicates_value = []
    # Use no gradient for less-memory consumption
    with torch.no_grad():
        for batch in loader:
            # Calculate Cosine similarity of SSCD scores for each sample from batch to every other sample in dataset
            cosines_batch = torch.einsum('ik,dk->id', dataset.data, batch)

            # Fill similarity scores of sample with itself with 0, since they are no duplicates
            # The diagonal is located at the position of the batch in the original dataset
            cosines_batch[offset:].fill_diagonal_(0)

            # Get a mask for all duplicates
            mask = cosines_batch > duplication_threshold
            # Extract the ids of the duplicates as numpy arrays where [id in dataset, id in batch]
            idx = mask.nonzero()
            # Append the similarity scores for the duplicates
            duplicates_value.append(cosines_batch[mask])

            # Apply the offset to second dimension to map id in batch to global id in dataset
            idx[:, 1] = idx[:, 1] + offset
            duplicates_idx.append(idx)
            offset = offset + batchsize

            # Get the highest similarity for each sample in the dataset
            max_cosine_batch = torch.max(cosines_batch, 1)
            # Check if the current highest similarity is higher than previous highest similarity for each sample
            highest_cosine_mask = max_cosine_batch.values > highest_cosine
            # If so change save new highest cosine similarity for these samples
            highest_cosine[highest_cosine_mask] = max_cosine_batch.values[highest_cosine_mask]

        # Transform highest cosine tensor to numpy
        if device == "cuda":
            # If tensor on GPU, first retrieve it to CPU
            highest_cosine = highest_cosine.cpu().numpy()
        else:
            highest_cosine = highest_cosine.numpy()

        os.mkdir(deduplication_data_path)
        np.save(deduplication_data_path + "/highest_cosine.npy", highest_cosine)

        # Save graph of duplicates as sparse matrix for possible altes use
        adjacency_matrix = torch.sparse_coo_tensor(torch.transpose(torch.cat(duplicates_idx, 0), 1, 0),
                                                   torch.cat(duplicates_value, 0))
        torch.save(adjacency_matrix, os.path.join(deduplication_data_path, "adjacency_matrix.pt"))

    # Get all duplicates ids as a numpy 2d array
    if device == "cuda":
        duplicates_idx_np = torch.cat(duplicates_idx, 0).cpu().numpy()
    else:
        duplicates_idx_np = torch.cat(duplicates_idx, 0).numpy()

    # Construct graph for duplicates
    for double_ids in duplicates_idx_np:
        duplicates.add_connection(double_ids[0], double_ids[1])

    with open(os.path.join(deduplication_data_path, "Graph"), 'wb') as graph_file:
        pickle.dump(duplicates, graph_file)

    print("Calculate Pruning")

    # Prune by selecting one node from each graph component
    double_components = duplicates.get_subgraphs_nodes()
    # Get list of all duplicate ids
    all_duplicates = list(itertools.chain.from_iterable([list(x) for x in double_components]))
    # Select one random node from each graph component to not prune
    keep_duplicates = [random.choice(list(x)) for x in double_components]
    # Discard every other node of the graph
    double_ids = [x for x in all_duplicates if x not in keep_duplicates]

    # Remove the nodes from the trainings data and labels
    deduplicated_train_x = np.delete(train_data.data, double_ids, axis=0)
    deduplicated_train_y = np.delete(np.array(train_data.targets), double_ids, axis=0)

    # Save pruning information for later analysis
    pruning_stats={
        "keep_duplicates": keep_duplicates,
        "double_ids": double_ids
    }
    with open(os.path.join(deduplication_data_path, 'pruning_stats.pkl'), 'wb') as f:
        pickle.dump(pruning_stats, f)

    np.save(deduplication_data_path + "/deduplicated_train_x.npy", deduplicated_train_x)
    np.save(deduplication_data_path + "/deduplicated_train_y.npy", deduplicated_train_y)

    # If the selected dataset contains test data, also save it
    if test_data is not None:
        np.save(deduplication_data_path + "/test_x.npy", test_data.data)
        np.save(deduplication_data_path + "/test_y.npy", np.array(test_data.targets))

    # If the selected dataset contains validation data, also save it
    #if test_data is not None:
    #    np.save(deduplication_data_path + "/val_x.npy", val_data.data)
    #    np.save(deduplication_data_path + "/val_y.npy", np.array(val_data.targets))



def get_deduplicate_dataset(dataset_folder, data_set, read_data, data_desc, deduplication, train_transforms, deduplication_data_path):
    """Returns train dataset and if available test dataset while applying SSCD deduplication to train dataset"""

    print(data_set)
    # Check if SSCD scores for the dataset are already available
    sscd_score_path = os.path.join(dataset_folder, data_set + "_deduplication")
    if not os.path.isdir(sscd_score_path):
        print("Calculate SSCD scores")
        # Calculate the SSCD scores and save them in sscd_score_path
        calculate_sscd_score(read_data, sscd_score_path, data_desc["mean"], data_desc["std"])

    # Check if deduplication results for dataset + pruning_threshold are already available
    if not os.path.isdir(deduplication_data_path):
        print("Dataset " + data_set + " for threshold " + str(deduplication) + " not yet deduplicated")
        deduplicate_dataset(read_data, deduplication_data_path, sscd_score_path, deduplication)

    # After above steps, the pruned dataset is expected at the following locations
    train_x = np.load(os.path.join(deduplication_data_path, "deduplicated_train_x.npy"))
    train_y = np.load(os.path.join(deduplication_data_path, "deduplicated_train_y.npy"))

    # Retrieve the pruned samples for use in later pruning functions
    with open(os.path.join(deduplication_data_path, "pruning_stats.pkl"), "rb") as f:
        double_ids = pickle.load(f)["double_ids"]

    train_dataset = CustomDataset(train_x, train_y, train_transforms)
    # If the dataset is not the augmentation dataset also return test dataset
    if data_desc["is_augmented"] is False:
        print("load test of " + data_set)
        # Although not pruned, the test dataset is also saved with the pruned results
        test_x = np.load(os.path.join(deduplication_data_path, "test_x.npy"))
        test_y = np.load(os.path.join(deduplication_data_path, "test_y.npy"))
        test_dataset = CustomDataset(test_x, test_y, transforms.ToTensor())

        return train_dataset, test_dataset, double_ids
    else:
        # If augmentation dataset, return None as test dataset
        return train_dataset, None, double_ids
