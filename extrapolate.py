import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from npy_append_array import NpyAppendArray
import os
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##########################################################
### Configure extrapolation parameters and data        ###
##########################################################

batchsize= 16
## Distance metric called for the KNN
# 'euclidean distance' or 'cosine sim'
metric = 'euclidean distance'
## Weight sample importance with distance
# Set to False because in all experiments yields worse results
scaled_prediction = False
## K neibhours used for KNN
k = 12
## Name of the resulting file where the extrapolated scores are saved
score_name = "ranking_adversarial.npy"
## Path to the result directory
results_path = ""

# Load npy file for the pruning scores of the original data
pruning_scores_orig = np.load("")
# Load npy file for the pruning scores of the synthetic data
pruning_scores_generated = np.load("")

# Load npy file with encodings of original data
encodings_orig = np.load("")
# Load npy file with encodings of synthetic data
encoding_generated = np.load("")
# Load npy file with encodings of inference data
encodings_inference = np.load("").squeeze()

##########################################################
##########################################################

# Concetenate the original and synthetic data to get the train dataset
train_encoding = np.concatenate([encodings_orig, encoding_generated])
train_dyn_unc = np.concatenate([pruning_scores_orig, pruning_scores_generated])

train_dyn_unc_tensor = torch.from_numpy(train_dyn_unc.astype(np.float32)).to(device)
train_sscd_tensor = torch.from_numpy(train_encoding.astype(np.float32)).to(device)
inference_tensor = torch.from_numpy(encodings_inference)
inference_loader = DataLoader(inference_tensor, batch_size=batchsize, shuffle=False)

if not os.path.exists(results_path):
    os.makedirs(results_path)

predicted_scores = []
print("Start Inference")
with NpyAppendArray(os.path.join(results_path, score_name), delete_if_exists=True) as npaa:
    with torch.no_grad():
        for inference_batch in inference_loader:
            inference_batch = inference_batch.to(device)
            if metric == 'cosine sim':
                # Calculate Cosine similarity of SSCD scores for each sample from batch to every other sample in dataset
                cosines_batch = torch.einsum('ik,dk->id', train_sscd_tensor,
                                             inference_batch) / torch.einsum(
                    'ik,dk->id',
                    torch.norm(train_sscd_tensor, dim=1, keepdim=True),
                    torch.norm(inference_batch, dim=1, keepdim=True))
                top = torch.topk(cosines_batch, k, dim=0)

                if scaled_prediction and k > 1:
                    scale_factors = ((top.values.sum(axis=0) - top.values) / (
                                (k - 1) * top.values.sum(axis=0)))

            elif metric == 'euclidean distance':
                euclidean_batch = torch.linalg.vector_norm(
                    train_sscd_tensor.unsqueeze(1) - inference_batch, dim=2)
                top = torch.topk(euclidean_batch, k, dim=0, largest=False)
                if scaled_prediction and k > 1:
                    scale_factors = top.values / top.values.sum(axis=0)

            if scaled_prediction and k > 1:
                predicted = (train_dyn_unc_tensor[top.indices] * scale_factors).sum(dim=0)
            else:
                # get predicted pruning scores
                predicted = train_dyn_unc_tensor[top.indices].mean(dim=0)

            predicted_scores.append(predicted.cpu().numpy())
            # After 25000 samples write result to disk
            if len(predicted_scores) > 25000:
                results = np.concatenate(predicted_scores)
                npaa.append(results)
                predicted_scores = []


        results = np.concatenate(predicted_scores)
        npaa.append(results)
