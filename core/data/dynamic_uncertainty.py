import os
import numpy as np
import pickle
from .custom_dataset import CustomDataset

# J = 10 from Dynamic Uncertainty paper
# J describes size of window for which uncertainty is calculated
J = 10
def pruneDataset(data, pruned_ids, predictions_path, prune_percentage, output_path, transforms, use_adversarial_predictions=False,
                 adversarial=False, keep_class_distribution=False, no_overfitted_epochs=False, fundamental_frequency_pruning=False):
    """Apply Dynamic Uncertainty Pruning to a dataset given its prediction for each sample during training"""

    # Check if the results are already calculated
    if not os.path.isdir(output_path):
        if no_overfitted_epochs:
            if use_adversarial_predictions and adversarial:
                if fundamental_frequency_pruning:
                    ranking_name = "frequency_ranking_adversarial_no_overfitting.npy"
                else:
                    ranking_name = "ranking_adversarial_no_overfitting.npy"
                predictions_name = "predictions_adversarial.npy"
            else:
                if fundamental_frequency_pruning:
                    ranking_name = "frequency_ranking_no_overfitting.npy"
                else:
                    ranking_name = "ranking_no_overfitting.npy"
                predictions_name = "predictions.npy"

        else:
            if use_adversarial_predictions and adversarial:
                if fundamental_frequency_pruning:
                    ranking_name = "frequency_ranking_adversarial.npy"
                else:
                    ranking_name = "ranking_adversarial.npy"
                predictions_name = "predictions_adversarial.npy"
            else:
                if fundamental_frequency_pruning:
                    ranking_name = "frequency_ranking.npy"
                else:
                    ranking_name = "ranking.npy"
                predictions_name = "predictions.npy"

        # Check if Dynamic Uncertainty scores are already calculated
        if not os.path.isfile(os.path.join(predictions_path, ranking_name)):
            # Load the predictions of the model
            if no_overfitted_epochs:
                with open(os.path.join(predictions_path, "meta_dict.pkl"), 'rb') as f:
                    best_epoch = pickle.load(f)['best epoch']
            else:
                best_epoch = -1
            predictions = np.load(os.path.join(predictions_path, predictions_name))
            if best_epoch > 0:
                predictions = predictions[0:best_epoch]

            if fundamental_frequency_pruning:
                x = np.fft.fft(predictions, axis=0)
                # Get fundamental frequency of FFT
                scores = np.abs(x[1:]).sum(axis=0)
            else:
                K = predictions.shape[0]
                Dyn_Un = np.zeros((K-J, predictions.shape[1]))

                # Iterate over dataset and calculate for each window of size J the uncertainty
                for i in range(0, K - J):
                    # Get mean prediction value for each sample in J-sized window
                    p_mean = predictions[i: i + J].sum(axis=0)/J
                    # Calculate the     mean standard deviation for each sample in J
                    u_k = np.sqrt(np.square(predictions[i: i + J] - p_mean).sum(0)/(J-1))
                    Dyn_Un[i] = u_k
                # Calculate the mean for each samples uncertainty
                scores = Dyn_Un.sum(0)/(K-J)

            np.save(os.path.join(predictions_path, ranking_name), scores)
        else:
            scores = np.load(os.path.join(predictions_path, ranking_name))

        # Discard all samples already pruned in earlier pruning algorithms
        ranking = np.delete(scores, pruned_ids)


        if keep_class_distribution:
            prune_ids = []
            for i in np.unique(data.targets):
                class_ids = (data.targets == i).nonzero()[0]
                ranking_class = ranking[class_ids]
                ranking_class_sorted = np.argsort(ranking_class).astype(int)

                prune_ids = np.concatenate([prune_ids, class_ids[ranking_class_sorted[0: int(ranking_class_sorted.shape[0] *
                                                                                    prune_percentage)]]])
            prune_ids = np.array(prune_ids).astype(int)


        else:
            # Use the Dynamic Uncertainty as ranking for the pruning and sort it descending
            ranking = np.argsort(ranking)

            prune_ids_ranking = range(0, int(prune_percentage * len(ranking)))
            # Select the prune_percentage% samples with the highest score and remove them
            prune_ids = ranking[prune_ids_ranking]

        pruned_dataset_x = np.delete(data.data, prune_ids, axis=0)
        pruned_dataset_y = np.delete(data.targets, prune_ids, axis=0)

        # Save the results in the defined output location
        os.mkdir(output_path)
        np.save(os.path.join(output_path, "train_data_x"), pruned_dataset_x)
        np.save(os.path.join(output_path, "train_data_y"), pruned_dataset_y)
        with open(os.path.join(output_path, 'pruned_ids.pkl'), 'wb') as f:
            pickle.dump(prune_ids, f)
    else:
        # Load the results from disk
        pruned_dataset_x = np.load(os.path.join(output_path, "train_data_x.npy"))
        pruned_dataset_y = np.load(os.path.join(output_path, "train_data_y.npy"))

    # Return the dataset as a custom dataset
    dataset = CustomDataset(pruned_dataset_x, pruned_dataset_y, transforms)

    return dataset

