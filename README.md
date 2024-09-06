# Large-Scale Dataset Pruning in Adversarial Training via Data Importance Score Extrapolation

This repository contains the source code for the paper "Large-Scale Dataset Pruning in Adversarial Training via Data Importance Score
Extrapolation".

## Project structure

- core: contains source code of the project
- data_files: Here the datafiles for the experiments are stored. Code will automatically download CIFAR-10
  - dynamic_uncertainty: Here the calculated Dynamic Uncertainty scores are stored
- results: Here the results of the experiments are stored
- train_main.py: Training script to run experiments
- extrapolate.py: Script used for the KNN-extrapolation


## Acknowledgement

The code from this work is based on the [code](https://github.com/wzekai99/DM-Improves-AT) of [Wang et al., 2023](https://arxiv.org/abs/2302.04638).
Their adversarial training code is based on the [PyTorch implementation](https://github.com/imrahulr/adversarial_robustness_pytorch) of [Rebuffi et al., 2021](https://arxiv.org/abs/2103.01946). 

## Requirements
The code was tested using Python 3.10.<br/>
If you are using conda and want to create a new environment for this project run:
```.bash
conda create --name <env> --file ./requirements.txt
```
If you are using pip, you can install the requirements using the following command:
```.bash
pip install -r ./requirements.txt
```
## Repeat the experiments
 
As an example, the command for the 25% pruning experiment on the original CIFAR-10 dataset using the L-inf threat model is provided below: 
```.bash
python ./train_main.py --epochs=400 --dataset=cifar10 --prune-percentage=0.25 --use-adversarial-predictions=True --early-stop-diff=30 --resume-training=False --batch-size=512 --run-name=cifar10_25%PruningAdversarial --only-generated=False --tau=0.995 --keep-class-distribution=False --beta=5.0 --seed=23 --model=wrn-28-10-swish --lr=0.2 --ls=0.1 --weight-decay=5e-4 --scheduler=cosinew --nesterov=True --attack=linf-pgd --clip-value=0  --mart=False --generated-fraction=0.8 --attack-eps=0.03137254901960784 --attack-iter=10 --attack-step=0.00784313725490196  --adversarial=True
```
This code will automatically download the CIFAR-10 dataset.

In the following, the most important hyperparameters to configure the training are explained:
- dataset=cifar10 | cifar10_generated2m_random: Select the dataset for the experiment.
- epochs=Int: Change the number of epochs for the experiment.
- prune-percentage=0-1: Percentage of samples removed from the dataset
- use-adversarial-predictions=Bool: Use adversarial predictions to calculate the pruning scores, if applicable.
- early-stop-diff=Int: Number of epochs without improvement of the robust accuracy on the validation set until the training is stopped.
- resume-training:Bool: Resume a previous training run.
- batch-size=Int: Batch sized used during training.
- run-name=String: Name used to store the results on disk and in MLflow. Also used to resume a rum.
- only-generated:Bool: If true, use the whole synthetic dataset as a training set without the original data.
- tau:0-1: Weight averaging decay.
- keep-class-distribution=Bool: Prune with respect to the class balance.<
- beta=Float: Stability regularization term for TRADES.
- seed=Int: Seed used for random functions.
- model=wrn-28-10-swish|wrn-70-16-swish: Model architecture used.
- lr=Float: Learning rate.
- ls=Float: Lable smoothing. Prediction target 1-ls.
- weight-decay=5e-4: Weight decay during training.
- scheduler: cyclic | step | cosine | cosinew: Scheduler strategy used.
- nesterov=Bool: Use nesterov momentum.
- attack= fgsm | linf-pgd | fgm | l2-pgd | ling-df |l2-df | linf-apgd | l2-apgd: Attack used during training.
- clip-value=float: Gradient norm clipping.
- mart=Bool: Use mart loss.
- generated-fraction=0-1: Percentage of synthetic images used per epoch.
- attack-eps=float: Epsilon used in the threat model.
- attack-iter=Int: Number of steps if multi-step attack selected.
- attack-step=float: Step size for attack.
- adversarial=Bool: Run adversarial or standard training.


## Deduplication setting
Early versions of this code used the deduplication model SSCD during training as a configurable parameter. 
If you want to run deduplication, download the sscd_dics_mixup.classy.pt file from the [SSCD repository](https://github.com/facebookresearch/sscd-copy-detection) and move the file to 'core/sscd/'.
After this step, you can use the --deduplication-threshold hyperparameter to set the deduplication threshold

