## Reused code from:
## Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., and Yan, S.
## Better diffusion models further improve adversarial training, June 2023
## Code available at https://github.com/wzekai99/DM-Improves-AT

import torch

def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()/float(true.size(0))
    return accuracy.item()
