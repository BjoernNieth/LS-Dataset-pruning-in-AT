## Based on code from:
## Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., and Yan, S.
## Better diffusion models further improve adversarial training, June 2023
## Code available at https://github.com/wzekai99/DM-Improves-AT
import argparse

from core.models import MODELS
from core.utils.train import SCHEDULERS
from core.attacks import ATTACKS
from core.utils import str2bool, str2float
from core.data import DATASETS

def train_parser():
    """Returns an argument parser for train_main.py"""
    parser = argparse.ArgumentParser(description="Used for standard and adversarial training.")

    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="cifar10_generated1m", help="Which dataset used for training." )
    parser.add_argument("--seed", type=int, default=2701, help="Seed for random functions in script.")
    parser.add_argument("--model", choices=MODELS, default="wrn-28-10-swish")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs during training.")
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--batch-size-validation', type=int, default=64, help='Batch size for testing.')
    parser.add_argument("--early-stop-diff", type=int, default=0, help="Generations with not improvement till early stopp")

    # Pruning configurations
    parser.add_argument("--deduplication-threshold", type=float, default=None, help="Threshold for SSCD deduplication pruning.")
    parser.add_argument("--prune-percentage", type=float, default=None, help="Percentage pruned during dynamic uncertainty pruning.")
    parser.add_argument("--use-adversarial-predictions", type=str2bool, default=False, help="Use adversarial predictioncs to calculate Dynamic Uncertainty.")
    parser.add_argument("--keep-class-distribution", type=str2bool, default=False, help="Same class distribution before and after pruning.")
    parser.add_argument("--generated-fraction", type=float, default=0.8, help="Percentage of generated data per batch.")
    parser.add_argument("--prune-base", type=str2bool, default=False, help="Apply pruning to base dataset. Only if generated dataset selected.")
    parser.add_argument("--subsampling", type=float, default=0, help="Percentage for random subsampling.")
    parser.add_argument("--use-normal-predictions", type=str2bool, default=False, help="Use normal training logits for pruning.")
    parser.add_argument("--only-generated", type=str2bool, default=False, help="Only use generated data")
    parser.add_argument("--no-overfitted-epochs", type=str2bool, default=False, help="Calculate the dynamic uncertainty only up to the early stop point")
    parser.add_argument("--fundamental-frequency-pruning", type=str2bool, default=False, help="Use the fundamental frequency to prune the dataset")


    # Optimizer setting
    parser.add_argument("--lr", type=float, default=0.4, help="Learning rate for optimizer.")
    parser.add_argument('--beta', default=None, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.')
    parser.add_argument("--ls", type=float, default=0.1, help="Label smoothing.")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay during training.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Scheduler strategy..')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient norm clipping.')

    # Configure adverserial attacks
    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')
    parser.add_argument('--mart', type=str2bool, default=False, help='MART training.')
    parser.add_argument('--adversarial', type=str2bool, default=True, help='Train adversarial.')
    parser.add_argument('--clip-value', default=0, type=float)

    parser.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')
    parser.add_argument('--run-name', type=str, default="", help='Name of the run')

    # Consistency
    parser.add_argument('--consistency', action='store_true', default=False, help='use Consistency.')
    parser.add_argument('--cons_lambda', type=float, default=1.0, help='lambda for Consistency.')
    parser.add_argument('--cons_tem', type=float, default=0.5, help='temperature for Consistency.')
    parser.add_argument('--CutMix', action='store_true', default=False, help='use CutMix.')
    parser.add_argument('--LSE', type=str2bool, default=False, help='LSE training.')

    parser.add_argument('--resume-training', type=str2bool, default=False, help='Continue a previous training.')
    return parser
