import os
import sys
import warnings
import argparse
import shutil
# import submitit
from os.path import exists, realpath
from datetime import datetime
import numpy as np
from core.trainer import Trainer
from core.evaluate import Evaluator
from core.models.lipltArchitectures import getNetworkArchitecture


warnings.filterwarnings("ignore")


def main(config):
    folder = config.train_dir.split('/')[-1]
    if config.mode == 'train':
        trainer = Trainer(config)
        trainer()
    elif config.mode in ['certified', 'attack', 'certified_attack', "empiricalCurvature"]:
        evaluate = Evaluator(config)
        evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or Evaluate Lipschitz Networks.')

    # parameters training or eval
    parser.add_argument("--mode", type=str, default="train",
                        choices=['train', 'certified', 'attack', 'certified_attack', 'empiricalCurvature'])
    parser.add_argument("--train_dir", type=str, help="Name of the training directory.")
    parser.add_argument("--trainParentFolder", type=str, help="Name of the training parent directory.",
                        default='./trained_models')
    parser.add_argument("--data_dir", type=str, help="Name of the data directory.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use")

    parser.add_argument("--shift_data", type=bool, default=True, help="Shift dataset with mean.")
    parser.add_argument("--normalize_data", action='store_true', help="Normalize dataset.")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training.")
    parser.add_argument("--loss", type=str, default="xent", help="Define the loss to use for training.")
    parser.add_argument("--margin", type=float, default=0.7, help="Define margin")
    parser.add_argument("--offset", type=float, default=np.sqrt(2) * 3 / 2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--scheduler", type=str, default="interp")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0, help="Weight decay to use for training.")
    parser.add_argument("--nesterov", action='store_true', default=False)
    parser.add_argument("--warmup_scheduler", type=float, default=0., help="Percentage of training.")
    parser.add_argument("--decay", type=str, help="Milestones for MultiStepLR")
    parser.add_argument("--gamma", type=float, help="Gamma for MultiStepLR")
    parser.add_argument("--gradient_clip_by_norm", type=float, default=None)
    parser.add_argument("--gradient_clip_by_value", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, help="Make the training deterministic.", default=0)
    parser.add_argument("--print_grad_norm", action='store_true', help="Print of the norm of the gradients")
    parser.add_argument("--frequency_log_steps", type=int, default=1000, help="Print log for every step.")
    parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=25, help="Save checkpoint every epoch.")

    parser.add_argument("--activation", type=str, default='tanh', choices=['relu', 'tanh',
                                                                           'softplus', 'centered_softplus'])
    parser.add_argument("--penalizeCurvature", action='store_true')
    parser.add_argument("--boundCurvature", action='store_true')

    parser.add_argument("--hessianRegularizerCoefficient", type=float,
                        help="Coefficient for the hessian regularizer", default=1)
    parser.add_argument("--hessianRegularizerPrimalDualStepSize", type=float,
                        help="Step size for the primal dual of the hessian regularizer", default=0.001)
    parser.add_argument("--hessianRegularizerPrimalDualEpsilon", type=float,
                        help="Epsilon for the primal dual of the hessian regularizer", default=0.5)
    parser.add_argument("--hessianRegularizerMinimumCoefficient", type=float,
                        help="Minimum regularizer coefficient for the primal dual of the hessian regularizer",
                        default=0.0001)
    parser.add_argument("--accuracyEmaFactor", type=float, default=0.001)

    # specific parameters for eval
    parser.add_argument("--attack", type=str, choices=['pgd', 'autoattack'], help="Choose the attack.")
    parser.add_argument('--newtonStep', action='store_true')
    parser.add_argument("--eps", type=float, default=36)

    # parameters of the architectures
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--num_channels", type=int, default=30)
    parser.add_argument("--depth_linear", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=2048)
    parser.add_argument("--conv_size", type=int, default=5)
    parser.add_argument("--init", type=str, default='xavier_normal')
    parser.add_argument("--first_layer", type=str, default="padding_channels")
    parser.add_argument("--last_layer", type=str, default="pooling_linear")

    parser.add_argument("--crm", action="store_true")
    parser.add_argument("--learnableBeta", action="store_true",
                        help="Learnable beta for the softplus "
                             "activation. Currently only supported for liplt architectures")
    parser.add_argument("--loadCheckpoint", action="store_true")
    parser.add_argument("--checkpointPath", type=str, help="Full path to the checkpoint")

    parser.add_argument("--plot", action="store_true")


    # parse all arguments
    config = parser.parse_args()
    config.cmd = f"python3 {' '.join(sys.argv)}"



    def override_args(config, depth, num_channels, depth_linear, n_features):
        config.depth = depth
        config.num_channels = num_channels
        config.depth_linear = depth_linear
        config.n_features = n_features
        return config


    if config.model_name == 'small':
        config = override_args(config, 20, 45, 7, 2048)
    elif config.model_name == 'medium':
        config = override_args(config, 30, 60, 10, 2048)
    elif config.model_name == 'large':
        config = override_args(config, 50, 90, 10, 2048)
    elif config.model_name == 'xlarge':
        config = override_args(config, 70, 120, 15, 2048)
    elif config.model_name == '3C1F':
        config = override_args(config, 3, 15, 1, 1024)
    elif config.model_name == '4C3F':
        config = override_args(config, 4, 15, 3, 1024)
    elif config.model_name == '3C5F':
        config = override_args(config, 3, 15, 5, 100)
    elif config.model_name.startswith('liplt-'):  # liplt-(actualName)

        if config.dataset == 'cifar10':
            inputShape = (3, 32, 32)
            numberOfClasses = 10
        elif config.dataset == "mnist":
            inputShape = (1, 28, 28)
            numberOfClasses = 10
        elif config.dataset == "cifar100":
            inputShape = (3, 32, 32)
            numberOfClasses = 100
        else:
            raise ValueError("Dataset not supported")
        config.networkConfiguration = {"perClassLipschitz": False,
                                       "activation": config.activation,
                                       'learnableBeta': config.learnableBeta,
                                       "numberOfPowerIterations": 2,
                                       "inputShape": inputShape,
                                       "modelType": "liplt",
                                       'layers': getNetworkArchitecture(config.model_name[6:]),
                                       "weightInitialization": "standard",
                                       'pairwiseLipschitz': False,
                                       'architecture': None,
                                       'numberOfClasses': numberOfClasses, }
    elif config.model_name is None and \
            not all([config.depth, config.num_channels, config.depth_linear, config.n_features]):
        ValueError("Choose --model-name 'small' 'medium' 'large' 'xlarge'")

    # process argments
    eval_mode = ['certified', 'attack', 'certified_attack', "empiricalCurvature"]
    if config.data_dir is None:
        config.data_dir = os.environ.get('DATADIR', None)
    if config.data_dir is None:
        ValueError("the following arguments are required: --data_dir")
    os.makedirs(config.trainParentFolder, exist_ok=True)
    path = realpath(config.trainParentFolder)
    if config.train_dir is None:
        ValueError("--train_dir must be defined.")
    elif config.mode == 'train' and config.train_dir is not None:
        config.train_dir = f'{path}/{config.train_dir}'
        os.makedirs(config.train_dir, exist_ok=True)
        os.makedirs(f'{config.train_dir}/checkpoints', exist_ok=True)
    elif config.mode in eval_mode and config.train_dir is not None:
        config.train_dir = f'{path}/{config.train_dir}'
    elif config.mode in eval_mode and config.train_dir is None:
        ValueError("--train_dir must be defined.")

    if config.mode == 'attack' and config.attack is None:
        ValueError('With mode=attack, the following arguments are required: --attack')

    assert config.warmup_scheduler == 0  # due to library import errors, this is not supported on my device
    main(config)
