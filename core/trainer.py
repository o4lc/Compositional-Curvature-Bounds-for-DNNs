import os
import sys
import time
import random
import datetime
import pprint
import socket
import logging
import glob
from os.path import join, exists
# from contextlib import nullcontext
from tqdm import tqdm

from core import utils
from core.models.model import NormalizedModel, SllNetwork, lipschitzModel
from core.data.readers import readers_config
from core.evaluate import Evaluator

import numpy as np
import geoopt
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import wandb


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class Trainer:
    """A Trainer to train a PyTorch."""

    def __init__(self, config):
        wandb.init(project="secondOrderRobustness", entity="limitlessInfinite", config=config)
        self.config = config
        self.config.mode = "certified"
        init_seeds(config.seed)
        self.evaluate = Evaluator(self.config, wandb=True)

        self.regularizerCoefficient = config.hessianRegularizerCoefficient
        self.regularizerStepSize = config.hessianRegularizerPrimalDualStepSize
        self.pdEpsilon = config.hessianRegularizerPrimalDualEpsilon
        self.minimumRegularizerCoefficient = config.hessianRegularizerMinimumCoefficient
        self.accuracyMovingAverage = 0
        self.movingAverageFactor = config.accuracyEmaFactor


    def _load_state(self):
        # load last checkpoint
        checkpoints = glob.glob(join(self.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
        get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
        checkpoints = sorted(
            [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
        path_last_ckpt = join(self.train_dir, 'checkpoints', checkpoints[-1])
        self.checkpoint = torch.load(path_last_ckpt, map_location=self.model.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.saved_ckpts.add(self.checkpoint['epoch'])
        epoch = self.checkpoint['epoch']
        if self.local_rank == 0:
            logging.info('Loading checkpoint {}'.format(checkpoints[-1]))

    def _save_ckpt(self, step, epoch, final=False, best=False):
        """Save ckpt in train directory."""
        freq_ckpt_epochs = self.config.save_checkpoint_epochs
        if (epoch % freq_ckpt_epochs == 0 and self.is_master \
            and epoch not in self.saved_ckpts) \
                or (final and self.is_master) or best:
            prefix = "model" if not best else "best_model"
            checkpointFolder = join(self.train_dir, 'checkpoints')
            # delete everything in checkpointFolder
            for f in os.listdir(checkpointFolder):
                os.remove(os.path.join(checkpointFolder, f))
            ckpt_name = f"{prefix}.ckpt-{step}.pth"
            ckpt_path = join(self.train_dir, 'checkpoints', ckpt_name)
            if exists(ckpt_path) and not best: return
            self.saved_ckpts.add(epoch)
            state = {
                'epoch': epoch,
                'global_step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scheduler': self.scheduler.state_dict()
            }
            logging.debug("Saving checkpoint '{}'.".format(ckpt_name))
            torch.save(state, ckpt_path)
            self.evaluate()

    @record
    def __call__(self):
        """Performs training and evaluation
        """
        cudnn.benchmark = True

        self.train_dir = self.config.train_dir
        self.ngpus = 1

        # job_env = submitit.JobEnvironment()
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.num_nodes = int(os.environ['LOCAL_WORLD_SIZE'])
        self.num_tasks = int(os.environ['WORLD_SIZE'])
        self.is_master = bool(self.rank == 0)

        # Setup logging
        utils.setup_logging(self.config, self.rank)

        logging.info(self.rank)
        logging.info(self.local_rank)
        logging.info(self.num_nodes)
        logging.info(self.num_tasks)

        self.message = utils.MessageBuilder()
        # print self.config parameters
        if self.local_rank == 0:
            logging.info(self.config.cmd)
            pp = pprint.PrettyPrinter(indent=2, compact=True)
            logging.info(pp.pformat(vars(self.config)))
        # print infos
        if self.local_rank == 0:
            logging.info(f"PyTorch version: {torch.__version__}.")
            logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
            logging.info(f"Hostname: {socket.gethostname()}.")

        # ditributed settings
        self.world_size = 1
        self.is_distributed = False
        if self.num_nodes > 1 or self.num_tasks > 1:
            self.is_distributed = True
            self.world_size = self.num_nodes * self.ngpus
        if self.num_nodes > 1:
            logging.info(
                f"Distributed Training on {self.num_nodes} nodes")
        elif self.num_nodes == 1 and self.num_tasks > 1:
            logging.info(f"Single node Distributed Training with {self.num_tasks} tasks")
        else:
            assert self.num_nodes == 1 and self.num_tasks == 1
            logging.info("Single node training.")

        if not self.is_distributed:
            self.batch_size = self.config.batch_size * self.ngpus
        else:
            self.batch_size = self.config.batch_size

        self.global_batch_size = self.batch_size * self.world_size
        logging.info('World Size={} => Total batch size {}'.format(
            self.world_size, self.global_batch_size))

        torch.cuda.set_device(self.local_rank)

        # load dataset
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(self.config, self.batch_size, self.is_distributed, is_training=True)
        if self.local_rank == 0:
            logging.info(f"Using dataset: {self.config.dataset}")

        # load model
        self.model = lipschitzModel(self.config, self.reader.n_classes, self.config.activation)
        self.model = NormalizedModel(self.model, self.reader.means, self.reader.stds)
        self.model = self.model.cuda()
        nb_parameters = np.sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logging.info(f'Number of parameters to train: {nb_parameters}')

        if self.config.loadCheckpoint:
            checkpoint = torch.load(self.config.checkpointPath, map_location=self.model.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])


        # setup distributed process if training is distributed
        # and use DistributedDataParallel for distributed training
        if self.is_distributed:
            utils.setup_distributed_training(self.world_size, self.rank)
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            if self.local_rank == 0:
                logging.info('Model defined with DistributedDataParallel')
        else:
            self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        # define set for saved ckpt
        self.saved_ckpts = set([0])

        data_loader, sampler = self.reader.load_dataset()
        if sampler is not None:
            assert sampler.num_replicas == self.world_size

        if self.is_distributed:
            n_files = sampler.num_samples
        else:
            n_files = self.reader.n_train_files

        # define optimizer
        self.optimizer = utils.get_optimizer(self.config, self.model.parameters())

        # define learning rate scheduler
        num_steps = self.config.epochs * (self.reader.n_train_files // self.global_batch_size)
        self.scheduler, self.warmup = utils.get_scheduler(self.optimizer, self.config, num_steps)
        if self.config.warmup_scheduler is not None:
            logging.info(f"Warmup scheduler on {self.config.warmup_scheduler * 100:.0f}% of training")

        # define the loss
        self.criterion = utils.get_loss(self.config)

        if self.local_rank == 0:
            logging.info("Number of files on worker: {}".format(n_files))
            logging.info("Start training")

        print('Num of Params', self.model.module.model.calcParams())
        # training loop
        start_epoch, global_step = 0, 0
        self.best_checkpoint = None
        self.best_accuracy = None
        self.best_accuracy = [0., 0.]
        for epoch_id in tqdm(range(start_epoch, self.config.epochs)):

            if self.is_distributed:
                sampler.set_epoch(epoch_id)
            for n_batch, data in enumerate(data_loader):
                if global_step == 2 and self.is_master:
                    start_time = time.time()
                epoch = (int(global_step) * self.global_batch_size) / self.reader.n_train_files
                self.one_step_training(data, epoch, global_step)
                self._save_ckpt(global_step, epoch_id)
                if global_step == 20 and self.is_master:
                    self._print_approximated_train_time(start_time)
                global_step += 1


            with torch.no_grad():  # this step is for the convergence of the power iteration method.
                self.model.module.model.converge()

        self._save_ckpt(global_step, epoch_id, final=True)
        logging.info("Done training -- epoch limit reached.")

    def filter_parameters(self):
        conv_params, linear_params = [], []
        for name, params in self.model.named_parameters():
            if 'weight' in name.lower() and params.dim() == 4:
                conv_params.append(params)
            elif 'weight' in name.lower() and params.dim() == 2:
                linear_params.append(params)
            elif 'bias' in name.lower():
                conv_params.append(params)
        return conv_params, linear_params

    def compute_gradient_norm(self):
        grad_norm = 0.
        for name, p in self.model.named_parameters():
            if p.grad is None: continue
            norm = p.grad.detach().data.norm(2)
            grad_norm += norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        return grad_norm

    def _print_approximated_train_time(self, start_time):
        total_steps = self.reader.n_train_files * self.config.epochs / self.global_batch_size
        total_seconds = total_steps * ((time.time() - start_time) / 18)
        n_days = total_seconds // 86400
        n_hours = (total_seconds % 86400) / 3600
        logging.info(
            'Approximated training time: {:.0f} days and {:.1f} hours'.format(
                n_days, n_hours))

    def _to_print(self, step):
        frequency = self.config.frequency_log_steps
        if frequency is None:
            return False
        return (step % frequency == 0 and self.local_rank == 0) or \
            (step == 1 and self.local_rank == 0)

    def process_gradients(self, step):
        if self.config.gradient_clip_by_norm:
            if step == 0 and self.local_rank == 0:
                logging.info("Clipping Gradient by norm: {}".format(
                    self.config.gradient_clip_by_norm))
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_by_norm)
        elif self.config.gradient_clip_by_value:
            if step == 0 and self.local_rank == 0:
                logging.info("Clipping Gradient by value: {}".format(
                    self.config.gradient_clip_by_value))
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.config.gradient_clip_by_value)

    def one_step_training(self, data, epoch, step):
        numberOfClasses = self.model.module.model.n_classes

        self.optimizer.zero_grad()

        batch_start_time = time.time()
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        thisBatchSize = images.shape[0]

        if step == 0 and self.local_rank == 0:
            logging.info(f'images {images.shape}')
            logging.info(f'labels {labels.shape}')

        outputs = self.model(images)
        if step == 0 and self.local_rank == 0:
            logging.info(f'outputs {outputs.shape}')

        modelCurvature = self.model.module.model.calculateCurvature()
        if len(modelCurvature.shape) == 1:
            modelCurvature = modelCurvature[0]
        maxIndices = torch.argmax(outputs, 1)
        accuracy = (torch.sum(maxIndices == labels) / labels.shape[0]).item()

        if isinstance(self.model.module.model, SllNetwork):
            modelLipschitz = 1.
        else:
            modelLipschitz = self.model.module.model.calculateNetworkLipschitz()
            wandb.log({"Lipschitz": modelLipschitz.item()})

        if self.config.crm:
            lipschitzToUse = modelLipschitz
            minimumLipschitz = 1
            if modelLipschitz < minimumLipschitz:
                lipschitzToUse = modelLipschitz.detach().clone()
            self.criterion.offset = self.config.offset * lipschitzToUse
        # if self.config.dynamicOffset:
        #     self.accuracyMovingAverage = (self.movingAverageFactor * accuracy +
        #                                   (1 - self.movingAverageFactor) * self.accuracyMovingAverage)
        #
        #     self.regularizerCoefficient = (
        #         max(self.regularizerCoefficient +
        #             (self.accuracyMovingAverage - self.pdEpsilon) * self.regularizerStepSize,
        #             self.minimumRegularizerCoefficient))
        #     if modelLipschitz < self.config.minimumLipschitz:
        #         valueForOffset = modelLipschitz.detach().clone()
        #     else:
        #         valueForOffset = modelLipschitz
        #     self.criterion.offset = valueForOffset * self.config.offset * self.regularizerCoefficient
        #     # self.criterion.offset =\
        #     #     (torch.maximum(modelLipschitz, torch.tensor([self.config.minimumLipschitz]).to(modelLipschitz))
        #     #      * self.config.offset * self.regularizerCoefficient)
        #     wandb.log({"Regularizer Coefficient": self.regularizerCoefficient,
        #                "moving average accuracy": self.accuracyMovingAverage,
        #                "criterion offset": self.criterion.offset.item()})

        loss = self.criterion(outputs, labels)
        wandb.log({"xent loss": loss.item(),
                   "lr": self.optimizer.param_groups[0]['lr']})


        wandb.log({"Curvature Bound": modelCurvature.item(), "accuracy": accuracy,})
        if self.config.penalizeCurvature:
            self.accuracyMovingAverage = (self.movingAverageFactor * accuracy +
                                          (1 - self.movingAverageFactor) * self.accuracyMovingAverage)
            self.regularizerCoefficient = (
                max(self.regularizerCoefficient +
                    (self.accuracyMovingAverage - self.pdEpsilon) * self.regularizerStepSize,
                    self.minimumRegularizerCoefficient))
            loss += self.regularizerCoefficient * modelCurvature

            wandb.log({"Regularizer Coefficient": self.regularizerCoefficient,
                       "moving average accuracy": self.accuracyMovingAverage,
                       "total loss": loss.item(),})
        elif self.config.boundCurvature:
            self.regularizerCoefficient = (
                max(self.regularizerCoefficient +
                    (modelCurvature.item() - self.pdEpsilon) * self.regularizerStepSize,
                    self.minimumRegularizerCoefficient))
            loss += self.regularizerCoefficient * modelCurvature
            wandb.log({"Regularizer Coefficient": self.regularizerCoefficient,
                       "total loss": loss.item(), })
        elif self.config.penalizeHessian:
            hessian = torch.autograd.functional.hessian(lambda x: self.model(x).mean(), images)
            hessian = hessian.view(thisBatchSize, -1)
            hessian = torch.norm(hessian, p=2, dim=1)
            loss += self.regularizerCoefficient * hessian.mean()
            wandb.log({"Hessian Norm": hessian.mean().item(),
                       "total loss": loss.item(), })
        # elif self.config.crm:
            # margins = outputs[range(thisBatchSize), maxIndices].unsqueeze(1) - outputs
            # if isinstance(self.model, nn.DataParallel):
            #     lipschitzConstants = self.model.module.model.calculateNetworkLipschitz(selfPairDefaultValue=1)
            #     pairWiseLipschitzConstants =\
            #         self.model.module.model.createPairwiseLipschitzFromLipschitz(lipschitzConstants, numberOfClasses, 1)
            # else:
            #     raise NotImplementedError
            #
            # certifiedRadii = margins / pairWiseLipschitzConstants[maxIndices, :]
            # certifiedRadii[range(thisBatchSize), maxIndices] = torch.inf
        loss.backward()
        self.process_gradients(step)
        self.optimizer.step()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.model.miniBatchStep()
        else:
            self.model.model.miniBatchStep()
        # with self.warmup.dampening() if self.warmup else nullcontext():
        self.scheduler.step(step)

        seconds_per_batch = time.time() - batch_start_time
        examples_per_second = self.batch_size / seconds_per_batch
        examples_per_second *= self.world_size

        if self._to_print(step):
            lr = self.optimizer.param_groups[0]['lr']
            self.message.add("epoch", epoch, format="4.2f")
            self.message.add("step", step, width=5, format=".0f")
            self.message.add("lr", lr, format=".6f")
            self.message.add("loss", loss, format=".4f")
            if self.config.print_grad_norm:
                grad_norm = self.compute_gradient_norm()
                self.message.add("grad", grad_norm, format=".4f")
            self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
            message = self.message.get_message()
            logging.info(message)
            print(message)
