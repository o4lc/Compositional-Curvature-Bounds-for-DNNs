import json
import wandb
import time
import os
import re
import glob
import socket
import pprint
import logging
import shutil
from os.path import join, exists, basename
import torch
import wandb
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import time

from core import utils
from core.models.model import NormalizedModel, LipschitzNetwork
from core.data.readers import readers_config

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd.functional import jacobian 
from tqdm import tqdm

import pdb


class Evaluator:
    """Evaluate a Pytorch Model."""

    def __init__(self, config, wandb=False):
        self.config = config
        self.wandb = wandb


    def load_ckpt(self, ckpt_path=None):
        if ckpt_path is None:
            checkpoints = glob.glob(join(self.config.train_dir, "checkpoints", "model.ckpt-*.pth"))
            get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
            ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
            ckpt_path = join(self.config.train_dir, "checkpoints", ckpt_name)
        checkpoint = torch.load(ckpt_path)
        new_checkpoint = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'alpha' not in k:
                new_checkpoint[k] = v
        self.model.load_state_dict(new_checkpoint)
        epoch = checkpoint['epoch']
        return epoch


    def __call__(self):
        """Run evaluation of model or eval under attack"""

        cudnn.benchmark = True

        # create a mesage builder for logging
        self.message = utils.MessageBuilder()
        # Setup logging & log the version.
        utils.setup_logging(self.config, 0)

        ngpus = torch.cuda.device_count()
        if ngpus:
            self.batch_size = self.config.batch_size * ngpus
        else:
            self.batch_size = self.config.batch_size

        # load reader
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(self.config, self.batch_size, False, is_training=False)
        self.config.means = self.reader.means

        # load model
        self.model = LipschitzNetwork(self.config, self.reader.n_classes, activation=self.config.activation)
        self.model = NormalizedModel(self.model, self.reader.means, self.reader.stds)
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        self.load_ckpt()
        if self.config.mode == "certified":
            # for eps in [36, 72, 108, 255]:
            for eps in [36]:
                if self.config.last_layer == 'lln':
                    self.eval_certified_lln(eps)
                else:
                    self.eval_certified(eps)
                    # self.eval_certified_plot(eps)
        elif self.config.mode == "attack":
            self.eval_attack()

        logging.info("Done with batched inference.")
        delattr(self, 'model')


    @torch.no_grad()
    def eval_certified(self, eps):
        print("evaluating certified accuracy")
        eps_float = eps / 255
        self.model.eval()
        running_accuracy = 0
        running_certified = 0
        running_certified_lip, running_certified_lip_imp, running_certified_curv = 0, 0, 0
        running_inputs = 0
        lip_cst = np.sqrt(2.)
        data_loader, _ = self.reader.load_dataset()
        for batch_n, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            predicted = outputs.argmax(axis=1)
            correct = outputs.max(1)[1] == labels
            margins, index = torch.sort(outputs, 1)
            # Lip sqrt 2
            certified_lip = (margins[:, -1] - margins[:, -2]) >  lip_cst * eps_float
            running_certified_lip += torch.sum(correct & certified_lip).item()
            # Curv
            cert_tmp = torch.zeros_like(certified_lip)
            if isinstance(self.model, nn.DataParallel):
                activ = self.model.module.model.stable_block[0].activation
            else:
                activ = self.model.model.stable_block[0].activation
            if isinstance(activ, nn.Tanh):
                cert_tmp, __, lip_improved = self.secondOrderCert(inputs, margins, index, eps_float)
                running_certified_curv += torch.sum(correct & cert_tmp).item()
                lip_improved = torch.minimum(lip_improved, torch.ones_like(lip_improved) * lip_cst)
                certified_lip_imp = (margins[:, -1] - margins[:, -2]) >  lip_improved * eps_float
                running_certified_lip_imp += torch.sum(correct & certified_lip_imp).item()
                
            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            certified = torch.logical_or(cert_tmp, torch.logical_or(certified_lip, certified_lip_imp))
            running_certified += torch.sum(correct & certified).item()
            running_inputs += inputs.size(0)
        self.model.train()
        accuracy = running_accuracy / running_inputs
        certified = running_certified / running_inputs
        self.message.add('eps', [eps, 255], format='.0f')
        self.message.add('eps', eps_float, format='.5f')
        self.message.add('accuracy', accuracy, format='.5f')
        self.message.add('certified acc', certified, format='.5f')
        self.message.add('certified acc lip', running_certified_lip / running_inputs, format='.5f')
        self.message.add('certified acc lip imp', running_certified_lip_imp / running_inputs, format='.5f')
        self.message.add('certified acc curv', running_certified_curv / running_inputs, format='.5f')

        if self.wandb:
            wandb.log({"accuracy": accuracy, "certified accuracy " + str(eps): certified})
        message = self.message.get_message()
        logging.info(message)
        print(message)

        return accuracy, certified
    
    def secondOrderCert(self, inputs, margins, index, eps_float, useQueryCoefficients=False):
        queryCoefficient = torch.zeros(inputs.shape[0], self.model.module.model.n_classes).cuda()
        queryCoefficient[torch.arange(inputs.shape[0]), index[:, -1]] += 1
        queryCoefficient[torch.arange(inputs.shape[0]), index[:, -2]] += -1
        def grad_f(x):
            return torch.einsum('ij,ij->i', queryCoefficient, self.model(x))

        grad = jacobian(grad_f, inputs).sum(dim=1).reshape(inputs.shape[0], -1)
        grad_norm = torch.linalg.norm(grad, 2, dim=1)
        if useQueryCoefficients:
            M = self.model.module.model.calculateCurvature(queryCoefficient).unsqueeze(0)
        else:
            M = self.model.module.model.calculateCurvature().unsqueeze(0)
            M *= np.sqrt(2)
        if self.wandb:
            wandb.log({"Curvature Bound": M})
        diff = margins[:, -2] - margins[:, -1]
        cert_rad = (-grad_norm + torch.sqrt(grad_norm**2 - 2 * M * diff)) / M 
        cert_tmp = cert_rad > eps_float
        # 
        lip_improved = grad_norm + eps_float * M
        return cert_tmp, cert_rad, lip_improved


    @torch.no_grad()
    def eval_certified_lln(self, eps):
        eps_float = eps / 255
        self.model.eval()
        running_accuracy = 0
        running_certified = 0
        running_inputs = 0
        lip_cst = 1.
        data_loader, _ = self.reader.load_dataset()
        last_weight = self.model.module.model.last_last.weight
        normalized_weight = F.normalize(last_weight, p=2, dim=1)
        for batch_n, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            predicted = outputs.argmax(axis=1)
            correct = outputs.max(1)[1] == labels
            margins, indices = torch.sort(outputs, 1)
            margins = margins[:, -1][:, None] - margins[:, 0:-1]
            for idx in range(margins.shape[0]):
                margins[idx] /= torch.norm(
                    normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]], dim=1, p=2)
            margins, _ = torch.sort(margins, 1)
            certified = margins[:, 0] > eps_float * lip_cst
            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            running_certified += torch.sum(correct & certified).item()
            running_inputs += inputs.size(0)
        accuracy = running_accuracy / running_inputs
        certified = running_certified / running_inputs
        self.message.add('eps', [eps, 255], format='.0f')
        self.message.add('eps', eps_float, format='.5f')
        self.message.add('accuracy', accuracy, format='.5f')
        self.message.add('certified accuracy', certified, format='.5f')
        logging.info(self.message.get_message())
        return accuracy, certified


    def eval_attack(self):
        """Run evaluation under attack."""
        calculatePgdAccuracy(self.model, self.reader.load_dataset()[0], self.reader.means, self.config.eps / 255)
        # attack = utils.get_attack_eval(
        #                 self.model,
        #                 self.config.attack,
        #                 self.config.eps/255,
        #                 self.batch_size)
        #
        # running_accuracy = 0
        # running_accuracy_adv = 0
        # running_inputs = 0
        # data_loader, _ = self.reader.load_dataset()
        # for batch_n, data in enumerate(data_loader):
        #
        #   inputs, labels = data
        #   inputs, labels = inputs.cuda(), labels.cuda()
        #   inputs_adv = attack.perturb(inputs, labels)
        #
        #   outputs = self.model(inputs)
        #   outputs_adv = self.model(inputs_adv)
        #   _, predicted = torch.max(outputs.data, 1)
        #   _, predicted_adv = torch.max(outputs_adv.data, 1)
        #
        #   running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
        #   running_accuracy_adv += predicted_adv.eq(labels.data).cpu().sum().numpy()
        #   running_inputs += inputs.size(0)
        #
        # accuracy = running_accuracy / running_inputs
        # accuracy_adv = running_accuracy_adv / running_inputs
        # self.message.add(f'attack: {self.config.attack} - eps', self.config.eps, format='.0f')
        # self.message.add('Accuracy', accuracy, format='.5f')
        # self.message.add('Accuracy attack', accuracy_adv, format='.5f')
        # logging.info(self.message.get_message())

    
    @torch.no_grad()
    def eval_certified_plot(self, eps):
        print("evaluating certified accuracy")
        eps_float = eps / 255
        self.model.eval()
        radiusDistCurv, radiusDistLip = [], []

        lip_cst = 1.
        data_loader, _ = self.reader.load_dataset()
        for batch_n, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            correct = outputs.max(1)[1] == labels
            margins, index = torch.sort(outputs, 1)
            radiusDistLip.append((((margins[:, -1] - margins[:, -2]) * correct).cpu().numpy() / (np.sqrt(2.) * lip_cst)) )
            if isinstance(self.model, nn.DataParallel):
                activ = self.model.module.model.stable_block[0].activation
            else:
                activ = self.model.model.stable_block[0].activation
            if isinstance(activ, nn.Tanh):
                __, tmp_radius = self.secondOrderCert(inputs, margins, index, eps_float)
                radiusDistCurv.append((tmp_radius * correct).cpu().numpy() )
            
            
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        ranges = np.arange(0, 2.5, 0.1)
        counts, bins = plt.hist(np.concatenate(radiusDistLip), bins=ranges, alpha=0.5, label="Lip")[:2]
        counts2, bins2 = plt.hist(np.concatenate(radiusDistCurv), bins=ranges, alpha=0.5, label="Curv")[:2]
        plt.legend()

        counts /= len(data_loader.dataset)
        counts2 /= len(data_loader.dataset)

        plt.subplot(1, 2, 2)
        plt.plot(bins[:-1], np.cumsum(counts[::-1])[::-1] , label="Lip")
        plt.plot(bins2[:-1], np.cumsum(counts2[::-1])[::-1], label="Curv")
        print(np.cumsum(counts[::-1])[::-1] - np.cumsum(counts2[::-1])[::-1])
        plt.legend()
        plt.show()


def calculatePgdAccuracy(net, dataset, datasetShift, eps, lBall=2, device=torch.device("cuda"), verbose=True,
                         dataloaders=None):
    # converting the eps that is for the range [0, 1] to the range of the normalized dataset:
    lowerBound = float("inf")
    upperBound = float("-inf")
    for x in datasetShift:
        lowerBound = min(lowerBound, -x)
        upperBound = max(upperBound, (1 - x))

    accuracies = (1 - evaluate_madry(dataset, net, eps, False, device, lBall=lBall,
                                     lowerBound=lowerBound, upperBound=upperBound).item()) * 100
    if verbose:
        print("PGD accuracy percentage on test dataset: {}".format(accuracies))
    return accuracies


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _pgd(model, X, Y, epsilon, niters=20, alpha=0.001, lBall=torch.inf, lowerBound=0, upperBound=1):
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, Y)
    err = (out.data.max(1)[1] != Y.data).float().sum() / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters):
        # print(i, model(X_pgd))
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        out = model(X_pgd)
        out = out - out.mean(1, keepdim=True)
        # loss = nn.CrossEntropyLoss()(out, Y)
        loss = -(1 + nn.functional.softmax(out, dim=1)[range(Y.shape[0]), Y]).log().mean()
        # print(loss)
        loss.backward()
        eta = alpha * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

        if lBall == torch.inf:
            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
        elif lBall == 2:
            y = X_pgd.data - X.data
            if len(y.shape) == 2:
                norm = torch.linalg.norm(y, 2, 1, keepdim=True)
            elif len(y.shape) == 3:
                norm = torch.linalg.norm(y, "fro", (1, 2), keepdim=True)
            elif len(y.shape) == 4:
                norm = torch.linalg.norm(y.reshape(y.shape[0], -1), 2, 1).reshape(y.shape[0], 1, 1, 1)
            else:
                raise NotImplementedError
            X_pgd = Variable(torch.clamp(X.data + epsilon * y / norm, lowerBound, upperBound), requires_grad=True)
            # print(X_pgd)
        elif lBall == 1:
            raise NotImplementedError
        else:
            raise ValueError

    err_pgd = (model(X_pgd).data.max(1)[1] != Y.data).float().sum() / X.size(0)
    return err, err_pgd


def evaluate_madry(loader, model, epsilon, verbose, device=torch.device("cuda"), lBall=torch.inf,
                   lowerBound=0, upperBound=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    perrors = AverageMeter()

    model.eval()

    end = time.time()
    for i, (X, y) in tqdm(enumerate(loader), total=len(loader)):
        X, y = X.to(device), y.to(device)
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum() / X.size(0)

        # # perturb
        _, pgd_err = _pgd(model, Variable(X), Variable(y), epsilon, lBall=lBall,
                          lowerBound=lowerBound, upperBound=upperBound)

        # print to logfile
        # print(epoch, i, ce.item(), err, file=log)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        perrors.update(pgd_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PGD Error {perror.val:.3f} ({perror.avg:.3f})\t'
                  'Error {error.val:.3f} ({error.avg:.3f})'.format(
                i, len(loader), batch_time=batch_time, loss=losses,
                error=errors, perror=perrors))
        # log.flush()
    if verbose:
        print(' * PGD error {perror.avg:.3f}\t'
              'Error {error.avg:.3f}'
              .format(error=errors, perror=perrors))
    return perrors.avg