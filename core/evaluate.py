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

import matplotlib.pyplot as plt

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
        print('Num of Params', self.model.module.model.calcParams())
        secondOrderCertificates = True
        if self.config.mode == "certified":
            # self.eval_certified_plot(eps=36/255)
            accuracy, cert_rad, lip_cert_rad, curv_cert_rad, grad_norm, margins, corrects, M = self.evaluate_certified_radius()
            if len(M) == 0:
                secondOrderCertificates = False
            epss = [36, 72, 108, 255]
            if self.config.dataset == 'mnist':
                epss = [403]

            if self.config.newtonStep:
                M = self.model.module.model.calculateCurvature().unsqueeze(0).unsqueeze(0)
                data_loader, _ = self.reader.load_dataset()
                M = M.repeat(data_loader.dataset.__len__(), 1)
                curv_cert_rad_tmp = self.newton_cert_rad(M).unsqueeze(1)
                curv_cert_rad = torch.maximum(curv_cert_rad, curv_cert_rad_tmp)
                cert_rad = torch.maximum(cert_rad, curv_cert_rad)
                

            for eps in epss:
                eps_float = eps / 255
                if secondOrderCertificates:
                    lip_cst_imp = torch.minimum(grad_norm + M * eps_float, torch.ones_like(grad_norm) * np.sqrt(2.))
                    lip_cert_rad_imp = self.evaluate_certified_radius_lip(lip_cst_imp, margins) * corrects
                    cert_rad_eps = torch.maximum(cert_rad, lip_cert_rad_imp)
                    curv_cert_acc = (curv_cert_rad > eps_float).sum() / curv_cert_rad.shape[0]
                    lip_cert_acc_imp = (lip_cert_rad_imp > eps_float).sum() / lip_cert_rad_imp.shape[0]
                else:
                    cert_rad_eps = cert_rad
                    curv_cert_acc = -1
                    lip_cert_acc_imp = -1



                cert_acc = (cert_rad_eps > eps_float).sum() / cert_rad.shape[0]
                lip_cert_acc = (lip_cert_rad > eps_float).sum() / lip_cert_rad.shape[0]


                self.message.add('eps', [eps, 255], format='.0f')
                self.message.add('eps', eps_float, format='.5f')
                self.message.add('accuracy', accuracy, format='.5f')
                self.message.add('certified acc', cert_acc, format='.5f')
                self.message.add('certified acc lip', lip_cert_acc, format='.5f')
                self.message.add('certified acc curv', curv_cert_acc, format='.5f')
                self.message.add('certified acc lip imp', lip_cert_acc_imp, format='.5f')
                print(self.message.get_message())
                logging.info(self.message.get_message())

                if secondOrderCertificates:
                    assert lip_cert_acc_imp >= lip_cert_acc
                    assert lip_cert_acc_imp + curv_cert_acc >= cert_acc

            # for eps in [36, 72, 108, 255]:
            #     if self.config.last_layer == 'lln':
            #         self.eval_certified_lln(eps)
            #     else:
            #         self.eval_certified(eps)
            #         # self.eval_certified_plot(eps)
                
        elif self.config.mode == "attack":
            self.eval_attack()

        elif self.config.mode == "certified_attack":
            self.eval_certified_attack()

        elif self.config.mode == "empiricalCurvature":
            measure_curvature(self.model,
                              self.reader.load_dataset()[0],
                              self.reader.n_classes,
                              data_fraction=1,
                              batch_size=self.config.batch_size,
                              num_power_iter=20,
                              device=torch.device("cuda"))

        logging.info("Done with batched inference.")
        delattr(self, 'model')

    @torch.no_grad()
    def evaluate_certified_radius_lip(self, lip_cst, margins):
        lip_cert_rad = (margins[:, -1:] - margins[:, -2:-1]) / lip_cst
        return lip_cert_rad

    @torch.no_grad()
    def eval_certified_attack(self):
        print("evaluating certified radius")
        self.model.eval()
        running_accuracy = 0
        running_inputs = 0
        uncertRad, uncertAcc = [], []
        Ms, grad_norms, marginss = [], [], []
        corrects = []

        data_loader, _ = self.reader.load_dataset()

        def grad_f(x):
            return torch.einsum('ij,ij->i', queryCoefficient, self.model(x))

        for batch_n, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = self.model(inputs)
            predicted = outputs.argmax(axis=1)
            correct = (outputs.max(1)[1] == labels).unsqueeze(1)
            corrects.append(correct)
            margins, index = torch.sort(outputs, 1)
            M = self.model.module.model.calculateCurvature().unsqueeze(0)

            uncertRad_tmp, uncertAcc_tmp = [], []
            for attackClass in range(10): 
                queryCoefficient = torch.zeros(inputs.shape[0], 10).cuda()
                queryCoefficient[torch.arange(inputs.shape[0]), labels] += 1
                queryCoefficient[torch.arange(inputs.shape[0]), attackClass] += -1

                grad = jacobian(grad_f, inputs).sum(dim=1).reshape(inputs.shape[0], -1)
                grad_norm = torch.linalg.norm(grad, 2, dim=1, keepdim=True)
                isUncert = grad_norm**2 > 2 * M * (margins[:, -1] - outputs[:, attackClass]).unsqueeze(1)

                uncertAcc_tmp.append(isUncert)
                uncertRad_tmp.append((grad_norm - \
                                    torch.sqrt(grad_norm**2 - 2 * M * (margins[:, -1] - outputs[:, attackClass]).unsqueeze(1)))/M)
        

            uncertAcc_tmp = torch.hstack(uncertAcc_tmp)
            # print(torch.any(uncertAcc_tmp, 1))
            uncertAcc_tmp = uncertAcc_tmp.sum(1).unsqueeze(1)
            uncertAcc_tmp = torch.minimum(uncertAcc_tmp, torch.ones_like(uncertAcc_tmp))

            uncertRad_tmp = torch.hstack(uncertRad_tmp)
            uncertRad_tmp = uncertRad_tmp[(torch.any(uncertAcc_tmp, 1).unsqueeze(1) * correct)[:, 0], :]
            uncertRad_tmp[torch.isnan(uncertRad_tmp)] = torch.inf
            uncertRad_tmp[uncertRad_tmp == 0] = torch.inf
            uncertRad_tmp = torch.min(uncertRad_tmp, 1).values.unsqueeze(1)

            uncertRad.append(uncertRad_tmp)
            uncertAcc.append(uncertAcc_tmp)
            

            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            running_inputs += inputs.size(0)
            # break
            

        uncertRad = torch.vstack(uncertRad)
        uncertAcc = torch.vstack(uncertAcc)
        plt.hist(uncertRad.detach().cpu().numpy(), bins=20)
        plt.xlabel('Attack Radius')
        plt.savefig('attack_radius.pdf')
        plt.show()

        accuracy = running_accuracy / running_inputs
        print(accuracy, uncertRad.shape[0]/running_inputs)
        return accuracy, uncertRad

    @torch.no_grad()
    def evaluate_certified_radius(self):
        print("evaluating certified accuracy")
        self.model.eval()
        running_accuracy = 0
        running_inputs = 0
        lip_cert_rad, curv_cert_rad = [], []
        Ms, grad_norms, marginss = [], [], []
        corrects = []
        lip_cst = np.sqrt(2.)
        data_loader, _ = self.reader.load_dataset()
        for batch_n, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()


            outputs = self.model(inputs)
            predicted = outputs.argmax(axis=1)
            correct = (outputs.max(1)[1] == labels).unsqueeze(1)
            corrects.append(correct)
            margins, index = torch.sort(outputs, 1)

            # Lip
            tmp = self.evaluate_certified_radius_lip(lip_cst, margins)
            lip_cert_rad.append(tmp * correct)

            # Curv
            if isinstance(self.model, nn.DataParallel):
                activ = self.model.module.model.stable_block[0].activation
            else:
                activ = self.model.model.stable_block[0].activation
            if not isinstance(activ, nn.ReLU):
                tmp, grad_norm, M, allActiveCurvatures = self.secondOrderCert(inputs, margins, index)
                Ms.append(M); grad_norms.append(grad_norm); marginss.append(margins)
                curv_cert_rad.append(tmp * correct)
            # normalizedInputs = self.model.module.normalize(inputs)
            # if isinstance(self.model, nn.DataParallel):
            #     activ = self.model.module.model.updateLipschitz(normalizedInputs, 0.14, allActiveCurvatures)
            # else:
            #     activ = self.model.model.updateLipschitz(normalizedInputs, 0.14, allActiveCurvatures)
            # raise
            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            running_inputs += inputs.size(0)

        self.model.train()
        accuracy = running_accuracy / running_inputs
        print(accuracy)

        curv_cert_rad = utils.stackIfNonempty(curv_cert_rad)
        lip_cert_rad = utils.stackIfNonempty(lip_cert_rad)
        margins = utils.stackIfNonempty(marginss)
        grad_norm = utils.stackIfNonempty(grad_norms)
        M = utils.stackIfNonempty(Ms)
        if len(curv_cert_rad) > 0:
            cert_rad = torch.maximum(curv_cert_rad, lip_cert_rad)
        else:
            cert_rad = lip_cert_rad
        corrects = utils.stackIfNonempty(corrects)
        return accuracy, cert_rad, lip_cert_rad, curv_cert_rad, grad_norm, margins, corrects, M

    # @torch.no_grad()
    # def eval_certified(self, eps):
    #     print("evaluating certified accuracy")
    #     eps_float = eps / 255
    #     self.model.eval()
    #     running_accuracy = 0
    #     running_certified = 0
    #     running_certified_lip, running_certified_lip_imp, running_certified_curv = 0, 0, 0
    #     running_inputs = 0
    #     lip_cst = np.sqrt(2.)
    #     data_loader, _ = self.reader.load_dataset()
    #     for batch_n, data in tqdm(enumerate(data_loader), total=len(data_loader)):
    #         inputs, labels = data
    #         inputs, labels = inputs.cuda(), labels.cuda()
    #         outputs = self.model(inputs)
    #         predicted = outputs.argmax(axis=1)
    #         correct = outputs.max(1)[1] == labels
    #         margins, index = torch.sort(outputs, 1)
    #         # Lip sqrt 2
    #         certified_lip = (margins[:, -1] - margins[:, -2]) >  lip_cst * eps_float
    #         running_certified_lip += torch.sum(correct & certified_lip).item()
    #         # Curv
    #         cert_tmp = torch.zeros_like(certified_lip)
    #         if isinstance(self.model, nn.DataParallel):
    #             activ = self.model.module.model.stable_block[0].activation
    #         else:
    #             activ = self.model.model.stable_block[0].activation
    #         if isinstance(activ, nn.Tanh):
    #             cert_tmp, __, lip_improved = self.secondOrderCert(inputs, margins, index, eps_float)
    #             running_certified_curv += torch.sum(correct & cert_tmp).item()
    #             lip_improved = torch.minimum(lip_improved, torch.ones_like(lip_improved) * lip_cst)
    #             certified_lip_imp = (margins[:, -1] - margins[:, -2]) >  lip_improved * eps_float
    #             running_certified_lip_imp += torch.sum(correct & certified_lip_imp).item()

    #         running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
    #         certified = torch.logical_or(cert_tmp, torch.logical_or(certified_lip, certified_lip_imp))
    #         running_certified += torch.sum(correct & certified).item()
    #         running_inputs += inputs.size(0)
    #         break
    #     self.model.train()
    #     accuracy = running_accuracy / running_inputs
    #     certified = running_certified / running_inputs
    #     self.message.add('eps', [eps, 255], format='.0f')
    #     self.message.add('eps', eps_float, format='.5f')
    #     self.message.add('accuracy', accuracy, format='.5f')
    #     self.message.add('certified acc', certified, format='.5f')
    #     self.message.add('certified acc lip', running_certified_lip / running_inputs, format='.5f')
    #     self.message.add('certified acc lip imp', running_certified_lip_imp / running_inputs, format='.5f')
    #     self.message.add('certified acc curv', running_certified_curv / running_inputs, format='.5f')

    #     if self.wandb:
    #         wandb.log({"accuracy": accuracy, "certified accuracy " + str(eps): certified})
    #     message = self.message.get_message()
    #     logging.info(message)
    #     print(message)

    #     return accuracy, certified
    
    def secondOrderCert(self, inputs, margins, index, useQueryCoefficients=False):
        queryCoefficient = torch.zeros(inputs.shape[0], self.model.module.model.n_classes).cuda()
        queryCoefficient[torch.arange(inputs.shape[0]), index[:, -1]] += 1
        queryCoefficient[torch.arange(inputs.shape[0]), index[:, -2]] += -1
        def grad_f(x):
            return torch.einsum('ij,ij->i', queryCoefficient, self.model(x))

        # print(inputs.shape, queryCoefficient.shape)
        grad = jacobian(grad_f, inputs).sum(dim=1).reshape(inputs.shape[0], -1)
        grad_norm = torch.linalg.norm(grad, 2, dim=1, keepdim=True)
        normalizedInputs = self.model.module.normalize(inputs)  # We make the assumption that the normalization only
        # shifts the mean and does not alter the variance.
        anchorDerivativeTerm = False
        returnAll = True
        if useQueryCoefficients:
            M, allActiveCurvatures =\
                self.model.module.model.calculateCurvature(queryCoefficient,
                                                           localPoints=normalizedInputs,
                                                           returnAll=returnAll,
                                                           anchorDerivativeTerm=anchorDerivativeTerm)
        else:
            M, allActiveCurvatures =\
                self.model.module.model.calculateCurvature(localPoints=normalizedInputs,
                                                           returnAll=returnAll,
                                                           anchorDerivativeTerm=anchorDerivativeTerm)
        if True:
            diff = margins[:, -2:-1] - margins[:, -1:]
            cert_rad = (-grad_norm + torch.sqrt(grad_norm**2 - 2 * M * diff)) / M
        else:
            assert M.numel() == 1  # M shouldn't be local
            cert_rad = newton_step_cert(inputs, index[:, -1], index[:, -2], self.model, M, queryCoefficient)

        return cert_rad, grad_norm, M, allActiveCurvatures


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
        calculatePgdAccuracy(self.model, self.reader.load_dataset()[0], \
                            self.reader.means, self.config.eps / 255, M=self.model.module.model.calculateCurvature().item())
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

    def newton_cert_rad(self, M):
        self.model.eval()
        radiusDistCurv = []

        data_loader, _ = self.reader.load_dataset()

        print("evaluating certified accuracy")
        for batch_n, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            correct = outputs.max(1)[1] == labels
            margins, index = torch.sort(outputs, 1)
            tmpRad = newton_step_cert(inputs, index[:, -1], index[:, -2], \
                                              self.model, M[batch_n*inputs.shape[0]:(batch_n+1)*inputs.shape[0]])
            radiusDistCurv.append((tmpRad * correct))
        radiusDistCurv = torch.hstack(radiusDistCurv)
        return radiusDistCurv

    
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
            if not isinstance(activ, nn.ReLU):
                tmp_radius, __, __, _ = self.secondOrderCert(inputs, margins, index, eps_float)
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
                         dataloaders=None, M=torch.inf):
    # converting the eps that is for the range [0, 1] to the range of the normalized dataset:
    lowerBound = float("inf")
    upperBound = float("-inf")
    for x in datasetShift:
        lowerBound = min(lowerBound, -x)
        upperBound = max(upperBound, (1 - x))

    accuracies = (1 - evaluate_madry(dataset, net, eps, False, device, lBall=lBall,
                                     lowerBound=lowerBound, upperBound=upperBound).item()) * 100
    # alpha=1/M
    # show_pdg_cmp(dataset, net, eps, alpha, device,lBall=lBall,\
    #                                  lowerBound=lowerBound, upperBound=upperBound)
    # raise
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

def _pgdPlot(model, X, Y, epsilon, niters=20, alpha=0.001, lBall=torch.inf, lowerBound=0, upperBound=1):
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, Y)
    err = (out.data.max(1)[1] != Y.data).float().sum() / X.size(0)
    losses = []
    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters):
        # print(i, model(X_pgd))
        opt = optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        out = model(X_pgd)
        # out = out - out.mean(1, keepdim=True)
        # loss = nn.CrossEntropyLoss()(out, Y)
        loss = -(1 + nn.functional.softmax(out, dim=1)[range(Y.shape[0]), Y]).log().mean()
        losses.append(loss.item())

        loss.backward()
        # eta = alpha * X_pgd.grad.data.sign()
        eta = 1000 * alpha * X_pgd.grad.data * torch.maximum(X_pgd.grad.data.norm(2, (1, 2, 3), keepdim=True)/2, \
                                                             torch.ones_like(X_pgd.grad.data.norm(2, (1, 2, 3), keepdim=True)))

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

            X_pgd = Variable(torch.clamp(X.data +  y * torch.maximum(epsilon/ norm, torch.ones_like(norm)), lowerBound, upperBound), requires_grad=True)
            # print(X_pgd)
        elif lBall == 1:
            raise NotImplementedError
        else:
            raise ValueError

    err_pgd = (model(X_pgd).data.max(1)[1] != Y.data).float().sum() / X.size(0)
    return err, err_pgd, losses

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


def show_pdg_cmp(loader, model, epsilon, alpha=0, device=torch.device("cuda"), lBall=None, lowerBound=-1, upperBound=1):
    model.eval()
    for i, (X, y) in tqdm(enumerate(loader), total=len(loader)):
        X, y = X.to(device), y.to(device)
        X = X[110:111]
        y = y[110:111]
        out = model(X)
        # grad = jacobian(lambda x: nn.CrossEntropyLoss()(model(x), y), X).sum(0).unsqueeze(0)
        grad = jacobian(lambda x: model(x)[:, y], X).sum(0).unsqueeze(0)
        grad2 = jacobian(lambda x: model(x)[:, 0], X).sum(0).unsqueeze(0)
        t = torch.linspace(-1, 1, 20)
        t1, t2 = torch.meshgrid(t, t)
        t = torch.stack([t1.flatten(), t2.flatten()], 1).to(device)
        outs = []
        for tt1, tt2 in t:
            XX = X + tt1 * grad / torch.linalg.norm(grad.flatten(1), 2, 1) + tt2 * grad2 / torch.linalg.norm(grad2.flatten(1), 2, 1)
            XX = XX.squeeze(0).squeeze(0)
            out = model(XX)
            loss = nn.CrossEntropyLoss(reduction='none')(out, y * torch.ones_like(out))
            outs.append(loss.item())
        outs = np.vstack(outs).reshape(t1.shape)
        # XX = X + t * grad / torch.linalg.norm(grad.flatten(1), 2, 1)
        # out = model(XX)
        # loss = nn.CrossEntropyLoss(reduction='none')(out, y * torch.ones_like(out))
        # plt.plot(t.detach().cpu().numpy().reshape(-1, ), loss.detach().cpu().numpy())
        ax = plt.axes(projection='3d')
        ax.plot_surface(t1, t2, outs)
        plt.show()
    raise




    # perrors = AverageMeter()
    # epsilon = 0.1
    # alphas=[0.001 * alpha, 0.1 * alpha, alpha, 10 * alpha]
    # print(epsilon, alpha, lBall, lowerBound, upperBound)
    # for alpha in alphas:
    #     losses = []
    #     for i, (X, y) in tqdm(enumerate(loader), total=len(loader)):
    #         X, y = X.to(device), y.to(device)

    #         # # perturb
    #         _, pgd_err, loss = _pgdPlot(model, Variable(X), Variable(y), epsilon, alpha=alpha, lBall=lBall,
    #                         lowerBound=lowerBound, upperBound=upperBound, niters=10)
    #         losses.append(loss)
    #         perrors.update(pgd_err, X.size(0))
    #     losses = np.array(losses)
    #     # print(losses)
    #     plt.plot(losses.mean(0), label='alpha={}'.format(round(alpha, 4)))
    #     print(1 - perrors.avg)
    # plt.legend()
    # plt.show()
    # raise
    # return perrors.avg
    
        

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




def newton_step_cert(x0, true_label, false_target, model, M, verbose=True):
        queryCoefficient = torch.zeros(x0.shape[0], model.module.model.n_classes).cuda()
        queryCoefficient[torch.arange(x0.shape[0]), true_label] += 1
        queryCoefficient[torch.arange(x0.shape[0]), false_target] += -1
        def grad_f(x):
            return torch.einsum('ij,ij->i', queryCoefficient, model(x)).sum()

        batch_size = x0.shape[0]

        if M.numel() == 1:
            M = M[0, 0]
        else:
            M = M.unsqueeze(2).unsqueeze(3)
        eta = torch.zeros((batch_size, 1)).cuda()
        eta_min = -1/M*torch.ones((batch_size, 1, 1, 1)).cuda()
        eta_max =  1/M*torch.ones((batch_size, 1, 1, 1)).cuda()
        eta = (eta_min + eta_max)/2.
        

        x = x0.clone()
        outer_iters = 20
        inner_iters = 20
        for i in range(outer_iters):
            for j in range(inner_iters):
                g_batch = jacobian(grad_f, x).reshape(x0.shape)
                dual_grad = eta*g_batch - eta*M*x - x0
                dual_hess = 1 + eta*M
                x = -torch.reciprocal(dual_hess)*dual_grad


            if i < outer_iters:
                logits = model(x)
                logits_diff = logits[torch.arange(batch_size), true_label] - logits[torch.arange(batch_size), false_target]
                ge_indicator = (logits_diff > 0)
                eta_min[ge_indicator] = eta[ge_indicator]
                eta_max[~ge_indicator] = eta[~ge_indicator]
                eta = (eta_min + eta_max)/2.

        dist_sqrd = torch.linalg.norm((x-x0).flatten(1), 2, dim=1)**2 + 2*eta[:, 0, 0, 0]*logits_diff
        lower_bound = torch.sqrt((dist_sqrd>0).float()*dist_sqrd).detach()
        grad_norm = torch.norm((eta*g_batch + (x - x0)).flatten(1), 2, dim=1).detach()

        return lower_bound * (grad_norm < 1e-5)


# def gradient_x(x, model, true_label, false_target):
#     x_var = x.clone()
#     x_var.requires_grad = True
#     batch_size = x_var.shape[0]
#     with torch.enable_grad():
#         logits = model(x_var)
#         logits_diff = logits[torch.arange(batch_size), true_label] - logits[torch.arange(batch_size), false_target]
#     grad_x = torch.autograd.grad(logits_diff.sum(), x_var)[0]
#     return grad_x


def curvature_hessian_estimator(model: torch.nn.Module,
                                image: torch.Tensor,
                                target: torch.Tensor,
                                num_power_iter: int = 20,
                                queryCoefficient=None):
    model.eval()
    u = torch.randn_like(image)
    u /= torch.norm(u, p=2, dim=(1, 2, 3), keepdim=True)

    with torch.enable_grad():
        image = image.requires_grad_()
        out = model(image)
        if queryCoefficient is None:
            y = F.log_softmax(out, 1)
            output = F.nll_loss(y, target, reduction='none')
            model.zero_grad()
            # Gradients w.r.t. input
            gradients = torch.autograd.grad(outputs=output.sum(),
                                            inputs=image, create_graph=True)[0]
        else:
            output = torch.einsum('bi, bi -> b', queryCoefficient, out)
            gradients = torch.autograd.grad(outputs=output.sum(),
                                            inputs=image, create_graph=True)[0]

        gnorm = torch.norm(gradients, p=2, dim=(1, 2, 3))
        assert not gradients.isnan().any()

        # Power method to find singular value of Hessian
        for _ in range(num_power_iter):
            grad_vector_prod = (gradients * u.detach_()).sum()
            hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image, retain_graph=True)[0]
            assert not hessian_vector_prod.isnan().any()

            hvp_norm = torch.norm(hessian_vector_prod, p=2, dim=(1, 2, 3), keepdim=True)
            u = hessian_vector_prod.div(hvp_norm + 1e-6)  # 1e-6 for numerical stability

        grad_vector_prod = (gradients * u.detach_()).sum()
        hessian_vector_prod = torch.autograd.grad(outputs=grad_vector_prod, inputs=image)[0]
        hessian_singular_value = (hessian_vector_prod * u.detach_()).sum((1, 2, 3))

    # curvature = hessian_singular_value / (grad_norm + epsilon) by definition
    curvatures = hessian_singular_value.abs().div(gnorm + 1e-6)
    hess = hessian_singular_value.abs()
    grad = gnorm

    return curvatures, hess, grad


def measure_curvature(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      numberOfClasses,
                      data_fraction: float = 0.1,
                      batch_size: int = 64,
                      num_power_iter: int = 20,
                      device: torch.device = 'cpu'):
    """
    Compute curvature, hessian norm and gradient norm of a subset of the data given by the dataloader.
    These values are computed using the power method, which requires setting the number of power iterations (num_power_iter).
    """

    model.eval()
    datasize = int(data_fraction * len(dataloader.dataset))
    max_batches = int(datasize / batch_size)
    curvature_agg = torch.zeros(size=(datasize,))
    grad_agg = torch.zeros(size=(datasize,))
    hess_agg = torch.zeros(size=(datasize,))

    maximumM = 0
    for idx, (data, target) in tqdm(enumerate(dataloader), total=max_batches - 1):
        data, target = data.to(device).requires_grad_(), target.to(device)
        with torch.no_grad():
            queryCoefficient = torch.zeros(data.shape[0], numberOfClasses).cuda()
            queryCoefficient[torch.arange(data.shape[0]), target] += 1
            queryCoefficient[torch.arange(data.shape[0]), (target + 1) % numberOfClasses] += -1


            curvatures, hess, grad = curvature_hessian_estimator(model, data, target, num_power_iter=num_power_iter,
                                                                 queryCoefficient=queryCoefficient)

            normalizedInputs = model.module.normalize(data)
            queryCoefficient = None
            # normalizedInputs = None
            # is the local point thing wrong? Why is it worse when we provide it?
            M = model.module.model.calculateCurvature(queryCoefficient=queryCoefficient,
                                                      localPoints=normalizedInputs,
                                                      returnAll=False,
                                                      anchorDerivativeTerm=False)
            newMax = torch.max(M)
            if newMax > maximumM:
                maximumM = newMax
            # M /= np.sqrt(2)  # is multiplied extra by this term.
            # M = np.sqrt(numberOfClasses) * 1 + numberOfClasses * (numberOfClasses - 1) * M


        # print(M)
        # print(hess)
        curvature_agg[idx * batch_size:(idx + 1) * batch_size] = curvatures.detach()
        hess_agg[idx * batch_size:(idx + 1) * batch_size] = hess.detach()
        grad_agg[idx * batch_size:(idx + 1) * batch_size] = grad.detach()

        avg_curvature, std_curvature = curvature_agg.mean().item(), curvature_agg.std().item()
        avg_hessian, std_hessian = hess_agg.mean().item(), hess_agg.std().item()
        avg_grad, std_grad = grad_agg.mean().item(), grad_agg.std().item()




        if idx == (max_batches - 1):
            print('Average Curvature: {:.6f} +/- {:.2f} '.format(avg_curvature, std_curvature))
            print('Average Hessian Spectral Norm: {:.6f} +/- {:.2f} '.format(avg_hessian, std_hessian))
            print('Average Gradient Norm: {:.6f} +/- {:.2f}'.format(avg_grad, std_grad))

            maxCurvature = torch.max(curvature_agg)
            maxHessian = torch.max(hess_agg)
            print('Maximum Curvature: {:.6f} '.format(maxCurvature))
            print('Maximum Hessian Spectral Norm: {:.6f} '.format(maxHessian))
            print('Maximum Gradient Norm: {:.6f}'.format(torch.max(grad_agg)))

            print('Maximum M:', maximumM)
            return maxCurvature, maxHessian