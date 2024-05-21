import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from core.models.layers import LinearNormalized, PoolingLinear, PaddingChannels
from core.models.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer
import numpy as np
from torch.autograd.functional import jacobian
from time import time
from tqdm import tqdm
from core.models.lipltArchitectures import createLipLtModel



class NormalizedModel(nn.Module):

    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))


def lipschitzModel(config, n_classes, activation='relu'):
    if config.model_name.startswith("liplt"):
        return createLipLtModel(config.networkConfiguration, torch.device("cuda"))
    else:
        return SllNetwork(config, n_classes, activation=activation)


class SllNetwork(nn.Module):

    def __init__(self, config, n_classes, activation='relu'):
        super(SllNetwork, self).__init__()

        self.activation = activation
        self.depth = config.depth
        self.num_channels = config.num_channels
        self.depth_linear = config.depth_linear
        self.n_features = config.n_features
        self.conv_size = config.conv_size
        self.n_classes = n_classes

        if config.dataset == 'tiny-imagenet':
            imsize = 64
        elif config.dataset in ['mnist']:
            imsize = 28
        else:
            imsize = 32

        self.conv1 = PaddingChannels(self.num_channels, 3, "zero")

        layers = []
        block_conv = SDPBasedLipschitzConvLayer
        block_lin = SDPBasedLipschitzLinearLayer

        for _ in range(self.depth):
            layers.append(block_conv(config, (1, self.num_channels, imsize, imsize), self.num_channels, self.conv_size,
                                     activation=activation))

        layers.append(nn.AvgPool2d(4, divisor_override=4))
        self.stable_block = nn.Sequential(*layers)

        layers_linear = [nn.Flatten()]

        if config.dataset == 'mnist':
            in_channels = self.num_channels * 7 * 7
        elif config.dataset in ['cifar10', 'cifar100']:
            in_channels = self.num_channels * 8 * 8
        elif config.dataset == 'tiny-imagenet':
            in_channels = self.num_channels * 16 * 16

        for _ in range(self.depth_linear):
            layers_linear.append(block_lin(config, in_channels, self.n_features, activation=activation))

        if config.last_layer == 'pooling_linear':
            self.last_last = PoolingLinear(in_channels, self.n_classes, agg="trunc")
        elif config.last_layer == 'lln':
            self.last_last = LinearNormalized(in_channels, self.n_classes)
        else:
            raise ValueError("Last layer not recognized")

        self.layers_linear = nn.Sequential(*layers_linear)
        self.base = nn.Sequential(*[self.conv1, self.stable_block, self.layers_linear])

    def forward(self, x):
        return self.last_last(self.base(x))

    def calculateCurvature(self, queryCoefficient=None, localPoints=None, returnAll=False, anchorDerivativeTerm=False):
        if localPoints is not None:
            originalInputs = localPoints.clone()
            localPoints = self.conv1(localPoints)
        allActiveCurvatures = []
        curvatureTillHere = 0
        lip = 1.
        for layerCount, layer in enumerate(self.stable_block):
            if isinstance(layer, SDPBasedLipschitzConvLayer):
                layerJacobianLipschitz = layer.calculateCurvature(localPoints=localPoints)

                # layerJacobianLipschitz = 4 / np.sqrt(27) * wNorm ** 2 * gNorm
                if anchorDerivativeTerm and localPoints is not None:
                    pass
                    localLip = lip
                else:
                    localLip = lip
                curvatureTillHere = layerJacobianLipschitz * lip * localLip + lip * curvatureTillHere
                allActiveCurvatures.append(curvatureTillHere)
            if localPoints is not None:
                localPoints = layer(localPoints)
        # curvatureOfConv = curvatureTillHere
        # curvatureTillHere = 0
        for layerCount, layer in enumerate(self.layers_linear):
            if isinstance(layer, SDPBasedLipschitzLinearLayer):
                # print("performing 100 power iterations")
                # for _ in range(100):
                #     layer.powerIterate(-2 / layer.computeT())
                layerJacobianLipschitz = layer.calculateCurvature(localPoints=localPoints)
                # layerJacobianLipschitz = 4 / np.sqrt(27) * wNorm ** 2 * gNorm
                if anchorDerivativeTerm and localPoints is not None and layerCount == self.depth_linear:

                    # derivatives = jacobian(
                    #     lambda x: self.layers_linear[:layerCount + 1](self.stable_block(self.conv1(x))).sum(0),
                    #     originalInputs, create_graph=False).permute(1, 0, 2, 3, 4).flatten(2)

                    # The following implementation uses the fact that the Lipschitz constant is 1.
                    localLip = []
                    miniBatchSize = 32
                    for i in tqdm(range(0, localPoints.shape[0], miniBatchSize)):
                    # for i in range(0, localPoints.shape[0], miniBatchSize):
                        derivatives = jacobian(
                            lambda x: self.layers_linear[:layerCount + 1](self.stable_block(self.conv1(x))).sum(0),
                            originalInputs[i:i + miniBatchSize], create_graph=False).permute(1, 0, 2, 3, 4).flatten(2)

                        # derivatives = jacobian(lambda x: layer(x).sum(0), localPoints[i:i + miniBatchSize],
                        #                        create_graph=False).permute(1, 0, 2)

                        localLip.append(torch.linalg.norm(derivatives, 2, dim=(1, 2)).unsqueeze(1))
                    localLip = torch.vstack(localLip)
                    print(localLip)
                    # derivatives = jacobian(lambda x: layer(x).sum(0), localPoints, create_graph=False).permute(1, 0, 2)
                    # localLip = torch.linalg.norm(derivatives, 2, dim=(1, 2)).unsqueeze(1)
                else:
                    localLip = lip

                curvatureTillHere = layerJacobianLipschitz * lip * localLip + lip * curvatureTillHere
                allActiveCurvatures.append(curvatureTillHere)
            if localPoints is not None:
                localPoints = layer(localPoints)
        # print(curvatureTillHere.mean())
        # print(curvatureOfConv.mean())
        # raise
        if isinstance(self.last_last, LinearNormalized):
            weight = F.normalize(self.last_last.weight, p=2, dim=1)
            dims = (0, 1)
            if queryCoefficient is not None:
                weight = queryCoefficient @ weight
                dims = (1,)
            else:
                weight = np.sqrt(2) * weight
            curvatureTillHere = torch.linalg.norm(weight, 2, dim=dims) * curvatureTillHere
        elif isinstance(self.last_last, PoolingLinear):
            curvatureTillHere = np.sqrt(2) * curvatureTillHere
        else:
            raise NotImplementedError

        # if localPoints is not None:
        #     originalCurvature = self.calculateCurvature()
        #     print("Original curvature: ", originalCurvature)
        #     print("Curvature till here: ", curvatureTillHere)
        #     print(curvatureTillHere / originalCurvature)
        #     raise
        if returnAll:
            return curvatureTillHere, allActiveCurvatures
        return curvatureTillHere

    def updateLipschitz(self, inputs, eps, hessians):
        raise NotImplementedError
        """
        Note that the "inputs" must be normalized by the model.normalize function. 
        """
        count = 0
        # inputs = self.conv1(inputs)
        updatedLipschitz = []

        # for layer in self.stable_block:
        #     count += 1
        #     if isinstance(layer, SDPBasedLipschitzConvLayer):
        #         derivatives = jacobian(lambda x: self.stable_block[:count](self.conv1(x)).sum(0),
        #                                inputs, create_graph=False).permute(3, 0, 1, 2, 4, 5, 6).flatten(4).flatten(1, 3)
        #         derivativeNorms = torch.linalg.norm(derivatives, 2, dim=(1, 2)).unsqueeze(1)
        #         updatedLipschitz.append(derivativeNorms + eps * hessians[count - 1])
        #         print(updatedLipschitz[-1], hessians[count - 1], derivativeNorms)


        derivatives = jacobian(lambda x: self.stable_block(self.conv1(x)).sum(0),
                                       inputs, create_graph=False).permute(3, 0, 1, 2, 4, 5, 6).flatten(4).flatten(1, 3)
        count = len(self.stable_block) - 1
        derivativeNorms = torch.linalg.norm(derivatives, 2, dim=(1, 2)).unsqueeze(1)
        updatedLipschitz.append(derivativeNorms + eps * hessians[count - 1])
        print(updatedLipschitz[-1], hessians[count - 1], derivativeNorms)


        numberOfLinears = 7
        derivatives = jacobian(lambda x: self.layers_linear[:numberOfLinears + 1](self.stable_block(self.conv1(x))),
                               inputs, create_graph=False).sum(0).permute(1, 0, 2, 3, 4).flatten(2)
        count = len(self.stable_block) - 1
        derivativeNorms = torch.linalg.norm(derivatives, 2, dim=(1, 2)).unsqueeze(1)
        updatedLipschitz.append(derivativeNorms + eps * hessians[count - 1 + numberOfLinears])
        print(updatedLipschitz[-1], hessians[count - 1 + numberOfLinears], derivativeNorms)

        derivatives = jacobian(lambda x: self.last_last(self.base(x)).sum(0),
                               inputs, create_graph=False).permute(1, 0, 2, 3, 4).flatten(2)
        derivativeNorms = torch.linalg.norm(derivatives, 2, dim=(1, 2)).unsqueeze(1)
        print(derivativeNorms)
        derivatives = jacobian(lambda x: self.base(x).sum(0),
                               inputs, create_graph=False).permute(1, 0, 2, 3, 4).flatten(2)
        derivativeNorms = torch.linalg.norm(derivatives, 2, dim=(1, 2)).unsqueeze(1)
        print(derivativeNorms)
        # If you want to update the linear layers, remember that AveragePooling has affected the "count" variable.
        pass

    def calcParams(self):
        sum = 0
        for parameter in self.parameters():
            if parameter.requires_grad:
                sum += parameter.shape.numel()
        return sum

    def miniBatchStep(self):
        pass  # required for other network models.

    def converge(self):
        for layer in self.stable_block:
            if isinstance(layer, SDPBasedLipschitzConvLayer):
                for _ in range(20):
                    T = self.computeT()
                    neg2TInv = -2 / T
                    layer.powerIterate(neg2TInv)
        for layerCount, layer in enumerate(self.layers_linear):
            if isinstance(layer, SDPBasedLipschitzLinearLayer):
                for _ in range(20):
                    T = self.computeT()
                    neg2TInv = -2 / T
                    layer.powerIterate(neg2TInv)
