import torch.nn as nn
from core.models.lipltModels import *


def getNetworkArchitecture(architectureName):
    # image width after a conv layer: width - kernelSize + 1
    if architectureName == "3F":
        # 3F, MNIST
        layers = [("flatten", None),
                  ("linear", 784, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 10)]
    elif architectureName == "4F":  
        # 4F, MNIST
        layers = [("flatten", None),
                  ("linear", 784, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 10)]
    elif architectureName == "5F":
        # 5F, MNIST
        layers  = [("flatten", None),
                   ("linear", 784, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 10)]
    elif architectureName == "6F":
        # 6F, MNIST
        layers = [("flatten", None),
                  ("linear", 784, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 10)]
    elif architectureName == "6C2F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("flatten",),
                  ("linear", 1024, 512),
                  ("linear", 512, 10)]
    elif architectureName == "6FCifar":
        # 6F, CIFAR10
        layers = [("flatten", None),
                ("linear", 3072, 1024),
                ("linear", 1024, 512),
                ("linear", 512, 256),
                ("linear", 256, 256),
                ("linear", 256, 128),
                ("linear", 128, 10),]
    else:
        raise NotImplementedError
    return layers






def createLipLtModel(networkConfiguration, device):
    layerConfigurations = networkConfiguration['layers']
    layers = []
    activation = networkConfiguration['activation']
    learnableBeta = networkConfiguration['learnableBeta']
    for layer in layerConfigurations:
        layerName = layer[0]
        if layerName == "flatten":
            layers.append(nn.Flatten())
        elif layerName == "linear":
            layers.append(nn.Linear(layer[1], layer[2], bias=True))
        elif layerName == "conv2d":
            stride = 1
            padding = 0
            if len(layer) >= 5:
                stride = layer[4]
            if len(layer) == 6:
                padding = layer[5]

            layers.append(nn.Conv2d(layer[1],
                                    layer[2],
                                    layer[3],
                                    stride, padding,
                                    bias=True))
        elif layerName == "conv2dBatchNorm":
            stride = 1
            padding = 0
            if len(layer) >= 5:
                stride = layer[4]
            if len(layer) == 6:
                padding = layer[5]
            layers.append(Conv2dBatchNorm(layer[1],
                                          layer[2],
                                          layer[3],
                                          stride, padding,
                                          bias=True))
        elif layerName == "activation":
            currentActivation = layer[0]
            if currentActivation == "relu":
                layers.append(nn.ReLU())
            elif currentActivation == "tanh":
                layers.append(nn.Tanh())
            elif currentActivation == "softplus":
                layers.append(Softplus(learnable=learnableBeta))
            elif currentActivation == "centered_softplus":
                layers.append(CenteredSoftplus(learnable=learnableBeta))
            else:
                raise ValueError
        else:
            raise ValueError
        if activation is not None and not layerName in ["activation", "flatten"]:
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "softplus":
                layers.append(Softplus())
            elif activation == "centered_softplus":
                layers.append(CenteredSoftplus())
            else:
                raise ValueError

    if activation is not None:
        layers.pop()
    if networkConfiguration['modelType'] == "naive":
        raise NotImplementedError("The current code base does not support this option."
                                  " If you want to use naive Lipschitz networks,"
                                  " please use the original codebase of CRM or modify this codebase accordingly. "
                                  "You just need to implement one function, I think, to calculate the curvature.")
        # net = SequentialNaiveLipschitz(layers, networkConfiguration['inputShape'], device,
        #                                networkConfiguration['perClassLipschitz'],
        #                                networkConfiguration['numberOfPowerIterations'],
        #                                networkConfiguration['weightInitialization'],
        #                                networkConfiguration['pairwiseLipschitz'],
        #                                activation)
    elif networkConfiguration['modelType'] == "liplt":
        net = SequentialLipltLipschitz(layers, networkConfiguration['inputShape'], device,
                                       networkConfiguration['perClassLipschitz'],
                                       networkConfiguration['numberOfPowerIterations'],
                                       networkConfiguration['weightInitialization'],
                                       networkConfiguration['pairwiseLipschitz'],
                                       activation)
    else:
        raise ValueError

    return net


