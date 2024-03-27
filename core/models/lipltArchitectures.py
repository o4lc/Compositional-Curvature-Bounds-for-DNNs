import torch.nn as nn
from core.models.lipltModels import *


def getNetworkArchitecture(architectureName):
    # image width after a conv layer: width - kernelSize + 1
    if architectureName == "6F":
        # 6F, CIFAR10
        layers = [("flatten", None),
                ("linear", 3072, 1024),
                ("linear", 1024, 512),
                ("linear", 512, 256),
                ("linear", 256, 256),
                ("linear", 256, 128),
                ("linear", 128, 10),]
    elif architectureName == "3F":
        # 3F, MNIST
        layers = [("flatten", None),
                  ("linear", 784, 100),
                  ("linear", 100, 100),
                  ("linear", 100, 10)]
    elif architectureName == "vggSim":
        # similar to vgg, MNIST
        layers = [("conv2d", 3, 64, 7),
                  ("conv2d", 64, 64, 7),
                  ("conv2d", 64, 128, 7),
                  ("conv2d", 128, 32, 7),
                  ("flatten",),
                  ("linear", 2048, 256),
                  ("linear", 256, 10)]
    elif architectureName == "2C2F":
        # 2C2F, MNIST
        # layers = [("conv2d", 1, 16, 4, 2,),
        #           ("conv2d", 16, 32, 4, 2,),
        #           ("flatten",),
        #           ("linear", 800, 100),
        #           ("linear", 100, 10)]
        layers = [("conv2d", 1, 16, 4, 2, 1),
                  ("conv2d", 16, 32, 4, 2, 1),
                  ("flatten",),
                  ("linear", 1568, 100),
                  ("linear", 100, 10)]
    elif architectureName == "4C3F":
        # 4C3F, MNIST
        # layers = [("conv2d", 1, 32, 3, 1, 1),
        #           ("conv2d", 32, 32, 4, 2,),
        #           ("conv2d", 32, 64, 3, 1, 1),
        #           ("conv2d", 64, 64, 4, 2,),
        #           ("flatten",),
        #           ("linear", 1600, 512),
        #           ("linear", 512, 512),
        #           ("linear", 512, 10)]
        layers = [("conv2d", 1, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("flatten",),
                  ("linear", 3136, 512),
                  ("linear", 512, 512),
                  ("linear", 512, 10)]
    elif architectureName == "6C2FMNIST":
        layers = [("conv2d", 1, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 0),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 2),
                  ("flatten",),
                  ("linear", 1024, 512),
                  ("linear", 512, 10)]
    elif architectureName == "6C1F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("flatten",),
                  ("linear", 1024, 10)]
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
    elif architectureName == "6C2FTinyImagenet":
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("flatten",),
                  ("linear", 4096, 512),
                  ("linear", 512, 200)]
    elif architectureName == "6C3F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("flatten",),
                  ("linear", 1024, 512),
                  ("linear", 512, 512),
                  ("linear", 512, 10)]
    elif architectureName == "6C4F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("flatten",),
                  ("linear", 1024, 512),
                  ("linear", 512, 512),
                  ("linear", 512, 256),
                  ("linear", 256, 10)]
    elif architectureName == "7C2F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("flatten",),
                  ("linear", 1024, 512),
                  ("linear", 512, 10)]
    elif architectureName == "8C2FCifar10":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("flatten",),
                  ("linear", 256, 256),
                  ("linear", 256, 10)]
    elif architectureName == "9C2F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("flatten",),
                  ("linear", 256, 256),
                  ("linear", 256, 10)]
    elif architectureName == "9C3F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("flatten",),
                  ("linear", 256, 128),
                  ("linear", 128, 128),
                  ("linear", 128, 10)]
    elif architectureName == "9C4F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("flatten",),
                  ("linear", 256, 128),
                  ("linear", 128, 128),
                  ("linear", 128, 64),
                  ("linear", 64, 10)]
    elif architectureName == "10C4F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("flatten",),
                  ("linear", 64, 64),
                  ("linear", 64, 64),
                  ("linear", 64, 32),
                  ("linear", 32, 10)]
    elif architectureName == "11C4F":
        # 6C2F, CIFAR10
        layers = [("conv2d", 3, 32, 3, 1, 1),
                  ("conv2d", 32, 32, 4, 2, 1),
                  ("conv2d", 32, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("flatten",),
                  ("linear", 64, 64),
                  ("linear", 64, 64),
                  ("linear", 64, 32),
                  ("linear", 32, 10)]
    elif architectureName == "8C2FOriginal":
        layers = [("conv2d", 3, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 4, 2, 0),
                  ("conv2d", 64, 128, 3, 1, 1),
                  ("conv2d", 128, 128, 3, 1, 1),
                  ("conv2d", 128, 128, 4, 2, 0),
                  ("conv2d", 128, 256, 3, 1, 1),
                  ("conv2d", 256, 256, 4, 2, 0),
                  ("flatten",),
                  ("linear", 9216, 256),
                  ("linear", 256, 200)]
    elif architectureName == "8C2F":
        layers = [("conv2d", 3, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 3, 1, 0),
                  ("conv2d", 64, 64, 4, 2, 0),
                  ("conv2d", 64, 128, 3, 1, 1),
                  ("conv2d", 128, 128, 3, 1, 1),
                  ("conv2d", 128, 128, 4, 2, 0),
                  ("conv2d", 128, 256, 3, 1, 1),
                  ("conv2d", 256, 256, 4, 2, 0),
                  ("flatten",),
                  # ("linear", 16384, 256),
                  ("linear", 9216, 256),
                  ("linear", 256, 200)]
    elif architectureName == "8C2FBatchNorm":
        layers = [("conv2dBatchNorm", 3, 64, 3, 1, 1),
                  ("conv2dBatchNorm", 64, 64, 3, 1, 0),
                  ("conv2dBatchNorm", 64, 64, 4, 2, 0),
                  ("conv2dBatchNorm", 64, 128, 3, 1, 1),
                  ("conv2dBatchNorm", 128, 128, 3, 1, 1),
                  ("conv2dBatchNorm", 128, 128, 4, 2, 0),
                  ("conv2dBatchNorm", 128, 256, 3, 1, 1),
                  ("conv2dBatchNorm", 256, 256, 4, 2, 0),
                  ("flatten",),
                  ("linear", 9216, 256),
                  ("linear", 256, 200)]

    elif architectureName == "7C1F":
        layers = [("conv2d", 3, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 3, 1, 0),
                  ("conv2d", 64, 64, 4, 2, 0),
                  ("conv2d", 64, 128, 3, 1, 1),
                  ("conv2d", 128, 128, 3, 1, 1),
                  ("conv2d", 128, 128, 4, 2, 0),
                  ("conv2d", 128, 256, 3, 1, 1),
                  ("flatten",),
                  ("linear", 9216, 100)]
    elif architectureName == "8C2FCIFAR100":
        layers = [("conv2d", 3, 64, 3, 1, 1),
                  ("conv2d", 64, 64, 3, 1, 0),
                  ("conv2d", 64, 64, 4, 2, 0),
                  ("conv2d", 64, 128, 3, 1, 1),
                  ("conv2d", 128, 128, 3, 1, 1),
                  ("conv2d", 128, 128, 4, 2, 0),
                  ("conv2d", 128, 256, 3, 1, 1),
                  ("conv2d", 256, 256, 4, 2, 0),
                  ("flatten",),
                  # ("linear", 16384, 256),
                  ("linear", 1024, 256),
                  ("linear", 256, 100)]
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
        net = SequentialNaiveLipschitz(layers, networkConfiguration['inputShape'], device,
                                       networkConfiguration['perClassLipschitz'],
                                       networkConfiguration['numberOfPowerIterations'],
                                       networkConfiguration['weightInitialization'],
                                       networkConfiguration['pairwiseLipschitz'],
                                       activation)
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


