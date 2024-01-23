import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from core.models.layers import LinearNormalized, PoolingLinear, PaddingChannels
from core.models.layers import SDPBasedLipschitzConvLayer, SDPBasedLipschitzLinearLayer
import numpy as np


class NormalizedModel(nn.Module):

  def __init__(self, model, mean, std):
    super(NormalizedModel, self).__init__()
    self.model = model
    self.normalize = Normalize(mean, std)

  def forward(self, x):
    return self.model(self.normalize(x))


class LipschitzNetwork(nn.Module):

  def __init__(self, config, n_classes, activation='relu'):
    super(LipschitzNetwork, self).__init__()

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
      layers.append(block_conv(config, (1, self.num_channels, imsize, imsize), self.num_channels, self.conv_size, activation=activation))
    

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

  def calculateCurvature(self, queryCoefficient=None):
    curvatureTillHere = 0
    lip = 1.
    for layer in self.stable_block:
      if isinstance(layer, SDPBasedLipschitzConvLayer):
        wNorm, gNorm = layer.calculateElementLipschitzs()
        layerJacobianLipschitz = 4/np.sqrt(27) * wNorm**2 * gNorm
        curvatureTillHere = layerJacobianLipschitz * lip ** 2 + lip * curvatureTillHere

    for layer in self.layers_linear:
      if isinstance(layer, SDPBasedLipschitzLinearLayer):
        wNorm, gNorm = layer.calculateElementLipschitzs()
        layerJacobianLipschitz = 4/np.sqrt(27) * wNorm**2 * gNorm
        curvatureTillHere = layerJacobianLipschitz * lip ** 2 + lip * curvatureTillHere

    if isinstance(self.last_last, LinearNormalized):
      weight = F.normalize(self.last_last.weight, p=2, dim=1)
      dims = (0, 1)
      if queryCoefficient is not None:
        weight = queryCoefficient @ weight
        dims = (1, )
      curvatureTillHere = torch.linalg.norm(weight, 2, dim=dims) * curvatureTillHere
    elif isinstance(self.last_last, PoolingLinear):
      pass
    else:
      raise NotImplementedError



    return curvatureTillHere





