import logging
import random
import math
from os.path import join, exists

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from core.data.tiny_imagenet import TinyImageNet
import torchvision


class BaseReader:

  def __init__(self, config, batch_size, is_distributed, is_training):
    self.config = config
    self.batch_size = batch_size
    self.is_training = is_training
    self.is_distributed = is_distributed
    self.num_workers = 10
    self.prefetch_factor = self.batch_size * 2
    # self.path = join(self.get_data_dir(), self.config.dataset)
    self.path = join("../data", self.get_data_dir())


  def get_data_dir(self):
    if self.config.dataset in ['cifar10', 'cifar100']:
      return "cifarpy"
    elif self.config.dataset == 'mnist':
      return ""
    else:
      raise NotImplementedError
    paths = self.config.data_dir.split(':')
    data_dir = None
    for path in paths:
      if exists(join(path, self.config.dataset)):
        data_dir = path
        break
    if data_dir is None:
      raise ValueError("Data directory not found.")
    return data_dir

  def transform(self):
    """Create the transformer pipeline."""
    raise NotImplementedError('Must be implemented in derived classes')

  def load_dataset(self):
    """Load or download dataset."""
    sampler = None
    if self.is_distributed:
      sampler = DistributedSampler(self.dataset, shuffle=self.is_training)
    loader = DataLoader(self.dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=self.is_training and not sampler,
                        pin_memory=True,
                        prefetch_factor=self.prefetch_factor,
                        sampler=sampler)
    return loader, sampler

class MNISTReader(BaseReader):
  def __init__(self, config, batch_size, num_gpus, is_training):
    super(MNISTReader, self).__init__(
      config, batch_size, num_gpus, is_training)
    self.n_classes = 10
    self.means = (0.0000, )
    self.stds = (1.0000, )
    self.is_training = is_training = is_training

    self.batch_size = batch_size
    self.height, self.width = 28, 28
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.img_size = (None, 1, 28, 28)

    if config.shift_data:
      self.means = (0.1307,)

    # if self.is_training:
    #   transform = self.transform()
    # else:
    transform = transforms.Compose(
      [transforms.ToTensor(),
      ])

    self.dataset = torchvision.datasets.MNIST(root=self.path, train=self.is_training,
                                            download=True, transform=transform)

  def transform(self):
    hue = 0.02
    saturation = (.3, 2.)
    brightness = 0.1
    contrast = (.5, 2.)
    if self.is_training:
      transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
          brightness=brightness, contrast=contrast,
          saturation=saturation, hue=hue),
        transforms.ToTensor(),
      ])
    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
      ])
    return transform


class CIFARReader(BaseReader):

  def __init__(self, config, batch_size, num_gpus, is_training):
    super(CIFARReader, self).__init__(
      config, batch_size, num_gpus, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.img_size = (None, 3, 32, 32)

  def transform(self):
    hue = 0.02
    saturation = (.3, 2.)
    brightness = 0.1
    contrast = (.5, 2.)
    if self.is_training:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
          brightness=brightness, contrast=contrast, 
          saturation=saturation, hue=hue),
        transforms.ToTensor(),
      ])
    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
      ])
    return transform


class CIFAR10Reader(CIFARReader):

  def __init__(self, config, batch_size, num_gpus, is_training):
    super(CIFAR10Reader, self).__init__(
      config, batch_size, num_gpus, is_training)
    self.n_classes = 10
    self.means = (0.0000, 0.0000, 0.0000)
    self.stds = (1.0000, 1.0000, 1.0000)
    if config.shift_data:
      self.means = (0.4913, 0.4821, 0.4465)

    transform = self.transform()
    # self.dataset = CIFAR10(self.path, train=self.is_training,
    #                        download=False, transform=transform)
    # trainTransform = transforms.Compose(
    #   [transforms.ToTensor(),
    #    transforms.RandomHorizontalFlip()
    #    ])

    # testTransform = transforms.Compose(
    #   [transforms.ToTensor(),
    #    ])
    self.dataset = torchvision.datasets.CIFAR10(root=self.path, train=self.is_training,
                                                download=True, transform=transform)
    # self.dataset = torchvision.datasets.CIFAR10(root='../data', train=self.is_training,
    #                                         download=True, transform=testTransform)


class CIFAR100Reader(CIFARReader):

  def __init__(self, config, batch_size, num_gpus, is_training):
    super(CIFAR100Reader, self).__init__(
      config, batch_size, num_gpus, is_training)
    self.n_classes = 100
    self.means = (0.0000, 0.0000, 0.0000)
    self.stds = (1.0000, 1.0000, 1.0000)
    if config.shift_data:
      self.means = (0.5071, 0.4865, 0.4409)

    transform = self.transform()
    self.dataset = CIFAR100(self.path, train=self.is_training,
                           download=True, transform=transform)


class TinyImageNetReader(BaseReader):

  def __init__(self, config, batch_size, num_gpus, is_training):
    super(TinyImageNetReader, self).__init__(
      config, batch_size, num_gpus, is_training)
    raise NotImplementedError('TinyImageNetReader data dir not correct. Fix ...')
    self.batch_size = batch_size
    self.is_training = is_training
    self.n_classes = 200
    self.height, self.width = 64, 64
    self.n_train_files = 100000
    self.n_test_files = 10000
    self.img_size = (None, 3, 64, 64)

    self.means = (0.0000, 0.0000, 0.0000)
    self.stds = (1.0000, 1.0000, 1.0000)
    if config.shift_data:
      self.means = (0.485, 0.456, 0.406)

    split = 'train' if self.is_training else 'val'
    transform = self.transform()
    self.dataset = TinyImageNet(self.path, split=split,
                                download=False, transform=transform)

  def transform(self):
    hue = 0.02
    saturation = (.3, 2.)
    brightness = 0.1
    contrast = (.5, 2.)
    if self.is_training:
      transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
          brightness=brightness, contrast=contrast, 
          saturation=saturation, hue=hue),
        transforms.ToTensor(),
      ])
    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
      ])
    return transform






readers_config = {
  'mnist': MNISTReader,
  'cifar10': CIFAR10Reader,
  'cifar100': CIFAR100Reader,
  'tiny-imagenet': TinyImageNetReader
}