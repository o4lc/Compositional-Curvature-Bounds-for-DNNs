import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(input):
    norm = torch.linalg.norm(input.flatten(1), 2, 1, True)
    while len(norm.shape) < len(input.shape):
        norm = norm.unsqueeze(-1)
    return input / norm


class SDPBasedLipschitzConvLayer(nn.Module):

    def __init__(self, config, input_size, cin, cout, kernel_size=3, epsilon=1e-6, activation='relu'):
        super(SDPBasedLipschitzConvLayer, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        else:
            self.activation = nn.Tanh()

        self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.randn(cout))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

        self.wEigen = nn.Parameter(normalize(torch.randn(input_size)), requires_grad=False)
        self.gEigen = \
          nn.Parameter(normalize(F.conv2d(torch.randn(input_size), self.kernel, padding=1).detach().clone()),
                       requires_grad=False)

    def computeT(self):
        method = "abs"
        if method == "abs":
            q = torch.abs(self.q)[None, :, None, None]
            qInv = 1 / torch.abs(self.q)
        elif method == "exp":
            q = torch.exp(self.q)[None, :, None, None]
            qInv = torch.exp(-self.q)
        kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)

        T = (torch.abs(q * kkt).sum((1, 2, 3)) * qInv)
        return T

    def forward(self, x):
        res = F.conv2d(x, self.kernel, bias=self.bias, padding=1)
        res = self.activation(res)
        T = self.computeT()[None, :, None, None]
        res = - 2 / T * res
        res = F.conv_transpose2d(res, self.kernel, padding=1)
        out = x + res

        with torch.no_grad():
            self.wEigen.data = \
              normalize(F.conv_transpose2d(F.conv2d(self.wEigen, self.kernel, padding=1), self.kernel, padding=1))
            self.gEigen.data = \
              normalize(-2 / T * F.conv2d(F.conv_transpose2d(
                -2 / T * self.gEigen, self.kernel, padding=1), self.kernel, padding=1))

        return out

    def calculateElementLipschitzs(self):
        T = self.computeT()[None, :, None, None]

        wForward = F.conv2d(self.wEigen, self.kernel, padding=1)
        wNorm = torch.linalg.norm(wForward.flatten(), 2)
        wNorm2Inf = torch.max(torch.linalg.norm(self.kernel.flatten(1), 2, 1))

        gForward = F.conv_transpose2d(-2 / T * self.gEigen, self.kernel, padding=1)
        gNorm = torch.linalg.norm(gForward.flatten(), 2)
        return wNorm, gNorm, wNorm2Inf


class SDPBasedLipschitzLinearLayer(nn.Module):

    def __init__(self, config, cin, cout, epsilon=1e-6, activation='relu'):
        super(SDPBasedLipschitzLinearLayer, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        else:
            self.activation = nn.Tanh()
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon
        self.gEigen = \
            nn.Parameter(normalize(torch.randn(1, cout)), requires_grad=False)
        self.wEigen = nn.Parameter(normalize(torch.randn((1, cin))), requires_grad=False)

    def computeT(self):
        method = "abs"  # "abs", "exp
        if method == "abs":
            q_abs = torch.abs(self.q)
            q = q_abs[None, :]
            q_inv = (1 / (q_abs + self.epsilon))[:, None]
        elif method == "exp":
            q = torch.exp(self.q)[None, :]
            q_inv = torch.exp(-self.q)[:, None]

        T = torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1)
        return T

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        T = self.computeT()
        res = -2 / T * res
        res = F.linear(res, self.weights.t())
        out = x + res

        # updating eigenVectors
        with torch.no_grad():
            self.wEigen.data = normalize(self.wEigen @ self.weights.T @ self.weights)
            gForward = (-2 / T * self.gEigen) @ self.weights
            gBackward = -2 / T * (gForward @ self.weights.T)
            self.gEigen.data = normalize(gBackward)
        return out

    def calculateElementLipschitzs(self):
        T = self.computeT()
        wNorm = torch.linalg.norm((self.wEigen @ self.weights.T).flatten(), 2)
        wNorm2Inf = torch.max(torch.linalg.norm(self.weights, 2, 1))
        gForward = (-2 / T * self.gEigen) @ self.weights
        gNorm = torch.linalg.norm(gForward.flatten(), 2)
        return wNorm, gNorm, wNorm2Inf

class PaddingChannels(nn.Module):

    def __init__(self, ncout, ncin=3, mode="zero"):
        super(PaddingChannels, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.mode = mode

    def forward(self, x):
        if self.mode == "clone":
            return x.repeat(1, int(self.ncout / self.ncin), 1, 1) / np.sqrt(int(self.ncout / self.ncin))
        elif self.mode == "zero":
            bs, _, size1, size2 = x.shape
            out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
            out[:, :self.ncin] = x
            return out


class PoolingLinear(nn.Module):

    def __init__(self, ncin, ncout, agg="mean"):
        super(PoolingLinear, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.agg = agg

    def forward(self, x):
        if self.agg == "trunc":
            return x[:, :self.ncout]
        k = 1. * self.ncin / self.ncout
        out = x[:, :self.ncout * int(k)]
        out = out.view(x.shape[0], self.ncout, -1)
        if self.agg == "mean":
            out = np.sqrt(k) * out.mean(axis=2)
        elif self.agg == "max":
            out, _ = out.max(axis=2)
        return out


class LinearNormalized(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        self.Q = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, self.Q, self.bias)
