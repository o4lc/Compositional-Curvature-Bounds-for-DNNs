import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.bisectionSolver import createSlopeDictionary


def normalize(input):
    norm = torch.linalg.norm(input.flatten(1), 2, 1, True)
    while len(norm.shape) < len(input.shape):
        norm = norm.unsqueeze(-1)
    return input / norm


class SlopeDictionary:

    def __init__(self, activationFunction):
        if "softplus" in activationFunction.lower():
            activationFunction = "softplus"
        self.betaDictionary, self.betaPrimeDictionary, maximumValue = createSlopeDictionary(activationFunction)
        self.maximumValue = torch.tensor(maximumValue)


    @staticmethod
    def readFromDictionary(dictionary, points, maximumValue):

        xx = [round(y, 2) for y in torch.minimum(torch.abs(points), maximumValue).flatten().tolist()]
        slopes = []
        for anchorPoint in xx:
            slopes.append(dictionary[anchorPoint])
        # print(anchorPoint, max(slopes))
        results = torch.tensor(slopes).reshape(points.shape).to(points.device).to(points.dtype)
        assert not torch.any(results < dictionary[round(maximumValue.item(), 2)])
        return results

    def getBeta(self, x):
        return SlopeDictionary.readFromDictionary(self.betaDictionary, x, self.maximumValue)


    def getBetaPrime(self, x):
        return SlopeDictionary.readFromDictionary(self.betaPrimeDictionary, x, self.maximumValue)

    @staticmethod
    def multiplyDictionary(dictionary, multiplier):
        for key in dictionary.keys():
            dictionary[key] *= multiplier


class SDPBasedLipschitzConvLayer(nn.Module):

    def __init__(self, config, input_size, cin, cout, kernel_size=3, epsilon=1e-6, activation='relu'):
        super(SDPBasedLipschitzConvLayer, self).__init__()

        self.slopeDictionary = SlopeDictionary(activation)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
            self.maxSlope = 1
            self.maxCurv = None
        elif activation == 'tanh':
            self.activation = nn.Tanh()
            self.maxSlope = 1
            self.maxCurv = 4 / np.sqrt(27)
        elif 'softplus' in activation.lower():
            betaValue = 1
            self.maxSlope = 1
            self.maxCurv = 0.25 * betaValue
            self.slopeDictionary.multiplyDictionary(self.slopeDictionary.betaPrimeDictionary, betaValue)
            self.activation = lambda x: nn.functional.softplus(x, beta=betaValue)
            if activation == 'centered_softplus':
                self.activation = lambda x: self.activation(x) - torch.log(torch.tensor(2.))/betaValue
        else:
            raise NotImplementedError


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
        neg2TInv = -2 / T
        res = neg2TInv * res
        res = F.conv_transpose2d(res, self.kernel, padding=1)
        out = x + res

        with torch.no_grad():
            self.powerIterate(neg2TInv)

        return out

    def powerIterate(self, neg2TInv):
        self.wEigen.data = \
            normalize(F.conv_transpose2d(F.conv2d(self.wEigen, self.kernel, padding=1), self.kernel, padding=1))
        self.gEigen.data = \
            normalize(neg2TInv * F.conv2d(F.conv_transpose2d(
                neg2TInv * self.gEigen, self.kernel, padding=1), self.kernel, padding=1))


    def calculateElementLipschitzs(self, localPoints=None):
        T = self.computeT()[None, :, None, None]

        wForward = F.conv2d(self.wEigen, self.kernel, padding=1)
        wNorm = torch.linalg.norm(wForward.flatten(), 2)
        if localPoints is None:
            activationWNorm2Inf =\
                self.maxCurv * torch.max(torch.linalg.norm(self.kernel.flatten(1), 2, 1))
        else:
            assert len(localPoints.shape) == 4
            wPassed = F.conv2d(localPoints, self.kernel, bias=self.bias, padding=1)
            betaPrimes = self.slopeDictionary.getBetaPrime(wPassed)
            betaPrimes = torch.max(betaPrimes.flatten(2), 2).values
            # print(torch.max(betaPrimes, 1).values)
            # raise
            # raise
            wRowNorm = torch.linalg.norm(self.kernel.flatten(1), 2, 1)
            activationWNorm2Inf = torch.max(wRowNorm.unsqueeze(0) * betaPrimes, 1).values.unsqueeze(1)

        # wNorm2Inf = torch.max(torch.linalg.norm(self.kernel.flatten(1), 2, 1))

        gForward = F.conv_transpose2d(-2 / T * self.gEigen, self.kernel, padding=1)
        gNorm = torch.linalg.norm(gForward.flatten(), 2)
        return wNorm, gNorm, activationWNorm2Inf

    def calculateCurvature(self, localPoints=None):
        wNorm, gNorm, activationWNorm2Inf = self.calculateElementLipschitzs(localPoints=localPoints)
        layerJacobianLipschitz = wNorm * gNorm * activationWNorm2Inf
        return layerJacobianLipschitz


class SDPBasedLipschitzLinearLayer(nn.Module):

    def __init__(self, config, cin, cout, epsilon=1e-6, activation='relu'):
        super(SDPBasedLipschitzLinearLayer, self).__init__()

        self.slopeDictionary = SlopeDictionary(activation)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
            self.maxSlope = 1
            self.maxCurv = None
        elif activation == 'tanh':
            self.activation = nn.Tanh()
            self.maxSlope = 1
            self.maxCurv = 4 / np.sqrt(27)
        elif 'softplus' in activation.lower():
            betaValue = 1
            self.maxSlope = 1
            self.maxCurv = 0.25 * betaValue
            self.slopeDictionary.multiplyDictionary(self.slopeDictionary.betaPrimeDictionary, betaValue)
            self.activation = lambda x: nn.functional.softplus(x, beta=betaValue)
            if activation == 'centered_softplus':
                self.activation = lambda x: self.activation(x) - torch.log(torch.tensor(2.)) / betaValue
        else:
            raise NotImplementedError
        
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
        self.aEigen = nn.Parameter(normalize(torch.randn((1, cout))), requires_grad=False)
        # self.aEigen = normalize(torch.randn((1, cout))).to(torch.device("cuda:0"))

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
        neg2TInv = -2 / T
        res = neg2TInv * res
        res = F.linear(res, self.weights.t())
        out = x + res

        # updating eigenVectors
        with torch.no_grad():
            self.powerIterate(neg2TInv)
        return out

    def powerIterate(self, neg2TInv):
        self.wEigen.data = normalize(self.wEigen @ self.weights.T @ self.weights)
        gForward = (neg2TInv * self.gEigen) @ self.weights
        gBackward = neg2TInv * (gForward @ self.weights.T)
        self.gEigen.data = normalize(gBackward)

        aForward = torch.einsum("bl,l,li,lj->bij",
                                self.aEigen, neg2TInv, self.weights, self.weights)
        aBackward = torch.einsum("bij, ni, nj, n->bn", aForward, self.weights, self.weights, neg2TInv)
        self.aEigen.data = normalize(aBackward)

    def calculateElementLipschitzs(self, localPoints=None):
        T = self.computeT()
        wNorm = torch.linalg.norm((self.wEigen @ self.weights.T).flatten(), 2)
        if localPoints is None:
            activationWNorm2Inf = self.maxCurv * torch.max(torch.linalg.norm(self.weights, 2, 1))
        else:
            assert len(localPoints.shape) == 2
            wPassed = F.linear(localPoints, self.weights, self.bias)
            betaPrimes = self.slopeDictionary.getBetaPrime(wPassed)

            wRowNorm = torch.linalg.norm(self.weights, 2, 1)
            activationWNorm2Inf = torch.max(wRowNorm.unsqueeze(0) * betaPrimes, 1).values.unsqueeze(1)


        gForward = (-2 / T * self.gEigen) @ self.weights
        gNorm = torch.linalg.norm(gForward.flatten(), 2)
        return wNorm, gNorm, activationWNorm2Inf

    def calculateCurvature(self, localPoints=None):
        method = 2

        if method == 1:
            wNorm, gNorm, activationWNorm2Inf = self.calculateElementLipschitzs(localPoints=localPoints)
            layerJacobianLipschitz = wNorm * activationWNorm2Inf * gNorm
        elif method == 2:
            if localPoints is None:
                wForward = self.wEigen @ self.weights.T
                activationWNorm = self.maxCurv * torch.linalg.norm(wForward.flatten(), 2, )
            else:
                assert len(localPoints.shape) == 2
                wPassed = F.linear(localPoints, self.weights, self.bias)
                betaPrimes = self.slopeDictionary.getBetaPrime(wPassed)

                # perform a few power iterations just to converge to local answer
                eigenVector = self.wEigen.data
                with torch.no_grad():
                    for _ in range(2):
                        wForward = betaPrimes * (eigenVector @ self.weights.T)
                        wBackward = (betaPrimes * wForward) @ self.weights
                        eigenVector = normalize(wBackward)

                wForward = betaPrimes * (eigenVector @ self.weights.T)
                activationWNorm = torch.linalg.norm(wForward.flatten(1), 2, 1, keepdim=True)
            T = self.computeT()
            neg2TInv = -2 / T
            aForward = torch.einsum("bl,l,li,lj->bij",
                         self.aEigen, neg2TInv, self.weights, self.weights)

            aNorm = torch.linalg.norm(aForward.flatten(), 2)
            layerJacobianLipschitz = aNorm * activationWNorm

        # wNorm, gNorm, activationWNorm2Inf = self.calculateElementLipschitzs(localPoints=localPoints)
        # layerJacobianLipschitz2 = wNorm * activationWNorm2Inf * gNorm
        # print(torch.mean(layerJacobianLipschitz), torch.mean(layerJacobianLipschitz2),
        #       torch.mean(layerJacobianLipschitz / layerJacobianLipschitz2))

        return layerJacobianLipschitz



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
