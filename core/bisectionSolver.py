import torch
import torch.nn as nn
import numpy as np


def tanhBisection(x0, numberOfIterations=100):
    if x0 == 0:
        return 0, 1
    upperBound = x0
    lowerBound = -upperBound
    assert lowerBound <= upperBound
    g = np.tanh
    gPrime = lambda x: 1 - g(x) ** 2
    functionToSolve = lambda x: gPrime(x) - (g(x) - g(x0)) / (x - x0)

    for i in range(numberOfIterations):
        xk = (lowerBound + upperBound) / 2
        if functionToSolve(xk) > 0:
            upperBound = xk
        else:
            lowerBound = xk

        if np.abs(upperBound - lowerBound) < 1e-5:
            break
    return upperBound, gPrime(upperBound)


tanhPrimeInflectionPoint = 0.5 * np.log(2 - np.sqrt(3))


def tanhPrimeBisection(x0, numberOfIterations=100):
    if abs(x0 - tanhPrimeInflectionPoint) < 1e-5:
        return x0, 4 / np.sqrt(27)
    assert x0 >= 0
    x0 = -x0
    if x0 > tanhPrimeInflectionPoint:
        lowerBound = -1.1
        upperBound = x0
        sign = 1
    else:
        upperBound = 0
        lowerBound = x0
        sign = -1

    assert lowerBound <= upperBound
    g = np.tanh
    gPrime = lambda x: 1 - g(x) ** 2
    gZegond = lambda x: -2 * g(x) * gPrime(x)
    functionToSolve = lambda x: gZegond(x) - (gPrime(x) - gPrime(x0)) / (x - x0)

    for i in range(numberOfIterations):
        xk = (lowerBound + upperBound) / 2
        if functionToSolve(xk) * sign > 0:
            upperBound = xk
        else:
            lowerBound = xk

        if np.abs(upperBound - lowerBound) < 1e-5:
            break
    return upperBound, gZegond(upperBound)


def createSlopeDictionary():
    maximumValue = 20
    slopeDictionary = {}
    curvDictionary = {}
    rangeOfValues = np.linspace(0, maximumValue, 2001)
    for x in rangeOfValues:
        _, slope = tanhBisection(x)
        slopeDictionary[round(x, 2)] = slope
        _, curv = tanhPrimeBisection(x)
        curvDictionary[round(x, 2)] = curv
    return slopeDictionary, curvDictionary, round(maximumValue, 2)
