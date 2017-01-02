import math

import numpy as np

def linear(u, v):
    return np.dot(u,v)

def polynomial(gamma, c, d):
    def kernel(u, v):
        return (gamma * np.dot(u, v) + c)**d
    return kernel

def rbf(gamma):
    def kernel(u, v):
        v = u - v
        return math.exp( -gamma * np.dot(v, v))
    return kernel

def sigmoid(gamma, c):
    def kernel(u, v):
        return math.tanh(gamma * np.dot(u, v) + c)
    return kernel
