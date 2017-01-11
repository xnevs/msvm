#!/usr/bin/python3

import sys
import math
import time

import numpy as np
np.set_printoptions(precision=2, linewidth=1000)

import msvm
from msvm_kernels import *

from utilities import *


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        X, y = read_data(f)

    N, d = X.shape

    train_N = int(0.8 * N)

    sel_idx = np.random.choice(np.arange(N), train_N, replace=False)
    selection = np.full((N,), False, dtype=bool)
    selection[sel_idx] = True

    train_X = X[selection,:]
    train_y = y[selection]

    test_X = X[np.invert(selection),:]
    test_y = y[np.invert(selection)]


    kernels = [linear] + \
              [rbf(0.01), rbf(0.1), rbf(1), rbf(10)] + \
              [polynomial(0.5, 0, 2), polynomial(0.5, 0.5, 2), polynomial(0.5, 1, 2), polynomial(0.5, 0, 3)] + \
              [sigmoid(1, 0), sigmoid(1, 1), sigmoid(2, 0)]

    Cs = [0.01, 0.1, 1, 10, 100]

    kernels_and_Cs = [(kernel, C) for C in Cs for kernel in kernels]

    print("Average kernel", file=sys.stderr)
    print("==============", file=sys.stderr)

    average_kernel = msvm.Combined_kernel(kernels)
    algs = list(map(lambda C: lambda XX, yy: msvm.learn(XX, yy, C, average_kernel), Cs))

    start_time = time.process_time()
    alg = tune(train_X, train_y, algs)
    try:
        model = alg(train_X, train_y)

        end_time = time.process_time()
        elapsed = end_time - start_time

        CA_train = classification_accuracy(model, train_X, train_y)
        CA_test  = classification_accuracy(model, test_X, test_y)
        print("avg", model.C, CA_train, CA_test, elapsed, sep=',')
    except Exception as e:
        print("failed:", e)

    print()
    print()

    print("Cross validation", file=sys.stderr)
    print("================", file=sys.stderr)

    algs = list(map(lambda kernel_C: lambda XX, yy: msvm.learn(XX, yy, kernel_C[1], kernel_C[0]), kernels_and_Cs ))

    start_time = time.process_time()
    alg = tune(train_X, train_y, algs)
    try:
        model = alg(train_X, train_y)

        end_time = time.process_time()
        elapsed = end_time - start_time

        CA_train = classification_accuracy(model, train_X, train_y)
        CA_test  = classification_accuracy(model, test_X, test_y)
        print("cross", model.C, CA_train, CA_test, elapsed, sep=',')
    except Exception as e:
        print("failed:", e)

    print()
    print()

    print("Multiple kernel learning", file=sys.stderr)
    print("========================", file=sys.stderr)

    algs = list(map(lambda C: lambda XX, yy: msvm.multi_learn(XX, yy, C, kernels), Cs))
    start_time = time.process_time()
    alg = tune(train_X, train_y, algs)
    try:
        model = alg(train_X, train_y)

        end_time = time.process_time()
        elapsed = end_time - start_time

        CA_train = classification_accuracy(model, train_X, train_y)
        CA_test  = classification_accuracy(model, test_X, test_y)
        print("mkl", model.C, CA_train, CA_test, elapsed, '"{}"'.format(model.kernel.beta), sep=',')
    except Exception as e:
        print("failed: ", e)
