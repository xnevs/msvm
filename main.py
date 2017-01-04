#!/usr/bin/python3

import sys
import math

import numpy as np

import msvm
from msvm_kernels import *

def read_data(f):
    xs = []
    ys = []
    for line in f:
        *x_str,y_str = line.split(',')
        xs.append(np.array(list(map(float,x_str))))
        ys.append(float(y_str))
    return (np.array(xs), np.array(ys))
    
if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        X, y = read_data(f)

    N, d = X.shape

    train_N = int(0.8 * N)

    sel_idx = np.random.choice(np.arange(N), train_N, replace=False)

    selection = np.array(N*[False])
    for idx in sel_idx:
        selection[idx] = True


    train_X = X[selection,:]
    train_y = y[selection]

    test_X = X[np.invert(selection),:]
    test_y = y[np.invert(selection)]

    kernels = [linear] + \
              [rbf(1), rbf(5)] + \
              [polynomial(1, 0, 2), polynomial(2, 0, 2), polynomial(1, 0, 3), polynomial(2, 5, 2)] + \
              [sigmoid(1, 0), sigmoid(2, 0), sigmoid(2, 2)]

    print("Average kernel")
    print("==============")

    combined_kernel = msvm.Combined_kernel(kernels)
    model = msvm.learn(train_X, train_y, 100, combined_kernel)

    count = 0
    for x, label in zip(test_X, test_y):
        real = label > 0
        if model(x) == real:
            count += 1
    print("CA: ", count/test_y.size)


    print()
    print()

    print("Multiple kernel learning")
    print("========================")

    model = msvm.multi_learn(train_X, train_y, 100, kernels)

    if hasattr(model.kernel, 'beta'):
        print("beta: ", model.kernel.beta)

    count = 0
    for x, label in zip(test_X, test_y):
        real = label > 0
        if model(x) == real:
            count += 1
    print("CA: ", count/test_y.size)
