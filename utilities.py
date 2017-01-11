import math

import numpy as np

def read_data(f):
    xs = []
    ys = []
    for line in f:
        *x_str,y_str = line.split(',')
        xs.append(np.array(list(map(float,x_str))))
        ys.append(float(y_str))
    return (np.array(xs), np.array(ys))

def classification_accuracy(model, X, y):
    N = len(y)
    count = 0
    for x, label in zip(X, y):
        real = label > 0
        if model(x) == real:
            count += 1
    return count / N

def tune(X, y, algs, k=5):
    """ tuning using k-fold cross-validation
    """
    avg_CA = np.zeros(len(algs))

    N, d = X.shape

    idx = np.random.permutation(N)

    num_iterations = k * len(algs)
    cnt_iterations = 1

    step = math.ceil(N/k)
    for n in range(0, N, step):
        sel = np.full((N,),  False, dtype=bool)
        sel[n:n+step] = True

        train_X = X[np.invert(sel),:]
        train_y = y[np.invert(sel)]
        val_X = X[sel,:]
        val_y = y[sel]

        for i, alg in enumerate(algs):
            print("tune {}/{}".format(cnt_iterations, num_iterations))
            cnt_iterations += 1
            try:
                model = alg(train_X, train_y)
            except:
                continue
            avg_CA[i] += classification_accuracy(model, val_X, val_y)
    avg_CA /= k

    best_alg = algs[ np.argmax(avg_CA) ]

    return best_alg
