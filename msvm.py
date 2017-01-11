import sys

import numpy as np

from cvxopt import matrix, spmatrix

import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False

class svm_model:
    def __init__(self, kernel, alpha, bias, support_vectors, support_labels, objective_value, sv_idx, C):
        self.kernel          = kernel
        self.alpha           = alpha
        self.bias            = bias
        self.support_vectors = support_vectors
        self.support_labels  = support_labels

        self.objective_value = objective_value
        self.sv_idx          = sv_idx
        self.C               = C

    def __call__(self, x):
        result = self.bias
        for n in range(self.alpha.size):
            result += self.alpha[n] * self.support_labels[n] * self.kernel(self.support_vectors[n], x)
        return result > 0

def gram_matrix(kernel, X):
    N, d = X.shape
    gram = np.zeros((N, N))
    for n, x_n in enumerate(X):
        for m, x_m in enumerate(X):
            gram[n,m] = kernel(x_n, x_m)
    return gram


def learn(X, y, C, kernel, gram=None):
    if gram is None:
        gram = gram_matrix(kernel, X)

    N, d = X.shape

    P = matrix( np.outer(y, y) * gram )
    q = matrix(-1.0, (N,1))

    G = spmatrix(N*[-1.0]+N*[1.0], range(2*N), 2*list(range(N)))
    h = matrix(N*[0.0] + N*[C])

    A = matrix(y, (1,N))
    b = matrix(0.0)

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    objective_value = solution['primal objective']
    alpha = np.ravel(solution['x'])

    over_threshold = alpha > 1e-5 

    alpha = alpha[over_threshold]
    support_vectors = X[over_threshold,:]
    support_labels = y[over_threshold]
    sv_idx = np.arange(N)[over_threshold]

    alpha2 = np.array(alpha)
    alpha2[alpha2 > 0.95*C] = 0

    margin_sv_idx = np.argmax(alpha2)
    if alpha2[margin_sv_idx] == 0:
        margin_sv_idx = np.argmin(alpha)

    bias = support_labels[margin_sv_idx]
    for n in range(alpha.size):
        bias -= alpha[n] * support_labels[n] * kernel(support_vectors[n],support_vectors[margin_sv_idx])

    return svm_model(kernel, alpha, bias, support_vectors, support_labels, objective_value, sv_idx, C)


class Combined_kernel:
    def __init__(self, kernels):
        self.beta = (1.0 / len(kernels)) * np.ones(len(kernels))
        self.kernels = kernels
    def __call__(self, a, b):
        return np.dot(self.beta, np.array([kernel(a, b) for kernel in self.kernels]))

def multi_learn(X, y, C, kernels):
    N, d = X.shape
    K = len(kernels)

    kernel = Combined_kernel(kernels)

    grams = np.array([gram_matrix(kernel, X) for kernel in kernels])

    model = learn(X, y, C, kernel, gram=np.tensordot(kernel.beta, grams, axes=(0,0)))

    c = matrix(0.0, (K+1, 1))
    c[K] = -1.0

    A = matrix(1.0, (1, K+1))
    A[K] = 0.0
    b = matrix(1.0)

    constraints = [ k*[0.0]+[-1.0]+(K-k)*[0] for k in range(K) ]

    while True:
        last_constraint = []
        M = np.outer(model.alpha, model.alpha) * \
            np.outer(model.support_labels, model.support_labels)
        alpha_sum = np.sum(model.alpha)
        for k in range(K):
            S_k = 0.5 * np.sum(M * grams[k, model.sv_idx][:,model.sv_idx]) - alpha_sum
            last_constraint.append(-S_k)
        last_constraint.append(1)

        constraints.append(last_constraint)

        G = matrix(constraints).trans()
        h = matrix(0.0, (G.size[0],1))

        solution = cvxopt.solvers.lp(c, G, h, A, b)
        solution_x = solution['x']

        theta = solution_x[-1]
        kernel.beta = np.ravel(solution_x[:-1])

        model = learn(X, y, C, kernel, gram=np.tensordot(kernel.beta, grams, axes=(0,0)))
        S = model.objective_value

        eps = abs(1.0 - S / theta)
        #print("eps: ", eps, file=sys.stderr)
        if  eps < 1e-3:
            break

    return model
