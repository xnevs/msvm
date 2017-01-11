# msvm

An implementation of a Multiple Kernel Learning Support Vector Machine learning algorithm, as described in [1].

[1]: Sonnenburg, Sören, et al. "Large scale multiple kernel learning." Journal of Machine Learning Research 7.Jul (2006): 1531-1565. Available online at http://www.jmlr.org/papers/volume7/sonnenburg06a/sonnenburg06a.pdf.

## Dependencies

The algorithms are implemented in Python 3 using NumPy and CVXOPT to solve the optimization problems arising in the algorithms.

## Running

Currently, only binary classification problems are supported. An example of using the `learn` and `multi_learn` functions from `msvm.py` can be found in `main.py`.

`main.py` reads a CSV formatted input data set of the form
```
attr_1,attr2,...,attr_d,label
attr_1,attr2,...,attr_d,label
...
```
where `label` must be either `1` or `-1` and attributes are real-valued.

`main.py` splits the data set into a training and a testing set and builds three classifiers:
  1. a standard SVM classifier using the average kernel function,
  2. an SVM classifier where the kernel function is chosen through cross-validation and
  3. a Multiple Kernel Learning SVM classifier, where the kernel function is learned (as a combination of candidate kernel functions) from the training data as part of the optimization.

The classification accuracy of all three models is reported.
