from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from ..cifar import *
from .linear import *

def svm_loss_naive(W, X, y, reg):
    """
    W is a 3072x10 matrix of weights (or 3073*10 if a bias term has been added)
    X is a (num_samples, 3072) matrix of input images
    y is a (num_samples, ) vector of class labels
    reg is the regularization term
    """
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] += (-X[i])

    loss /= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += (reg * W)

    return dW, loss

def svm_loss_vectorized(W, X, y, reg):
    """
    W is a 3072x10 matrix of weights (or 3073*10 if a bias term has been added)
    X is a (num_samples, 3072) matrix of input images
    y is a (num_samples, ) vector of class labels
    reg is the regularization term
    """
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y]
    thresh = np.maximum(scores - correct_class_scores.reshape((-1, 1)) + 1, 0)
    thresh[np.arange(num_train), y] = 0
    loss += thresh.sum() / num_train

    binarized = (thresh != 0).astype('float')
    bin_sum = binarized.sum(axis=1)
    binarized[np.arange(num_train), y] -= bin_sum

    dW += X.T.dot(binarized)
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += (reg * W)

    return dW, loss

class LinearSVM(LinearClassifier):
    def loss(self, W, X, y, reg):
        return svm_loss_vectorized(W, X, y, reg)

if __name__ == '__main__':
    test(LinearSVM, 7e-3, 5)
