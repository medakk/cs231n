from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from ..cifar import *
from .linear import *

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)

        correct_class_score = scores[y[i]]
        numerator = np.exp(correct_class_score)
        denominator = np.sum(np.exp(scores))

        for j in range(num_classes):
            exp_scores = np.exp(scores[j])
            dW[:, j] += (exp_scores / denominator) * X[i]

        dW[:, y[i]] -= X[i]
        loss += -np.log(numerator / denominator)

    loss = loss / num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += (reg * W)

    return dW, loss
    
def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    exp_scores = np.exp(scores)
    correct_class_scores = exp_scores[np.arange(num_train), y]
    exp_sums = exp_scores.sum(axis=1)

    loss = -np.log(correct_class_scores / exp_sums)
    loss = loss.sum() / num_train

    c = exp_scores / exp_sums.reshape((-1, 1))
    c[np.arange(num_train), y] -= 1
    dW += X.T.dot(c)

    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += (reg * W)

    return dW, loss

class SoftmaxClassifier(LinearClassifier):
    def loss(self, W, X, y, reg):
        return softmax_loss_vectorized(W, X, y, reg)

if __name__ == '__main__':
    test(SoftmaxClassifier, 0.006, 0.05)
