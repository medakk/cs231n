from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from ..cifar import *

def numerical_gradient(f, x, h=0.00001):
    fx = f(x)
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + h
        fxph = f(x)
        x[ix] = old_val - h
        fxmh = f(x)
        x[ix] = old_val

        grad[ix] = (fxph - fxmh) / (2 * h)
        it.iternext()
    return grad

class LinearClassifier(object):
    
    def __init__(self):
        pass

    def fit(self, X_train, y_train, learning_rate=0.001, num_iters=100,
                  batch_size=200, reg=0.0, verbose=False):
        if not hasattr(self, 'W'):
            num_classes = np.max(y_train)+1
            self.W = np.random.randn(X_train.shape[1], num_classes) * 0.0001

        for i in range(num_iters):
            choices = np.random.choice(X_train.shape[0], batch_size, replace=True)
            X = X_train[choices]
            y = y_train[choices]

            dW, loss = self.loss(self.W, X, y, reg)
            if verbose:
                print('Iteration {}, loss={}'.format(i, loss))

            self.W -= learning_rate*dW

    def predict(self, X):
        y_pred = X.dot(self.W)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

class LinearSVM(LinearClassifier):
    def loss(self, W, X, y, reg):
        return svm_loss_vectorized(W, X, y, reg)

class SoftmaxClassifier(LinearClassifier):
    def loss(self, W, X, y, reg):
        return softmax_loss_vectorized(W, X, y, reg)

def grid_search(learning_rates, regularization_strengths, classifier, X_train, y_train, X_val, y_val, num_iters=300, batch_size=200):
    results = {}
    best_accuracy, best_clf = -1, None
    for lr, reg in product(learning_rates, regularization_strengths):
        clf = classifier()
        print('lr: {}, reg: {}'.format(lr, reg))
        clf.fit(X_train, y_train, learning_rate=lr, reg=reg, num_iters=num_iters, batch_size=batch_size, verbose=True)
        validation_accuracy = (clf.predict(X_val) == y_val).mean()
        results[(lr, reg)] = validation_accuracy
        
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_clf = clf
    return results, best_clf, best_accuracy

def test(classifier, learning_rate, reg):
    X_train, y_train = load_batches(count=5, vectorize=False, normalize=True)
    X_test, y_test = load_test_batch(vectorize=False, normalize=True)

    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    mean_img = X_train.mean(axis=0)

    # Validation set
    X_val = X_train[num_training:num_training + num_validation]
    X_val = X_val - mean_img
    y_val = y_train[num_training:num_training + num_validation]

    # Train set
    X_train = X_train[:num_training]
    X_train = X_train - mean_img
    y_train = y_train[:num_training]

    # Test set
    X_test = X_test[:num_test]
    X_test = X_test - mean_img
    y_test = y_test[:num_test]

    # Dev set
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    X_dev = X_dev - mean_img
    y_dev = y_train[mask]

    # Add an extra column for the bias
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    print('X_train.shape: {}'.format(X_train.shape))
    print('X_test.shape: {}'.format(X_test.shape))
    print('X_val.shape: {}'.format(X_val.shape))
    print('X_dev.shape: {}'.format(X_dev.shape))

    clf = classifier()
    clf.fit(X_train, y_train, learning_rate=learning_rate, reg=reg, batch_size=500, num_iters=400, verbose=True)

    accuracy = (clf.predict(X_test) == y_test).mean()
    print('Test Accuracy: {}'.format(accuracy))

    W = clf.W[:-1, :]
    classes = labels()

    for i in range(10):
        w = W[:, i]
        w = w - w.min()
        w = w / w.max()
        w = (w * 255.0).astype('uint8')
        
        w = w.reshape((3, 32, 32))
        w = w.swapaxes(0, 2)
        w = w.swapaxes(0, 1)

        plt.subplot(2, 5, i+1)
        plt.imshow(w)
        plt.axis('off')
        plt.title(classes[i])

    plt.show()
