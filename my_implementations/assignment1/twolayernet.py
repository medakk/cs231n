import numpy as np
import matplotlib.pyplot as plt

from ..cifar import *
from .linear import numerical_gradient

DISABLE_TRAIN_ACCURACY = True
PERFORM_PCA = False

class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, dropout=0.0, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.dropout = dropout

    def __repr__(self):
        return str(self)

    def __str__(self):
        input_size = self.params['W1'].shape[0]
        hidden_size = self.params['b1'].shape[0]
        output_size = self.params['W2'].shape[1]
        return '<TwoLayerNet {} - {} - {} | dropout={}>'.format(input_size, hidden_size, output_size, self.dropout)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 =  self.params['W1'], self.params['b1']
        W2, b2 =  self.params['W2'], self.params['b2']
        N, D = X.shape

        scores1 = X.dot(W1) + b1
        scores1 = np.maximum(scores1, 0)

        if self.dropout != 0.0:
            dropout_mask = np.random.rand(*scores1.shape) > self.dropout
            scores1 *= dropout_mask

        scores2 = scores1.dot(W2) + b2
        if y is None:
            return scores2

        # subtract scores2.max() for better stabilitiy. exp will skyrocket for large scores
        exp_scores = np.exp(scores2 - scores2.max())
        exp_sums = np.sum(exp_scores, axis=1)
        correct_class_scores = exp_scores[np.arange(N), y]
        loss = -np.log((correct_class_scores / exp_sums)).sum() / N

        loss = loss + reg * 0.5 * ((W1 ** 2).sum() + (W2 ** 2).sum())

        grad = {}
        c = exp_scores / exp_sums.reshape((-1, 1))
        c[np.arange(N), y] -= 1
        grad['W2'] = scores1.T.dot(c) / N
        grad['b2'] = np.sum(c, axis=0) / N

        d_scores1 = c.dot(W2.T) / N
        d_thresh = d_scores1 * (scores1 != 0)
        grad['W1'] = X.T.dot(d_thresh)
        grad['b1'] = d_thresh.sum(axis=0)

        grad['W2'] += reg * W2
        grad['W1'] += reg * W1

        return loss, grad

    def fit(self, X_train, y_train, X_val=None, y_val=None,
                    learning_rate=1e-4, reg=5e-2, learning_rate_decay=0.95,
                    num_iters=150, batch_size=200, verbose=False):
        
        N_train = X_train.shape[0]

        loss_history = []
        train_accuracy_history = []

        if X_val is not None:
            N_val = X_val.shape[0]
            val_accuracy_history = []

        for i in range(num_iters):
            mask = np.random.choice(N_train, batch_size, replace=True)
            X = X_train[mask]
            y = y_train[mask]

            loss, grad = self.loss(X, y, reg)
            for param in self.params:
                self.params[param] -= learning_rate * grad[param]

            if DISABLE_TRAIN_ACCURACY:
                train_accuracy = 0
            else:
                train_accuracy = (self.predict(X_train) == y_train).mean()

            loss_history.append(loss)
            train_accuracy_history.append(train_accuracy)

            if X_val is not None:
                val_accuracy = (self.predict(X_val) == y_val).mean()
                val_accuracy_history.append(val_accuracy)

            if verbose:
                if X_val is not None:
                    print('Iteration {}: loss={} train_accuracy={:.3f} val_accuracy={:.3f}'.format(
                            i, loss, train_accuracy, val_accuracy))
                else:
                    print('Iteration {}: loss={} train_accuracy={:.3f}'.format(
                            i, loss, train_accuracy))

            learning_rate = learning_rate_decay * learning_rate

        result = {
            'loss_history': loss_history,
            'train_accuracy_history': train_accuracy_history,
        }
        if X_val is not None:
            result['val_accuracy_history'] = val_accuracy_history
        return result

    def predict(self, X):
        W1, b1 =  self.params['W1'], self.params['b1']
        W2, b2 =  self.params['W2'], self.params['b2']

        scores1 = np.maximum(X.dot(W1) + b1, 0) * (1-self.dropout)
        scores2 = (scores1.dot(W2) + b2)
        y_pred = scores2.argmax(axis=1)
        return y_pred

def test():
    X_train, y_train = load_batches(count=5, vectorize=False, normalize=False)
    X_test, y_test = load_test_batch(vectorize=False, normalize=False)

    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500
    features = 32 * 32 * 3 

    mean_img = X_train.mean(axis=0)
    X_train = X_train - mean_img
    X_test = X_test - mean_img

    if PERFORM_PCA:
        print('Finding covariance...')
        cov = np.dot(X_train.T, X_train) / X_train.shape[0]
        print('Performing SVD...')
        U, S, V = np.linalg.svd(cov)
        print('Finished SVD!')
        features = 500
        X_train = np.dot(X_train, U[:, :features])
        X_test = np.dot(X_test, U[:, :features])

    # Validation set
    X_val = X_train[num_training:num_training + num_validation]
    y_val = y_train[num_training:num_training + num_validation]

    # Train set
    X_train = X_train[:num_training]
    y_train = y_train[:num_training]

    # Test set
    X_test = X_test[:num_test]
    y_test = y_test[:num_test]

    # Dev set
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    print('X_train.shape: {}'.format(X_train.shape))
    print('X_test.shape: {}'.format(X_test.shape))
    print('X_val.shape: {}'.format(X_val.shape))
    print('X_dev.shape: {}'.format(X_dev.shape))

    clf = TwoLayerNet(features, 50, 10, dropout=0.10)

    learning_rate = 0.0005756259848890687
    reg = 5.595546771648509e-05
    clf.fit(X_train, y_train, X_val, y_val, learning_rate=learning_rate, reg=reg, learning_rate_decay=1,
            batch_size=500, num_iters=1000, verbose=True)

    learning_rate = 0.0005756259848890687/10
    reg = 5.595546771648509e-04
    clf.fit(X_train, y_train, X_val, y_val, learning_rate=learning_rate, reg=reg, learning_rate_decay=1,
            batch_size=500, num_iters=500, verbose=True)

    accuracy = (clf.predict(X_test) == y_test).mean()
    print('Test Accuracy: {}'.format(accuracy))

if __name__ == '__main__':
    test()
