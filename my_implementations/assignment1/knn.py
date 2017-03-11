# Build a classifier using knn(K Nearest Neighbours)

import numpy as np
import matplotlib.pyplot as plt

from ..cifar import *

class KNN(object):

    def __init__(self):
        pass

    def fit(self, X_train, y_train, k=1):
        """
        X_train is a (n_samples, n_features) matrix.
        y_train is a (n_samples, ) vector where every element is the class label
        k is the number of nearest neighbours to consider
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        y_pred = np.ndarray((n_samples, ))

        dot_product = X_test.dot(self.X_train.T)
        a2 = np.sum(self.X_train**2, axis=1)
        b2 = np.sum(X_test**2, axis=1)
        dists = a2 + b2.reshape((-1, 1)) - 2*dot_product

        # This line gets the indices of the k smallest values in l1_dist
        # see https://stackoverflow.com/a/23734295
        top_k = np.argpartition(dists, self.k, axis=1)[:, :self.k]

        # Get the labels for the training samples that we have picked
        top_k =  self.y_train[top_k]

        # Find the most frequent class label in every row
        f = lambda x : np.argmax(np.bincount(x))
        y_pred = np.apply_along_axis(f, axis=1, arr=top_k)

        return y_pred

def test():
    X_train, y_train, X_test, y_test = load_tiny()

    knn = KNN()
    knn.fit(X_train, y_train, k=5)

    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print('Accuracy: {}'.format(accuracy))

if __name__ == '__main__':
    test()
