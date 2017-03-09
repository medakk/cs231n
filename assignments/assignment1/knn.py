# Build a classifier using knn(K Nearest Neighbours)

import numpy as np
import matplotlib.pyplot as plt

from cifar import *

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

        for i in range(n_samples):
            l1_dist = np.sum(np.abs(self.X_train - X_test[i, :]), axis=1)

            # This line gets the indices of the k smallest values in l1_dist
            # see https://stackoverflow.com/a/23734295
            top_k = np.argpartition(l1_dist, self.k)[:self.k]

            # Find the classes of the closest matches
            top_k = [self.y_train[idx] for idx in top_k]

            # Find most frequent class label
            y_pred[i] = np.argmax(np.bincount(top_k))
        return y_pred
