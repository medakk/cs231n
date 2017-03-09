import numpy as np
import matplotlib.pyplot as plt
import pickle
from functools import lru_cache

@lru_cache(maxsize=1)
def labels():
    """returns a list of labels in the data set"""
    unpickled = None
    with open('data/cifar-10-batches-py/batches.meta', 'rb') as fd:
        unpickled = pickle.load(fd)

    return unpickled['label_names']

def load_batch(batch_name):
    """opens the pickled file and loads it"""
    assert type(batch_name)==str

    unpickled = None
    with open('data/cifar-10-batches-py/'+batch_name, 'rb') as fd:
        unpickled = pickle.load(fd, encoding='bytes')

    # convert b-strings to utf-8 for the keys
    batch = {str(k, 'utf-8'):v for k,v in unpickled.items()}
    return batch

def load_batch_xy(batch_name, vectorize=True, normalize=True):
    """
    Only obtain the pixel data and corresponding
    class-labels in numpy-arrays
    The pixels can be normalized to 0.0-1.0 and the
    class-labels can be vectorized
    """
    batch = load_batch(batch_name)

    X = batch['data']
    if normalize:
        X = X/255.0

    Y = np.array(batch['labels'])
    if vectorize:
        Y = vectorize_y(Y)

    return X,Y

def load_batches(count=5, vectorize=True, normalize=True):
    """Loads the first ``count`` batches,
    stacks them vertically and returns a two-tuple
    (X,Y). The Y axis is optionally vectorized"""
    assert count>=1 and count<=5

    X,Y = [], []
    for i in range(1,count+1):
        x,y = load_batch_xy('data_batch_'+str(i), vectorize, normalize)
        X.append(x)
        Y.append(y)
    if vectorize:
        return np.vstack(X), np.vstack(Y)
    else:
        return np.vstack(X), np.hstack(Y)

def load_test_batch(vectorize=True, normalize=True):
    """loads the test batch"""
    return load_batch_xy('test_batch', vectorize, normalize)

def vectorize_y(Y):
    """converts the output from a single class label
    to a probability distribution. Ex:
    if y==4, it will be converted to [0,0,0,0,1,0,0,0,0,0]"""

    N_LABELS = 10
    y_size = Y.shape[0]

    new_Y = np.zeros((y_size, 10))
    for i in range(y_size):
        new_Y[i][Y[i]] = 1
    return new_Y

def show_img(x, title=None):
    """given a (3072,) np.ndarray, display the image using
    matplotlib. ``title`` can be either a string label or an index
    to a label in the list of labels. ``title`` is optional"""

    img = x.reshape((3,32,32))
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)

    plt.imshow(img)
    if title is not None:
        if type(title)==str:
            plt.title(title)
        else:
            plt.title(labels()[title])

    plt.show()

def shared(batch):
    """takes a (X,Y) tuple and outputs a shared theano object"""
    import theano
    shared_X = theano.shared(
        np.asarray(batch[0], dtype=theano.config.floatX),
        borrow=True)
    shared_Y = theano.shared(
        np.asarray(batch[1], dtype=theano.config.floatX),
        borrow=True)
    return shared_X, shared_Y

def decolorize(batch, algorithm='green'):
    """removes the colour data, ie: creates a grayscale image
    algorithm can be:
    *green - replace every pixel with the green value in RGB
    """
    if algorithm=='green':
        return batch[0][:,1024:2048],batch[1]
    else:
        raise Exception('decolorize: no algorithm {} found'.format(algorithm))

def accuracy(expected_Y, Y):
    """returns the fraction of correct outputs"""
    if Y.shape[1]==1:
        eq_count = (expected_Y==Y).sum()
        total = Y.shape[0]
        return eq_count/total
    else:
        raise NotImplementedError('accuracy has not been implemented for vectorized output')
