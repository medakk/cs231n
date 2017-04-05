import numpy as np

from .layers import *
from .fast_layers import *
from .layer_utils import *
from .solver import Solver

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, spatial_batch_norm=False, dropout=0.0,
               dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.spatial_batch_norm = spatial_batch_norm
        self.use_dropout = dropout > 0.0

        conv_stride = 1
        pad = (filter_size - 1) // 2
        pool_height = 2
        pool_width = 2
        pool_stride = 2
        self.conv_param = {'stride': conv_stride, 'pad': pad}
        self.pool_param = {'pool_height': pool_height, 'pool_width': pool_width, 'stride': pool_stride}

        C, H, W = input_dim
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)

        if spatial_batch_norm:
            self.spatialbn_params = {'eps': 1e-5}
            self.params['spatial_gamma'] = np.random.randn(num_filters)
            self.params['spatial_beta'] = np.random.randn(num_filters)

        W2_input_dim_h = (H - filter_size + 2 * pad) // conv_stride + 1
        W2_input_dim_h = (W2_input_dim_h - pool_height) // pool_stride + 1
        W2_input_dim_w = (W - filter_size + 2 * pad) // conv_stride + 1
        W2_input_dim_w = (W2_input_dim_w - pool_width) // pool_stride + 1
        W2_input_dim = num_filters * W2_input_dim_h * W2_input_dim_w
        self.params['W2'] = np.random.randn(W2_input_dim, hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)

        if self.use_dropout:
            self.dropout_params = {'p': dropout}

        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
     

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        N, C, H, W = X.shape

        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        layers_cache = []

        scores, cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
        layers_cache.append(cache)

        if self.spatial_batch_norm:
            self.spatialbn_params['mode'] = mode
            gamma, beta = self.params['spatial_gamma'], self.params['spatial_beta']
            scores, cache = spatial_batchnorm_forward(scores, gamma, beta, self.spatialbn_params)
            layers_cache.append(cache)

        scores, cache = affine_relu_forward(scores, W2, b2)
        layers_cache.append(cache)

        if self.use_dropout:
            self.dropout_params['mode'] = mode
            scores, cache = dropout_forward(scores, self.dropout_params)
            layers_cache.append(cache)

        scores, cache = affine_forward(scores, W3, b3)
        layers_cache.append(cache)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)

        dx, dW3, db3 = affine_backward(dout, layers_cache.pop())
        grads['W3'] = dW3
        grads['b3'] = db3

        if self.use_dropout:
            dx = dropout_backward(dx, layers_cache.pop())

        dx, dW2, db2 = affine_relu_backward(dx, layers_cache.pop())
        grads['W2'] = dW2
        grads['b2'] = db2

        if self.spatial_batch_norm:
            dx, dgamma, dbeta = spatial_batchnorm_backward(dx, layers_cache.pop())
            grads['spatial_gamma'] = dgamma
            grads['spatial_beta'] = dbeta

        dx, dW1, db1 = conv_relu_pool_backward(dx, layers_cache.pop())
        grads['W1'] = dW1
        grads['b1'] = db1

        loss += 0.5 * self.reg * (W3 ** 2).sum()
        loss += 0.5 * self.reg * (W2 ** 2).sum()
        loss += 0.5 * self.reg * (W1 ** 2).sum()
        grads['W3'] += self.reg * W3
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def hyperparameter_search(num_filters, filter_size, hidden_dim,
                          weight_scale=1e-3, spatial_batch_norm=False,
                          dropout=0, lr_bound=(-6, 3), reg_bound=(-5, -1), 
                          data=None, random_search=True):
    if data is None:
        from ..cifar import cifar_3d_data
        data = cifar_3d_data()

    # Take a subset of the data for training
    data = {'X_train': data['X_train'][:3000],
            'y_train': data['y_train'][:3000],
            'X_val':   data['X_val'],
            'y_val':   data['y_val']}

    if random_search:
        print('Random search...')
        for i in range(15):
            learning_rate = 10 ** np.random.uniform(*lr_bound)
            reg = 10 ** np.random.uniform(*reg_bound)

            print('{}: learning_rate={:.3e}, reg={:.3e}'.format(i+1, learning_rate, reg))
            model = ThreeLayerConvNet(num_filters=num_filters, filter_size=filter_size, hidden_dim=hidden_dim, dropout=dropout,
                                      reg=reg, weight_scale=weight_scale, dtype=np.float64, spatial_batch_norm=spatial_batch_norm)
            solver = Solver(model, data, optim_config={'learning_rate': learning_rate}, lr_decay=1.0,
                            batch_size=200, update_rule='adam', num_epochs=2, verbose=False)
            solver.train()

            loss, train_acc, val_acc = solver.loss_history[-1], solver.train_acc_history[-1], solver.val_acc_history[-1]
            print('loss={:.3f}, train_acc={:.3f}, val_acc={:.3f}\n'.format(loss, train_acc, val_acc))
    else:
        i = 0
        print('Grid search...')
        for learning_rate in [10**x for x in np.arange(*lr_bound, (max(lr_bound) - min(lr_bound)) / 7.0)]:
            for reg in [10**x for x in np.arange(*reg_bound, max(reg_bound) - min(reg_bound) / 7.0)]:
                print('{}: learning_rate={:.3e}, reg={:.3e}'.format(i+1, learning_rate, reg))
                model = ThreeLayerConvNet(num_filters=num_filters, filter_size=filter_size, hidden_dim=hidden_dim, dropout=dropout,
                                          reg=reg, weight_scale=weight_scale, dtype=np.float64, spatial_batch_norm=spatial_batch_norm)
                solver = Solver(model, data, optim_config={'learning_rate': learning_rate}, lr_decay=1.0,
                                batch_size=200, update_rule='adam', num_epochs=2, verbose=False)
                solver.train()

                loss, train_acc, val_acc = solver.loss_history[-1], solver.train_acc_history[-1], solver.val_acc_history[-1]
                print('loss={:.3f}, train_acc={:.3f}, val_acc={:.3f}\n'.format(loss, train_acc, val_acc))
                i += 1

def model1(data):
    """
    Achieves 67.1% validation accuracy
    """
    model = ThreeLayerConvNet(num_filters=28,
                              filter_size=3,
                              hidden_dim=200,
                              weight_scale=1e-3,
                              reg=1.037e-05,
                              dtype=np.float64, spatial_batch_norm=True,
                              dropout=0.0)

    print('Training step 1/3')
    solver = Solver(model, data,
                    optim_config={'learning_rate': 7.489e-04},
                    lr_decay=1.0,
                    batch_size=200,
                    update_rule='adam',
                    num_epochs=10)
    solver.train()


    print('Training step 2/3')
    solver = Solver(model, data,
                    optim_config={'learning_rate': 7.489e-05},
                    lr_decay=1.0,
                    batch_size=200,
                    update_rule='sgd',
                    num_epochs=10)
    solver.train()

    print('Training step 3/3')
    solver = Solver(model, data,
                    optim_config={'learning_rate': 7.489e-06},
                    lr_decay=1.0,
                    batch_size=200,
                    update_rule='sgd',
                    num_epochs=10)
    solver.train()

    return model
