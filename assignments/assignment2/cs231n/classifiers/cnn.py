import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 spatial_batch_norm=False, dropout=0.0,
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

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

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


pass
