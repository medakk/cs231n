import numpy as np

from .layers import *
from .fast_layers import *

def affine_relu_forward(x, w, b):
    fout, fcache = affine_forward(x, w, b)
    relu_out, rcache = relu_forward(fout)
    cache = (fcache, rcache)

    return relu_out, cache

def affine_relu_backward(dout, cache):
    fcache, rcache = cache
    da = relu_backward(dout, rcache)
    dx, dw, db = affine_backward(da, fcache)

    return dx, dw, db

def affine_relu_batchnorm_forward(x, w, b, gamma, beta, bn_param):
    affine_relu_out, affine_relu_cache = affine_relu_forward(x, w, b)
    batchnorm_out, batchnorm_cache = batchnorm_forward(affine_relu_out, gamma, beta, bn_param)
    cache = (affine_relu_cache, batchnorm_cache)

    return batchnorm_out, cache

def affine_relu_batchnorm_backward(dout, cache):
    affine_relu_cache, batchnorm_cache = cache
    dx, dgamma, dbeta = batchnorm_backward(dout, batchnorm_cache)
    dx, dw, db = affine_relu_backward(dx, affine_relu_cache)

    return dx, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

