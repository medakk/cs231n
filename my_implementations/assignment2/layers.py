import numpy as np

def affine_forward(x, w, b):
    """
    reshapes x so that it is compatible with the weight w.
    then return x.dot(w) + b, along with a cache which contains
    the values of x, w, and b
    """
    x_reshaped = x.reshape((x.shape[0], w.shape[0]))
    out = x_reshaped.dot(w) + b

    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    x_reshaped = x.reshape((x.shape[0], w.shape[0]))

    db = dout.sum(axis=0)
    dw = x_reshaped.T.dot(dout)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)

    return dx, dw, db

def relu_forward(x):
    out = np.maximum(x, 0)
    cache = x

    return out, cache

def relu_backward(dx, cache):
    x = cache
    return dx * (x > 0)

def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(x - correct_class_scores.reshape((-1, 1)) + 1, 0)
    margins[np.arange(N), y] = 0
    loss = margins.sum() / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx

def softmax_loss(x, y):
    N = x.shape[0]
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, *D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(shape=D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(shape=D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        mu = x.mean(axis=0)
        x_minus_mu = x - mu
        var = np.mean(x_minus_mu ** 2, axis=0)

        xcap = x_minus_mu / np.sqrt(var + eps)
        out = gamma * xcap + beta

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        
        cache = (gamma, mu, var, eps, x, xcap)
    elif mode == 'test':
        x_minus_mu = x - running_mean
        xcap = x_minus_mu / np.sqrt(running_var + eps)
        out = gamma * xcap + beta

        cache = (gamma, running_mean, running_var, eps, x, xcap)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache

def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    gamma, mu, var, eps, x, xcap = cache

    N, *D = x.shape

    dxcap = dout * gamma
    dvar = np.sum(dxcap * (x - mu) * (-0.5) * (var + eps) ** (-3/2), axis=0)
    dmu = np.sum(dxcap * (-1) / np.sqrt(var + eps), axis=0) + dvar * np.sum((-2) * (x - mu), axis=0) / N

    dx = (dxcap / np.sqrt(var + eps)) + (dvar * 2 * (x - mu) / N) + (dmu / N)
    dgamma = np.sum(dout * xcap, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    dx, dgamma, dbeta = None, None, None
    gamma, mu, var, eps, x, xcap = cache

    N, D = x.shape

    dxcap = dout * gamma
    #dvar = np.sum(dxcap * (x - mu) * (-0.5) * (var + eps) ** (-3/2), axis=0)
    #dmu = np.sum(dxcap * (-1) / np.sqrt(var + eps), axis=0) + dvar * np.sum((-2) * (x - mu), axis=0) / N

    dx = (dxcap + np.sum(dxcap * ((-(xcap ** 2) / N) - (1 / N) + ((x - mu) / (var + eps)) * (x - mu).sum(axis=0) / (N ** 2)), axis=0)) / np.sqrt(var + eps)
    dgamma = np.sum(dout * xcap, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if p == 0.0:
        return x, (dropout_param, None)
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    out_x, mask = None, None

    if mode == 'train':
        mask = (np.random.random(x.shape) < p) / p
        out_x = (mask * x)
    elif mode == 'test':
        out_x = x

    cache = (dropout_param, mask)
    out_x = out_x.astype(x.dtype, copy=False)

    return out_x, cache

def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if dropout_param['p'] == 0.0:
        return dout

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout

    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    stride, pad = conv_param['stride'], conv_param['pad']

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    out_dim_h = (H + 2 * pad - HH) // stride + 1
    out_dim_w = (W + 2 * pad - WW) // stride + 1
    out = np.ndarray((N, F, out_dim_h, out_dim_h))

    for n in range(N):
        for f in range(F):
            for j in range(out_dim_h):
                for i in range(out_dim_w):
                    _y = stride * j
                    _x = stride * i
                    tmp = x_padded[n, :, _y:_y+HH, _x:_x+WW] * w[f] 
                    tmp = tmp.sum()
                    tmp += b[f]
                    out[n, f, j, i] = tmp

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    dx, dw, db = np.zeros_like(x), np.zeros_like(w), np.zeros_like(b)

    stride, pad = conv_param['stride'], conv_param['pad']

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.zeros_like(x_padded)

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    out_dim_h = (H + 2 * pad - HH) // stride + 1
    out_dim_w = (W + 2 * pad - WW) // stride + 1

    for n in range(N):
        for f in range(F):
            for j in range(out_dim_h):
                for i in range(out_dim_w):
                    _y = stride * j
                    _x = stride * i

                    dw[f] += dout[n, f, j, i] * x_padded[n, :, _y:_y+HH, _x:_x+WW]
                    db[f] += dout[n, f, j, i]
                    dx_padded[n, :, _y:_y+HH, _x:_x+WW] += dout[n, f, j, i] * w[f]
    
    dx = dx_padded[:, :, pad:-pad, pad:-pad]
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    cache = (x, pool_param)

    N, C, H, W = x.shape
    out_dim_h = (H - pool_height) // stride + 1
    out_dim_w = (W - pool_width) // stride + 1
    out = np.zeros((N, C, out_dim_h, out_dim_w))

    for n in range(N):
        for c in range(C):
            for j in range(out_dim_h):
                for i in range(out_dim_w):
                    _y = j * stride
                    _x = i * stride

                    out[n, :, j, i] = np.max(x[n, :, _y:_y+pool_height, _x:_x+pool_width], axis=(1, 2))

    return out, cache

def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    dx = np.zeros_like(x)

    N, C, H, W = x.shape
    out_dim_h = (H - pool_height) // stride + 1
    out_dim_w = (W - pool_width) // stride + 1

    for n in range(N):
        for c in range(C):
            for j in range(out_dim_h):
                for i in range(out_dim_w):
                    _y = j * stride
                    _x = i * stride

                    x_slice = x[n, :, _y:_y+pool_height, _x:_x+pool_width]
                    x_max = x_slice == np.max(x_slice, axis=(1, 2)).reshape((-1, 1, 1))
                    dx[n, :, _y:_y+pool_height, _x:_x+pool_width] = dout[n, :, j, i].reshape((-1, 1, 1)) * x_max

    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  #gamma_reshaped = gamma.reshape((-1, 1, 1))
  #beta_reshaped = beta.reshape((-1, 1, 1))
  N, C, H, W = x.shape
  x_reshaped = x.reshape((-1, C))
  out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
  out = out.reshape((N, C, H, W))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape
  dout_reshaped = dout.reshape((-1, C))
  dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
  dx = dx.reshape((N, C, H, W))
  #dgamma = dgamma.sum(axis=(1, 2))
  #dbeta = dbeta.sum(axis=(1, 2))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
