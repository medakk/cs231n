import pickle

import numpy as np

from .layers import *
from .fast_layers import *
from .layer_utils import *
from .solver import Solver

from ..gradient_check import *

from copy import deepcopy

class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, layers,
                 input_dim=(3, 32, 32),
                 num_classes=10,
                 xavier_init=True,
                 weight_scale=1e-3, reg=0,
                 dtype=np.float32):
        """
        Initialize a new network.


        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - layers: a list of tuples. Each tuple can be one of:
            ('conv_relu', num_filters, filter_size, stride, pad)
            ('conv', num_filters, filter_size, stride, pad)
            ('pool', filter_size, stride)
            ('spatial_batchnorm', )
            ('fc_relu', output_dim)
            ('fc', output_dim)
            ('batchnorm', )
            ('dropout', p)

        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.params2 = {} # For the parameters that are not controlled by gradient descent
        self.reg = reg
        self.dtype = dtype

        layers.append(('fc', num_classes))
        self.layers = layers

        current_input_dim = input_dim
        for i in range(len(layers)):
            if not isinstance(self.layers[i], tuple):
                raise AttributeError('layer {} - "{}" is NOT a tuple'.format(i, self.layers[i]))
            name, *layer_param = layers[i]
            if name in ('conv', 'conv_relu'):
                C, H, W = current_input_dim
                num_filters, filter_size, stride, pad = layer_param
                conv_param = {'stride': stride, 'pad': pad}

                if xavier_init:
                    n_input = C * filter_size * filter_size
                    weight_scale = np.sqrt(2.0 / n_input)
                self.params['W{}'.format(i)] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
                self.params['b{}'.format(i)] = np.zeros(num_filters)
                self.params2['conv_param{}'.format(i)] = conv_param

                output_dim_h = (H - filter_size + 2 * pad) // stride + 1
                output_dim_w = (W - filter_size + 2 * pad) // stride + 1
                current_input_dim = (num_filters, output_dim_h, output_dim_w)
            elif name == 'pool':
                C, H, W = current_input_dim
                filter_size, stride = layer_param
                pool_param = {'pool_height': filter_size, 'pool_width': filter_size, 'stride': stride}

                self.params2['pool_param{}'.format(i)] = pool_param
                output_dim_h = (H - filter_size) // stride + 1
                output_dim_w = (W - filter_size) // stride + 1

                current_input_dim = (C, output_dim_h, output_dim_w)
            elif name in ('fc_relu', 'fc'):
                if isinstance(current_input_dim, tuple):
                    current_input_dim = np.product(current_input_dim)
                output_dim = layer_param[0]

                if xavier_init:
                    n_input = current_input_dim
                    weight_scale = np.sqrt(2.0 / n_input)
                self.params['W{}'.format(i)] = np.random.randn(current_input_dim, output_dim) * weight_scale
                self.params['b{}'.format(i)] = np.zeros(output_dim)

                current_input_dim = output_dim
            elif name == 'spatial_batchnorm':
                C, H, W = current_input_dim
                self.params2['bn_param{}'.format(i)] = {}

                self.params['gamma{}'.format(i)] = np.random.randn(C)
                self.params['beta{}'.format(i)] = np.zeros(C)
            elif name == 'batchnorm':
                self.params2['bn_param{}'.format(i)] = {}

                self.params['gamma{}'.format(i)] = np.random.randn(current_input_dim)
                self.params['beta{}'.format(i)] = np.zeros(current_input_dim)
            elif name == 'dropout':
                p = layer_param[0]
                self.params2['dropout_param{}'.format(i)] = {'p': p}
            else:
                raise AttributeError('{} is not a valid layer.'.format(layers[i]))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def save_to_file(self, filename):
        with open(filename, 'wb') as fd:
            pickle.dump(self, fd)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as fd:
            model = pickle.load(fd)
        return model

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # N, C, H, W = X.shape

        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        scores = X
        layers_cache = []
        for i in range(len(self.layers)):
            name, *_ = self.layers[i]
            if name == 'conv_relu':
                W, b = self.params['W{}'.format(i)], self.params['b{}'.format(i)]
                conv_param = self.params2['conv_param{}'.format(i)]
                scores, cache = conv_relu_forward(scores, W, b, conv_param)
            elif name == 'conv':
                W, b = self.params['W{}'.format(i)], self.params['b{}'.format(i)]
                conv_param = self.params2['conv_param{}'.format(i)]
                scores, cache = conv_forward_fast(scores, W, b, conv_param)
            elif name == 'pool':
                pool_param = self.params2['pool_param{}'.format(i)]
                scores, cache = max_pool_forward_fast(scores, pool_param)
            elif name == 'fc_relu':
                W, b = self.params['W{}'.format(i)], self.params['b{}'.format(i)]
                scores, cache = affine_relu_forward(scores, W, b)
            elif name == 'fc':
                W, b = self.params['W{}'.format(i)], self.params['b{}'.format(i)]
                scores, cache = affine_forward(scores, W, b)
            elif name == 'spatial_batchnorm':
                bn_param = self.params2['bn_param{}'.format(i)]
                bn_param['mode'] = mode
                gamma, beta = self.params['gamma{}'.format(i)], self.params['beta{}'.format(i)]
                scores, cache = spatial_batchnorm_forward(scores, gamma, beta, bn_param)
            elif name == 'batchnorm':
                bn_param = self.params2['bn_param{}'.format(i)]
                bn_param['mode'] = mode
                gamma, beta = self.params['gamma{}'.format(i)], self.params['beta{}'.format(i)]
                scores, cache = batchnorm_forward(scores, gamma, beta, bn_param)
            elif name == 'dropout':
                dropout_param = self.params2['dropout_param{}'.format(i)]
                dropout_param['mode'] = mode

                # Allow the dropout to be changed later
                dropout_param['p'] = self.layers[i][1]

                scores, cache = dropout_forward(scores, dropout_param)
                

            layers_cache.append(cache)

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dout = softmax_loss(scores, y)
        for i in range(len(self.layers) - 1, -1, -1):
            name, *_ = self.layers[i]
            cache = layers_cache.pop()
            if name == 'conv_relu':
                dx, dW, db = conv_relu_backward(dout, cache)
                grads['W{}'.format(i)] = dW
                grads['b{}'.format(i)] = db

                W = self.params['W{}'.format(i)]
                loss += 0.5 * self.reg * (W ** 2).sum()
                grads['W{}'.format(i)] += self.reg * W

                dout = dx
            elif name == 'conv':
                dx, dW, db = conv_backward_fast(dout, cache)
                grads['W{}'.format(i)] = dW
                grads['b{}'.format(i)] = db

                W = self.params['W{}'.format(i)]
                loss += 0.5 * self.reg * (W ** 2).sum()
                grads['W{}'.format(i)] += self.reg * W

                dout = dx
            elif name == 'pool':
                dx = max_pool_backward_fast(dout, cache)

                dout = dx
            elif name == 'fc_relu':
                dx, dW, db = affine_relu_backward(dout, cache)
                grads['W{}'.format(i)] = dW
                grads['b{}'.format(i)] = db

                W = self.params['W{}'.format(i)]
                loss += 0.5 * self.reg * (W ** 2).sum()
                grads['W{}'.format(i)] += self.reg * W

                dout = dx
            elif name == 'fc':
                dx, dW, db = affine_backward(dout, cache)
                grads['W{}'.format(i)] = dW
                grads['b{}'.format(i)] = db

                W = self.params['W{}'.format(i)]
                loss += 0.5 * self.reg * (W ** 2).sum()
                grads['W{}'.format(i)] += self.reg * W

                dout = dx
            elif name == 'spatial_batchnorm':
                dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
                grads['gamma{}'.format(i)], grads['beta{}'.format(i)] = dgamma, dbeta

                dout = dx
            elif name == 'batchnorm':
                dx, dgamma, dbeta = batchnorm_backward(dout, cache)
                grads['gamma{}'.format(i)], grads['beta{}'.format(i)] = dgamma, dbeta

                dout = dx
            elif name == 'dropout':
                dx = dropout_backward(dout, cache)

                dout = dx
        return loss, grads

def hyperparameter_search(orig_model,
                          lr_bound=(-6, -3), reg_bound=(-5, -1), 
                          data=None, random_search=True, count=15, train_size=3000):
    if data is None:
        from ..cifar import cifar_3d_data
        data = cifar_3d_data()

    # Take a subset of the data for training
    data = {'X_train': data['X_train'][:train_size],
            'y_train': data['y_train'][:train_size],
            'X_val':   data['X_val'],
            'y_val':   data['y_val']}

    if random_search:
        print('Random search...')
        for i in range(count):
            learning_rate = 10 ** np.random.uniform(*lr_bound)
            reg = 10 ** np.random.uniform(*reg_bound)

            print('{}: learning_rate={:.3e}, reg={:.3e}'.format(i+1, learning_rate, reg))
            model = deepcopy(orig_model)
            model.reg = reg
            solver = Solver(model, data, optim_config={'learning_rate': learning_rate}, lr_decay=1.0,
                            batch_size=200, update_rule='adam', num_epochs=2, verbose=False)
            solver.train()

            loss, train_acc, val_acc = solver.loss_history[-1], solver.train_acc_history[-1], solver.val_acc_history[-1]
            print('loss={:.3f}, train_acc={:.3f}, val_acc={:.3f}\n'.format(loss, train_acc, val_acc))
    else:
        i = 0
        print('Grid search...')
        for learning_rate in [10**x for x in np.arange(*lr_bound, (max(lr_bound) - min(lr_bound)) / count)]:
            for reg in [10**x for x in np.arange(*reg_bound, max(reg_bound) - min(reg_bound) / count)]:
                print('{}: learning_rate={:.3e}, reg={:.3e}'.format(i+1, learning_rate, reg))
                model = deepcopy(orig_model)
                model.reg = reg
                solver = Solver(model, data, optim_config={'learning_rate': learning_rate}, lr_decay=1.0,
                                batch_size=200, update_rule='adam', num_epochs=2, verbose=False)
                solver.train()

                loss, train_acc, val_acc = solver.loss_history[-1], solver.train_acc_history[-1], solver.val_acc_history[-1]
                print('loss={:.3f}, train_acc={:.3f}, val_acc={:.3f}\n'.format(loss, train_acc, val_acc))
                i += 1

def gradient_check(layers, *args, **kwargs):
    num_inputs = 2
    input_dim = (3, 16, 16)
    reg = 0.0
    num_classes = 10
    X = np.random.randn(num_inputs, *input_dim)
    y = np.random.randint(num_classes, size=num_inputs)

    model = ConvNet(layers=layers, input_dim=input_dim, num_classes=num_classes, dtype=np.float64, **kwargs)
    loss, grads = model.loss(X, y)
    for param_name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
        e = rel_error(param_grad_num, grads[param_name])
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
