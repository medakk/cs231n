import numpy as np
import matplotlib.pyplot as plt

from cs231n.layers import softmax_loss
from cs231n.image_utils import *

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images, of shape (N, 3, H, W)
    - y: Labels for X, of shape (N,)
    - model: A PretrainedCNN that will be used to compute the saliency map.

    Returns:
    - saliency: An array of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    saliency = None
    ##############################################################################
    # TODO: Implement this function. You should use the forward and backward     #
    # methods of the PretrainedCNN class, and compute gradients with respect to  #
    # the unnormalized class score of the ground-truth classes in y.             #
    ##############################################################################
    N = X.shape[0]
    out, cache = model.forward(X)

    dout = np.zeros_like(out)
    dout[np.arange(N), y] = 1.0

    dX, _ = model.backward(dout, cache)
    saliency = dX.max(axis=1)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.
  
    Inputs:
    - X: Input image, of shape (1, 3, 64, 64)
    - target_y: An integer in the range [0, 100)
    - model: A PretrainedCNN
  
    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
      by the model.
    """
    X_fooling = X.copy()
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. Use gradient ascent on the target class score, using   #
    # the model.forward method to compute scores and the model.backward method   #
    # to compute image gradients.                                                #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    ##############################################################################
    
    N,C,H,W = X_fooling.shape
    i = 0
    y_pred = 10000
    while (y_pred != target_y) & (i<200): 
        
        scores, cache = model.forward(X_fooling, mode='test') # Score size (N,100)
        # The function we want to optimize
        """
        loss = scores[np.arange(N), target_y]# Size (N,)
        #The gradient of this loss wih respect to the input image simply writes
        dscores = np.zeros_like(scores)
        dscores[np.arange(N),target_y] = 1.0
        dX, grads = model.backward(dscores, cache)
    
        lr = 1000
        X_fooling += lr*dX
        y_pred = model.loss(X_fooling).argmax(axis=1) 
        """

        loss, dscores = softmax_loss(scores, np.array([target_y]))
        dX, _ = model.backward(dscores, cache)
        lr = 1000
        X_fooling -= lr*dX
        y_pred = model.loss(X_fooling).argmax(axis=1) 
        i+=1
    
    print('We fool the image in %d iterations'%(i))
    return X_fooling

def create_class_visualization(target_y, model, data, init_X=None, **kwargs):
    """
    Perform optimization over the image to generate class visualizations.

    Inputs:
    - target_y: Integer in the range [0, 100) giving the target class
    - model: A PretrainedCNN that will be used for generation

    Keyword arguments:
    - learning_rate: Floating point number giving the learning rate
    - blur_every: An integer; how often to blur the image as a regularizer
    - l2_reg: Floating point number giving L2 regularization strength on the image;
    this is lambda in the equation above.
    - max_jitter: How much random jitter to add to the image as regularization
    - num_iterations: How many iterations to run for
    - show_every: How often to show the image
    """

    learning_rate = kwargs.pop('learning_rate', 10000)
    blur_every = kwargs.pop('blur_every', 1)
    l2_reg = kwargs.pop('l2_reg', 1e-6)
    max_jitter = kwargs.pop('max_jitter', 4)
    num_iterations = kwargs.pop('num_iterations', 100)
    show_every = kwargs.pop('show_every', 25)

    X = np.random.randn(1, 3, 64, 64) if init_X is None else init_X.copy()
    N = X.shape[0]
    for t in range(num_iterations):
        # As a regularizer, add random jitter to the image
        ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
        X = np.roll(np.roll(X, ox, -1), oy, -2)

        dX = None
        ############################################################################
        # TODO: Compute the image gradient dX of the image with respect to the     #
        # target_y class score. This should be similar to the fooling images. Also #
        # add L2 regularization to dX and update the image X using the image       #
        # gradient and the learning rate.                                          #
        ############################################################################
        scores, cache = model.forward(X)
        dscores = np.zeros_like(scores)
        dscores[np.arange(N), target_y] = 1.0
        dX, _ = model.backward(dscores, cache)
        dX += l2_reg * X

        X += learning_rate * dX
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, -1), -oy, -2)

        # As a regularizer, clip the image
        X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])

        # As a regularizer, periodically blur the image
        if t % blur_every == 0:
            X = blur_image(X)

        # Periodically show the image
        if t % show_every == 0:
            plt.imshow(deprocess_image(X, data['mean_image']))
            plt.gcf().set_size_inches(3, 3)
            plt.axis('off')
            plt.show()
    return X

def invert_features(target_feats, layer, model, data, **kwargs):
    """
    Perform feature inversion in the style of Mahendran and Vedaldi 2015, using
    L2 regularization and periodic blurring.

    Inputs:
    - target_feats: Image features of the target image, of shape (1, C, H, W);
    we will try to generate an image that matches these features
    - layer: The index of the layer from which the features were extracted
    - model: A PretrainedCNN that was used to extract features

    Keyword arguments:
    - learning_rate: The learning rate to use for gradient descent
    - num_iterations: The number of iterations to use for gradient descent
    - l2_reg: The strength of L2 regularization to use; this is lambda in the
    equation above.
    - blur_every: How often to blur the image as implicit regularization; set
    to 0 to disable blurring.
    - show_every: How often to show the generated image; set to 0 to disable
    showing intermediate reuslts.

    Returns:
    - X: Generated image of shape (1, 3, 64, 64) that matches the target features.
    """
    learning_rate = kwargs.pop('learning_rate', 10000)
    num_iterations = kwargs.pop('num_iterations', 500)
    l2_reg = kwargs.pop('l2_reg', 1e-7)
    blur_every = kwargs.pop('blur_every', 1)
    show_every = kwargs.pop('show_every', 50)

    X = np.random.randn(1, 3, 64, 64)
    for t in range(num_iterations):
        ############################################################################
        # TODO: Compute the image gradient dX of the reconstruction loss with      #
        # respect to the image. You should include L2 regularization penalizing    #
        # large pixel values in the generated image using the l2_reg parameter;    #
        # then update the generated image using the learning_rate from above.      #
        ############################################################################
        features, cache = model.forward(X, end=layer)
        # loss = np.sum((features - target_feats) ** 2) # L2 Distance
        # loss += 0.5 * l2_reg * np.sum(X ** 2)
        dfeat = 2 * (features - target_feats)
        dX, _ = model.backward(dfeat, cache)
        dX += l2_reg * X

        X -= learning_rate * dX
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # As a regularizer, clip the image
        X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])

        # As a regularizer, periodically blur the image
        if (blur_every > 0) and t % blur_every == 0:
          X = blur_image(X)

        if (show_every > 0) and (t % show_every == 0 or t + 1 == num_iterations):
            plt.imshow(deprocess_image(X, data['mean_image']))
            plt.gcf().set_size_inches(3, 3)
            plt.axis('off')
            plt.title('t = %d' % t)
            plt.show()

def deepdream(X, layer, model, data, **kwargs):
    """
    Generate a DeepDream image.

    Inputs:
    - X: Starting image, of shape (1, 3, H, W)
    - layer: Index of layer at which to dream
    - model: A PretrainedCNN object

    Keyword arguments:
    - learning_rate: How much to update the image at each iteration
    - max_jitter: Maximum number of pixels for jitter regularization
    - num_iterations: How many iterations to run for
    - show_every: How often to show the generated image
    """

    X = X.copy()

    learning_rate = kwargs.pop('learning_rate', 5.0)
    max_jitter = kwargs.pop('max_jitter', 16)
    num_iterations = kwargs.pop('num_iterations', 100)
    show_every = kwargs.pop('show_every', 25)

    for t in range(num_iterations):
        # As a regularizer, add random jitter to the image
        ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
        X = np.roll(np.roll(X, ox, -1), oy, -2)

        dX = None
        ############################################################################
        # TODO: Compute the image gradient dX using the DeepDream method. You'll   #
        # need to use the forward and backward methods of the model object to      #
        # extract activations and set gradients for the chosen layer. After        #
        # computing the image gradient dX, you should use the learning rate to     #
        # update the image X.                                                      #
        ############################################################################
        features, cache = model.forward(X, end=layer)
        dX, _ = model.backward(features, cache)

        X += learning_rate * dX
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Undo the jitter
        X = np.roll(np.roll(X, -ox, -1), -oy, -2)

        # As a regularizer, clip the image
        mean_pixel = data['mean_image'].mean(axis=(1, 2), keepdims=True)
        X = np.clip(X, -mean_pixel, 255.0 - mean_pixel)

        # Periodically show the image
        if t == 0 or (t + 1) % show_every == 0:
            img = deprocess_image(X, data['mean_image'], mean='pixel')
            plt.imshow(img)
            plt.title('t = %d' % (t + 1))
            plt.gcf().set_size_inches(8, 8)
            plt.axis('off')
            plt.show()
    return X
