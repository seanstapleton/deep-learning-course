from builtins import range
import numpy as np


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    out = np.matmul(x, w)
    out += b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din)
    - dw: Gradient with respect to w, of shape (Din, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    N, d_out = dout.shape
    dx = np.matmul(dout, np.transpose(w))
    dw = np.matmul(np.transpose(x), dout)
    db = np.matmul(np.ones((1,dout.shape[0])), dout).reshape((-1,))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        
        sample_mean = np.mean(x, axis=0)
        sample_variance = np.var(x, axis=0)
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_variance
        
        num = x - sample_mean
        denom = np.sqrt(sample_variance + eps)
        
        x_hat = num/denom
        out = gamma*x_hat + beta
        
        cache = (gamma, x_hat, num, denom, eps, sample_variance)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        num = x - running_mean
        denom = np.sqrt(running_var + eps)
        
        x_hat = num/denom
        out = gamma*x_hat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    gamma, x_hat, num, denom, eps, sample_variance = cache
    N, D = dout.shape
    
    dbeta = np.sum(dout, axis=0)
    dyx_hat = dout
    dgamma = np.sum(dyx_hat*x_hat, axis=0)
    dx_hat = gamma*dyx_hat
    ddenom = np.sum(num*dx_hat, axis=0)
    dmu1 = (1/denom)*dx_hat
    dsqvar = ddenom*(-1)*(1/(denom**2))
    dvar = 0.5*((sample_variance+eps)**(-0.5))*dsqvar
    dsq = (1/N)*np.ones((N,D))*dvar
    dmu2 = 2*num*dsq
    dmu = (-1)*np.sum(dmu1+dmu2, axis=0)
    dx1 = dmu1 + dmu2
    dx2 = (1/N)*np.ones((N,D))*dmu
    dx = dx1+dx2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Implement the vanilla version of dropout.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.binomial([np.ones(x.shape)], p)[0] == 0
        out = (x * mask)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = mask*dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

# 2D 'valid' padding
def conv2D(a,b):
    if b.shape[0] > a.shape[0]:
        return conv2D(b,a)
    out_shape = tuple(np.subtract(a.shape,b.shape)+1) + b.shape
    conv_steps = np.lib.stride_tricks.as_strided(a,out_shape,a.strides+a.strides)
    # (3,3) (4,4,3,3)
    return np.einsum('ij,klij->kl',np.flip(b),conv_steps)

def conv3D(a,b):
    if b.shape[0] > a.shape[0]:
        return conv3D(b,a)
    out_shape = tuple(np.subtract(a.shape,b.shape)+1) + b.shape
    conv_steps = np.lib.stride_tricks.as_strided(a,out_shape,a.strides+a.strides)
    return np.einsum('ijk,abcijk->abc',np.flip(b),conv_steps)

def filt2D(a,b):
    if b.shape[0] > a.shape[0]:
        return filt2D(b,a)
    return conv2D(a,np.flip(b))

def filt3D(a,b):
    if b.shape[0] > a.shape[0]:
        return filt3D(b,a)
    return conv3D(a,np.flip(b))

def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW. Assume that stride=1 
    and there is no padding. You can ignore the bias term in your 
    implementation.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = H - HH + 1
      W' = W - WW + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    out = np.zeros((N,F,H-HH+1,W-WW+1))
    
    w = np.flip(w)

    for n in range(N):
        for f in range(F):
            out[n][f] = filt3D(x[n],w[f])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache

def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    _,_,A,B = dout.shape
    
    dx = np.zeros((N,C,A+HH-1,B+WW-1))
    dw = np.zeros((F,C,H-A+1,W-B+1))

    for n in range(N):
        for c in range(C):
            for f in range(F):
                padded = np.pad(w[f][c], ((A-1,),(B-1,)), mode='constant')
                dx[n][c] += conv2D(padded, dout[n][f])
                dw[f][c] += conv2D(x[n][c], np.flip(dout[n][f]))
    dw = np.flip(dw)
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
        
    N, C, H, W  = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    
    HH = 1 + (H - pool_height) // stride
    WW = 1 + (W - pool_width) // stride
    x_strides = x[0][0].strides
    strides = tuple(np.array(x_strides)*stride)
    
    out = np.zeros((N,C,HH,WW))
    
    for n in range(N):
        for c in range(C):
            out_shape = (HH,WW,pool_height,pool_width)
            pool_blocks = np.lib.stride_tricks.as_strided(x[n][c],out_shape,strides+x_strides)
            out[n][c] = np.max(pool_blocks, axis=(2,3))
                
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W  = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    
    HH = 1 + (H - pool_height) // stride
    WW = 1 + (W - pool_width) // stride
    x_strides = x[0][0].strides
    strides = tuple(np.array(x_strides)*stride)
    
    dx = np.zeros(x.shape)
    
    for n in range(N):
        for c in range(C):
            for h in range(HH):
                for w in range(WW):
                    h_start = stride * h
                    h_end = h_start + pool_height

                    w_start = stride * w
                    w_end = w_start + pool_width

                    # get the pool window in the input x
                    pool_window = x[n, c, h_start:h_end, w_start:w_end]
                    
                    m = np.max(pool_window)
                    dx_window = np.where(pool_window == m, 1, 0)
                    
                    dx[n, c, h_start:h_end, w_start:w_end] += dx_window * dout[n, c, h, w]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient for binary SVM classification.
    Inputs:
    - x: Input data, of shape (N,) where x[i] is the score for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    N,_ = x.shape
    
    y_p = np.where(y == 1,1,-1)
        
    losses = np.maximum(0,1-(x*y_p))
    loss = np.sum(losses)/N
    dx = np.where(losses > 0, 1, 0)*(-y_p)/N
    dx = dx.reshape((-1,))

    return loss, dx

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_loss(x, y):
    """
    Computes the loss and gradient for binary classification with logistic 
    regression.
    Inputs:
    - x: Input data, of shape (N,) where x[i] is the logit for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i], labels = +-1
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    x = x.reshape((-1,))
    y = y.reshape((-1,))
    
    N, = x.shape
    
    y_p = np.where(y == 1,1,0)

    p = sigmoid(x)
    loss = -(y_p*np.log(p) + (1-y_p)*np.log(1-p))
    loss = np.sum(loss)/N

    dx = (1/N)*(p - y_p)
        
    return loss, dx

def softmax(a):
    num = np.exp(a)
    denom = np.sum(num, axis=1, keepdims=True)+1e-5
    return num/(denom)

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    eps = 1e-5
    
    N,C = x.shape
    p = softmax(x)
    llikelihood = -np.log(p[range(N),y] + eps)
#     print(llikelihood)
    loss = np.sum(llikelihood) / N

    dx = p
    dx[range(N),y] -= 1
    dx = dx/N
    
    return loss, dx
