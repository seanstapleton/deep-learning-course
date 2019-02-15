import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
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
    C, H, W = input_dim
    F, HH, WW = num_filters, filter_size, filter_size
    
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, C, filter_size,filter_size))
#     conv_out_size = (F,H-HH+1,W-WW+1)
    pool_sizes = {
        'h': 1 + (H-HH+1 - 2)//2,
        'w': 1 + (W-WW+1 - 2)//2
    }
    pool_out_size = F*pool_sizes['h']*pool_sizes['w']
    
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(pool_out_size,hidden_dim))
    self.params['b2'] = np.ones((hidden_dim,))
    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.ones((num_classes,))
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
    W1 = self.params['W1']
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
    N = X.shape[0]
    X1,cache_1 = conv_forward(X,W1)
    A1,cache_relu1 = relu_forward(X1)
    Z1,cache_max_pool1 = max_pool_forward(A1,{ 'pool_height': 2, 'pool_width': 2, 'stride': 2 })
    max_pool_shape = Z1.shape
    X2,cache_2 = fc_forward(Z1.reshape((N,-1)), W2, b2)
    scores,cache_3 = fc_forward(X2, W3, b3)
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
    reg_cost = np.sum(np.square(W3)) + np.sum(np.square(W2)) + np.sum(np.square(W1))
    reg_cost = (1/N)*self.reg
    loss += reg_cost
    
    dx3,dw3,db3 = fc_backward(dout, cache_3)
    dx2,dw2,db2 = fc_backward(dx3, cache_2)
    dz1 = max_pool_backward(dx2.reshape(max_pool_shape), cache_max_pool1)
    da1 = relu_backward(dz1, cache_relu1)
    dx1,dw1 = conv_backward(da1, cache_1)
    
    grads['W1'] = dw1 + self.params['W1']*reg_cost
    grads['W2'] = dw2 + self.params['W2']*reg_cost
    grads['b2'] = db2
    grads['W3'] = dw3 + self.params['W3']*reg_cost
    grads['b3'] = db3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
