import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_class):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    softmax = np.exp(scores) / np.sum(np.exp(scores),keepdims = True)
    for j in range(num_class):
        if j==y[i]:
            loss += -np.log(softmax[j])
            dW[:,j]+=(softmax[j]-1)*X[i].T
        else:
            dW[:,j]+=(softmax[j])*X[i].T
        
  loss = loss/float(num_train)+0.5*reg*np.mean(W*W)
  dW = dW/float(num_train)+reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W) 
  softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  l1 = -np.log(softmax[range(num_train),y])
  data_loss = np.sum(l1)/num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss   
  dscores = softmax
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  dW = np.dot(X.T, dscores)
  dW += reg*W  
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

