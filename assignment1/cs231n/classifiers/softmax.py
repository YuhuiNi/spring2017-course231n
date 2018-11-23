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
  N=X.shape[0]
  score=X.dot(W)
  num_class=W.shape[1]
  for i in range(N):
        loss-=np.log(np.exp(score[i,y[i]])/np.sum(np.exp(score[i,:])))
        for j in range(num_class):
            if j==y[i]:
                dW[:,y[i]]-=(1-np.exp(score[i,y[i]])/np.sum(np.exp(score[i,:])))*X[i,:]
            else:
                dW[:,j]+=(np.exp(score[i,j])/np.sum(np.exp(score[i,:])))*X[i,:]
  loss/=N
  loss+=reg*np.sum(W*W)
    
  dW/=N
  dW+=2*reg*W
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
  N=X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #soft_max_score=1/np.sum(np.exp(X.dot(W)),1)*np.array(np.exp(X.dot(W))
  soft_max_score=np.array(np.exp(X.dot(W))/np.repeat(np.sum(np.exp(X.dot(W)),1).reshape(N,1),W.shape[1],1))
  loss-=np.sum(np.log(np.choose(y,soft_max_score.T)))
  
  loss/=N
  loss+=reg*np.sum(W*W)
  
  idx_matrix=soft_max_score
  idx_matrix[np.arange(N),y]-=1
  dW=X.T.dot(idx_matrix)

  dW/=N
  dW+=2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

