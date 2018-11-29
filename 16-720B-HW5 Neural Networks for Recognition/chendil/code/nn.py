import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    bound = np.sqrt(6/(in_size + out_size))
    W = np.random.uniform(-1*bound,bound,(in_size,out_size))
    b = np.zeros(out_size)
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = np.matmul(X,W)+b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    examples,classes = x.shape
    res = np.zeros((examples,classes))
    for i in range(examples):
        temp = x[i,:]
        max = np.max(temp)
        # print(max)
        temp = temp - max
        temp = np.exp(temp)
        sum = np.sum(temp)
        res[i,:] = temp/sum

    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    # loss, acc = None, None
    loss = -np.sum(y*np.log(probs))
    examples,classes = y.shape
    count = 0
    for i in range(examples):
        idx = np.argmax(probs[i,:])
        if (y[i,idx] == 1):
            count += 1
    acc = count/examples
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    examples,classes = delta.shape
    # print(delta.shape)
    # print(activation_deriv(post_act).shape)
    deriv = activation_deriv(post_act)
    grad_W = np.matmul(X.T,deriv*delta)
    grad_X = np.matmul(deriv*delta,W.T)
    grad_b = np.matmul(np.ones((1,examples)),deriv*delta).flatten()
    # print(examples)
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    Nx,Dx = x.shape
    Ny,Dy = y.shape
    idx = np.random.choice(Nx,size = (int(Nx/batch_size),batch_size))
    # print(idx)
    for i in range(len(idx)):
        batchx = x[idx[i],:]
        batchy = y[idx[i],:]
        batches.append((batchx,batchy))
    return batches
