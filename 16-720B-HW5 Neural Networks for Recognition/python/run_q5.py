import numpy as np
import scipy.io
from nn import *
from collections import Counter
import skimage.measure
import matplotlib.pyplot as plt


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
examples = train_x.shape[0]
valid_examples = valid_x.shape[0]

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()
examples, dimension = train_x.shape
# print(dimension)
# initialize layers here
initialize_weights(dimension,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'hidden')
initialize_weights(hidden_size,hidden_size,params,'hidden2')
initialize_weights(hidden_size,dimension,params,'output')
params['m_Wlayer1'] = np.zeros((dimension,hidden_size))
params['m_Whidden'] = np.zeros((hidden_size,hidden_size))
params['m_Whidden2'] = np.zeros((hidden_size,hidden_size))
params['m_Woutput'] = np.zeros((hidden_size,dimension))
params['m_blayer1'] = np.zeros(hidden_size)
params['m_bhidden'] = np.zeros(hidden_size)
params['m_bhidden2'] = np.zeros(hidden_size)
params['m_boutput'] = np.zeros(dimension)

training_loss_data = []
valid_loss_data = []
training_acc_data = []
valid_acc_data = []

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0

    for xb,_ in batches:

        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2 which is 2(x-y)
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        out = forward(h3, params, 'output', sigmoid)
        # print(xb,h1,h2,h3,out)
        loss = np.sum((xb - out)**2)
        total_loss += loss
        delta1 = 2*(out-xb)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2,params,'hidden2' , relu_deriv)
        delta4 = backwards(delta3,params,'hidden' , relu_deriv)
        backwards(delta3,params,'layer1' , relu_deriv)

        # apply gradient

        params['m_Wlayer1'] = 0.9* params['m_Wlayer1'] - learning_rate * params['grad_Wlayer1']
        params['Wlayer1'] += params['m_Wlayer1']
        params['m_blayer1'] = 0.9 * params['m_blayer1'] - learning_rate * params['grad_blayer1']
        params['blayer1'] += params['m_blayer1']
        params['m_Whidden'] = 0.9 * params['m_Whidden'] - learning_rate * params['grad_Whidden']
        params['Whidden'] += params['m_Whidden']
        params['m_bhidden'] = 0.9 * params['m_bhidden'] - learning_rate * params['grad_bhidden']
        params['bhidden'] += params['m_bhidden']
        params['m_Whidden2'] = 0.9 * params['m_Whidden2'] - learning_rate * params['grad_Whidden2']
        params['Whidden2'] += params['m_Whidden2']
        params['m_bhidden2'] = 0.9 * params['m_bhidden2'] - learning_rate * params['grad_bhidden2']
        params['bhidden2'] += params['m_bhidden2']
        params['m_Woutput'] = 0.9 * params['m_Woutput'] - learning_rate * params['grad_Woutput']
        params['Woutput'] += params['m_Woutput']
        params['m_boutput'] = 0.9 * params['m_boutput'] - learning_rate * params['grad_boutput']
        params['boutput'] += params['m_boutput']

        # params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        # params['blayer1'] -= learning_rate * params['grad_blayer1']
        # params['Whidden'] -= learning_rate * params['grad_Whidden']
        # params['blayer1'] -= learning_rate * params['grad_bhidden']
        # params['Whidden2'] -= learning_rate * params['grad_Whidden2']
        # params['bhidden2'] -= learning_rate * params['grad_bhidden2']
        # params['Woutput'] -= learning_rate * params['grad_Woutput']
        # params['boutput'] -= learning_rate * params['grad_boutput']
    valid_h1 = forward(valid_x, params, 'layer1', relu)
    valid_h2 = forward(valid_h1, params, 'hidden', relu)
    valid_h3 = forward(valid_h2, params, 'hidden2', relu)
    valid_out = forward(valid_h3, params, 'output', sigmoid)
    valid_loss = np.sum((valid_x - valid_out) ** 2)

    training_loss_data.append(total_loss / examples)
    valid_loss_data.append(valid_loss / valid_examples)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

plt.figure(0)
plt.plot(np.arange(max_iters),training_loss_data,'r')
plt.plot(np.arange(max_iters),valid_loss_data,'b')
plt.legend(['training loss','valid loss'])
plt.show()

# visualize some results
# Q5.3.1
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()




from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2

valid_h1 = forward(valid_x, params, 'layer1',relu)
valid_h2 = forward(valid_h1,params,'hidden',relu)
valid_h3 = forward(valid_h2,params,'hidden2',relu)
valid_out = forward(valid_h3,params,'output',sigmoid)

for i in range(5):
    plt.subplot(2,1,1)
    index = int(3600/5*i)
    plt.imshow(valid_x[index,:].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(valid_out[index,:].reshape(32,32).T)
    plt.show()
    plt.subplot(2, 1, 1)
    index = int(3600 / 5 * i+1)
    plt.imshow(valid_x[index, :].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(valid_out[index, :].reshape(32, 32).T)
    plt.show()
# psnr = skimage.measure.compare_psnr(valid_x,valid_out)
# print(psnr)
total = []
for pred,gt in zip(valid_out,valid_x):
    total.append(skimage.measure.compare_psnr(gt,pred))
print(np.array(total).mean())