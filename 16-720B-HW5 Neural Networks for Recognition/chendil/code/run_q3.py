import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50

# pick a batch size, learning rate
batch_size = 32
learning_rate = 3e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
examples, dimension = train_x.shape
examples, classes = train_y.shape
valid_examples= valid_x.shape[0]
print(examples,classes,dimension)
initialize_weights(dimension,hidden_size,params,'layer1')
initial_W = params['Wlayer1']
# fig2 = plt.figure()
# grid = ImageGrid(fig2, 111, nrows_ncols=(8,8,),axes_pad=0.0)
# for i in range(64):
#     grid[i].imshow(initial_W[:,i].reshape((32,32)))
initialize_weights(hidden_size,classes,params,'output')
training_loss_data = []
valid_loss_data = []
training_acc_data = []
valid_acc_data = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0

    for xb,yb in batches:
        # pass
        # training loop can be exactly the same as q2!
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        # print(probs)
        # print(probs.shape)
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        # print(acc)
        # print(loss)
        total_loss += loss
        total_acc += acc
        # backward
        delta1 = probs - yb
        # y_idx = np.where(yb == 1.0)[1]
        # print(y_idx)
        # print(delta1.shape)
        # delta1[np.arange(probs.shape[0]), y_idx] -= 1
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']
        # print(params['Wlayer1'][0][0])
        # print(params['grad_Wlayer1'][0][0])
        # print(params['grad_boutput'])
    avg_acc = total_acc / batch_num
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

    # run on validation set and report accuracy! should be above 75%
    # valid_acc = None
    valid_h1 = forward(valid_x, params, 'layer1')
    valid_probs = forward(valid_h1, params, 'output', softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, valid_probs)
    print('Validation accuracy: ',valid_acc)
    training_loss_data.append(total_loss/examples)
    valid_loss_data.append(valid_loss/valid_examples)
    training_acc_data.append(avg_acc)
    valid_acc_data.append(valid_acc)

    if False: # view the data
        for crop in xb:
            import matplotlib.pyplot as plt
            # print(crop.reshape(32,32))
            plt.imshow(crop.reshape(32,32).T)
            plt.show()
plt.figure(0)
plt.plot(np.arange(max_iters),training_loss_data,'r')
plt.plot(np.arange(max_iters),valid_loss_data,'b')
plt.legend(['training loss','valid loss'])
plt.figure(1)
plt.plot(np.arange(max_iters),training_acc_data,'r')
plt.plot(np.arange(max_iters),valid_acc_data,'b')
plt.legend(['training accuracy','valid accuracy'])
plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3

W_firstlayer = params['Wlayer1']
shape,cols = W_firstlayer.shape
print(examples,shape)
fig1 = plt.figure()
grid = ImageGrid(fig1, 111, nrows_ncols=(8,8,),axes_pad=0.0)
for i in range(cols):
    grid[i].imshow(W_firstlayer[:,i].reshape((32,32)))




# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
for i in range(examples):
    xb = train_x[i,:].reshape((1,dimension))
    yb = train_y[i,:].reshape((1,classes))
    h1 = forward(xb, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    # print(probs.shape)
    x_idx = np.argmax(probs[0, :])
    y_idx = np.where(yb == 1.0)[1][0]
    # print(y_idx)
    confusion_matrix[x_idx,y_idx]+=1

import string
plt.figure()
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()