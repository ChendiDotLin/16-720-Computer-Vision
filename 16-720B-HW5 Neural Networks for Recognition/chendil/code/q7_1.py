import numpy as np
import scipy.io
from nn import *
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import skimage.measure
import torch.optim as optim


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
# print(valid_y)
valid_examples, classes = valid_y.shape

valid_x = torch.from_numpy(valid_x).float()
valid_y = torch.from_numpy(valid_y).long()
max_iters = 50

batch_size = 64
learning_rate = 1e-2
hidden_size = 64
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

examples, dimension = train_x.shape
examples, classes = train_y.shape
print(examples)
# model of nn
model = torch.nn.Sequential(
    torch.nn.Linear(dimension,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,classes),
    # torch.nn.Softmax()
)
training_loss_data = []
valid_loss_data = []
training_acc_data = []
valid_acc_data = []
# torch.nn.init.xavier_uniform(model(Linear.weight))

criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCELoss()
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

for itr in range(max_iters):
    total_loss = 0
    total_acc = 0

    for xb,yb in batches:
        xb = torch.from_numpy(xb).float()
        yb = torch.from_numpy(yb).long()
        label = np.where(yb == 1)[1]
        label = torch.tensor(label)

        out = model(xb)
        # print(out.shape,yb.shape)
        # print(np.sum(out,axis = 1))
        # loss = torch.nn.functional.mse_loss(out,yb)
        loss = criterion(out,label)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        _, predicted = torch.max(out.data, 1)

        total_acc += ((label==predicted).sum().item())
        # print(total_acc)
        total_loss+=loss
    ave_acc = total_acc/examples

    valid_label = torch.tensor(np.where(valid_y==1)[1])
    valid_out = model(valid_x)
    valid_loss = criterion(valid_out,valid_label)
    # print(np.double(valid_loss/1.0))
    _, valid_predicted = torch.max(valid_out.data, 1)
    # print(valid_label[20:30])
    # print(valid_predicted[20:30])
    # print(label)
    valid_acc = (valid_label==valid_predicted).sum().item()/valid_examples
    training_loss_data.append(total_loss/batch_num)
    valid_loss_data.append(valid_loss)
    training_acc_data.append(ave_acc)
    valid_acc_data.append(valid_acc)
    print('Validation accuracy: ', valid_acc)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,ave_acc))

plt.figure(0)
plt.plot(np.arange(max_iters), training_acc_data, 'r')
plt.plot(np.arange(max_iters), valid_acc_data, 'b')
plt.legend(['training accuracy','valid accuracy'])
plt.show()
plt.figure(1)
plt.plot(np.arange(max_iters),training_loss_data,'r')
plt.plot(np.arange(max_iters),valid_loss_data,'b')
plt.legend(['training loss','valid loss'])
plt.show()