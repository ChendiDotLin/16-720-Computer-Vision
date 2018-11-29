import numpy as np
import scipy.io
from nn import *
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import skimage.measure
import torch.optim as optim

max_iters = 8

batch_size = 100
learning_rate = 1e-2
hidden_size = 64
# batches = get_random_batches(train_x,train_y,batch_size)
# batch_num = len(batches)

#


# EMNIST dataset
train_dataset = torchvision.datasets.EMNIST(root='../../data',
                                            split='balanced',
                                            train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.EMNIST(root='../../data',
                                           split='balanced',
                                           train=False,
                                          transform=transforms.ToTensor())
train_examples = len(train_dataset)
test_examples = len(test_dataset)
print(train_examples,test_examples)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# print(train_dataset.shape)

# model of nn
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.fc2 = torch.nn.Linear(500, 47)

    def forward(self, x):
        # print(x.size())
        x = torch.nn.functional.relu(self.conv1(x))
        # print(x.size())
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        # print(x.size())
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# model = torch.nn.Sequential(
#     torch.nn.Conv2d(1,20,5,1),
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(kernel_size=2),
#     torch.nn.Conv2d(20,50,5,1),
#     torch.nn.ReLU(),
#     torch.nn.MaxPool2d(kernel_size=2),
#     torch.nn.Linear(4,500),
#     torch.nn.ReLU(),
#     torch.nn.Linear(500,10)
#
# )

model = Net()
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
    valid_total_loss = 0
    valid_total_acc = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        out =  model(x)
        # print(target)
        loss = criterion(out,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(out.data, 1)

        total_acc += ((target==predicted).sum().item())
        # print(total_acc)
        total_loss+=loss
    ave_acc = total_acc/train_examples
    for batch_idx, (valid_x, valid_target) in enumerate(test_loader):
        valid_out =  model(valid_x)
        valid_loss = criterion(valid_out,valid_target)
        _, valid_predicted = torch.max(valid_out.data, 1)
        valid_total_acc += ((valid_target==valid_predicted).sum().item())
        # print(total_acc)
        valid_total_loss+=valid_loss
    valid_acc = valid_total_acc/test_examples

    training_loss_data.append(total_loss/(train_examples/batch_size))
    valid_loss_data.append(valid_loss/(test_examples/batch_size))
    training_acc_data.append(ave_acc)
    valid_acc_data.append(valid_acc)
    print('Validation accuracy: ', valid_acc)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,ave_acc))

# plt.figure(0)
# plt.plot(np.arange(max_iters), training_acc_data, 'r')
# plt.plot(np.arange(max_iters), valid_acc_data, 'b')
# plt.show()
# plt.figure(1)
# plt.plot(np.arange(max_iters),training_loss_data,'r')
# plt.plot(np.arange(max_iters),valid_loss_data,'b')
# plt.show()

# run on find letters
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    # plt.imshow(1-bw,cmap='gray')
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                             fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    all_boxes = []
    all_boxes.append([])
    line_num = 1
    bboxes.sort(key = lambda x:x[2])
    bottom = bboxes[0][2]
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        if(minr >= bottom):
            bottom = maxr
            all_boxes.append([])
            line_num += 1
        all_boxes[line_num-1].append(bbox)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([str(_) for _ in range(10)] +
                       [_ for _ in string.ascii_uppercase[:26]]+ ["A","B","D","E","F","G","H","N","Q","R","T"]  )
    params = pickle.load(open('q3_weights.pickle','rb'))
    for row in all_boxes:
        line = ""
        row.sort(key=lambda x: x[1])
        right = row[0][3]
        for box in row:
            minr, minc, maxr, maxc = box
            if (img == '01_list.jpg'):
                if (minc - right > 1.2 * (maxc - minc)):
                    line += " "
            else:
                if (minc - right > 0.8*(maxc-minc)):
                    line+= " "
            right = maxc
            letter = bw[minr:maxr, minc:maxc]
            #
            # print(letter)
            height,width = letter.shape
            # print(letter.shape)
            # letter = np.pad(letter,((width//5,height//5),(width//5,height//5)),'constant',constant_values=0.0)
            if (img == '01_list.jpg'):
                letter = skimage.morphology.dilation(letter, skimage.morphology.square(5))
                letter = np.pad(letter, ((25, 25), (25, 25)), 'constant', constant_values=0.0)
            else:
                letter = skimage.morphology.dilation(letter,skimage.morphology.square(15))
                letter = np.pad(letter,((70,70),(70,70)),'constant',constant_values=0.0)

            letter = skimage.transform.resize(letter,(28,28))
            # letter = 1.0-letter
            letter = letter *255.0
            #
            # plt.imshow(letter)

            # plt.show()

            letter = letter.T
            # print(letter)


            x = letter.reshape(1,28*28)
            x = torch.tensor(x).float()

            x = x.reshape(1, 1, 28, 28)  # add if using nist36
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            # print(predicted)
            # print(letters[predicted])
            line+=(letters[predicted])
        print(line)