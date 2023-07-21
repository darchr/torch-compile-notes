import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models.resnet import resnet50, Bottleneck
import torch.fx as fx
import torch
import numpy as np
import pickle
from math import ceil

# instead of using DataLoader which symbolic trace does not like, we can use other 
# methods to load data.

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
# test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(test, batch_size=128,shuffle=False, num_workers=2)


# 0: load the data 
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

data_batch_1 = unpickle('/home/lyy/torch_compile/data/cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('/home/lyy/torch_compile/data/cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('/home/lyy/torch_compile/data/cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('/home/lyy/torch_compile/data/cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('/home/lyy/torch_compile/data/cifar-10-batches-py/data_batch_5')
test_batch = unpickle('/home/lyy/torch_compile/data/cifar-10-batches-py/test_batch')


X_train_data = np.concatenate((data_batch_1['data'], data_batch_2['data'], data_batch_3['data'], data_batch_4['data'], data_batch_5['data']))
X_train_labels = data_batch_1['labels'] + data_batch_2['labels'] + data_batch_3['labels'] + data_batch_4['labels'] + data_batch_5['labels']
X_test_data = test_batch['data']
X_test_labels = test_batch['labels']

# 1: preprocess the data
X_train_data = X_train_data.reshape(len(X_train_data),3,32,32)

# 2: import the model
net = resnet50(10).to('cuda')

# 3: define optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

def segment(inputs, labels, losses):
  optimizer.zero_grad()

  outputs = net(inputs)
  loss = criterion(outputs, labels)
  losses.append(loss.item())

  loss.backward()
  optimizer.step()

  return loss.item()

batch_size = 128
split_data_list = np.array_split(X_train_data,ceil(X_train_data.shape[0]/batch_size), axis=0)
how_many_batches = len(split_data_list)
split_labels_list = np.array_split(X_train_labels,ceil(len(X_train_labels)/batch_size), axis=0)

# 4: start to train!

EPOCHS = 1 # actual:200
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0
    # for i, inp in enumerate(trainloader):
    #     inputs, labels = inp
    #     inputs, labels = inputs.to('cuda'), labels.to('cuda')

    #     optimizer.zero_grad()

    for i in range(how_many_batches):
        inputs = split_data_list[i]
        labels = split_labels_list[i]

        inputs = torch.from_numpy(inputs)
        inputs = inputs.float()
        inputs = inputs.to('cuda')

        labels = torch.from_numpy(labels)
        labels = labels.to('cuda')

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        # addon = segment(inputs, labels, losses)

        running_loss += loss.item()
        # running_loss += addon
        if i%100 == 0 and i > 0:
            print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
            running_loss = 0.0


    avg_loss = sum(losses)/len(losses)
    scheduler.step(avg_loss)

print('Training Done')

gm = torch.fx.symbolic_trace(segment)
# call and print the graph
gm.graph.print_tabular()

