# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: 3_neural_nets.py                                  #
#    DATE: 26 NOV 2020                                       #
# ********************************************************** #

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------------------------------------
# DEFINE THE NETWORK

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel
        # 6 output channels
        # 3x3 square convolution kernel

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # define the final ``fully connected'' layers
        self.fc1 = nn.Linear(in_features = 16 * 6 * 6,\
                             out_features = 120,\
                             bias=True) # 6 x 6
        
        self.fc2 = nn.Linear(in_features = 120,\
                             out_features = 84,\
                             bias=True)
        
        self.fc3 = nn.Linear(in_features = 84,\
                             out_features = 10,\
                             bias=True)

    def forward(self, x):
        # Max pooling over 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # If only one dim provided, assumed square, eg:
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):

        # all dimensions besides batch dimension
        size = x.size()[1:]

        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)


# ----------------------------------------------------------
# QUERY THE NETWORK

params = list(net.parameters())

# return length
print(len(params))

# conv1 weights
print(params[0])


# ----------------------------------------------------------
# FORWARD PROPAGATION

input = torch.randn(1, 1, 32, 32)
out1 = net(input)
print(out1)


# ----------------------------------------------------------
# RANDOM INITIALIZATION (?)

net.zero_grad()
out1.backward(torch.randn(1, 10))


# ----------------------------------------------------------
# LOSS FUNCTION

out2 = net(input)

target = torch.randn(10)        # a dummy target (y_true)
target = target.view(1, -1)     # reshape: [10] -> [1, 10]
criterion = nn.MSELoss()        # define loss function

loss = criterion(out2, target)
print(loss)

# inspect loss gradient a few steps backward:
# (1) MSE Loss
print(loss.grad_fn)

# (2) Linear Layer
print(loss.grad_fn.next_functions[0][0])

# (#) ReLU
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])


# ----------------------------------------------------------
# BACKPROPAGATION

net.zero_grad()    # zeros gradient buffers of all params

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# ----------------------------------------------------------
# UPDATE WEIGHTS

# python implementation of stochastic gradient descent (SGD)
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

    
# other optimizers can be implemented too...

# create optimizer:
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in training loop:
optimizer.zero_grad()    # zero the gradient buffers
out3 = net(input)
loss = criterion(out3, target)
loss.backward()
optimizer.step()    # does the update
