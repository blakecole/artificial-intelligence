# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: simple_nn.py                                      #
#    DATE: 26 NOV 2020                                       #
# ********************************************************** #

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# helper function to query user options
def yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} ? [y/n]: '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False

# set up saving / loading
LOAD_PATH = './simple_nn.pth'
SAVE_PATH = './simple_nn.pth'

# ----------------------------------------------------------
# GENERATE TRAINING DATA & VALIDATION DATA

# -------- TRAINING DATA --------
x = np.linspace(-np.pi, np.pi, 201)
y = np.sin(x)

N = 100000
xrand = np.random.uniform(-np.pi, np.pi, N)
yrand = np.sin(xrand)

# convert numpy arrays to torch tensors
xt = torch.from_numpy(xrand)
xt = xt.type(torch.float)
yt = torch.from_numpy(yrand)
yt = yt.type(torch.float)

# split into batches
batch_size = 20
N_batch = int(N/batch_size)
xtb = xt.view(N_batch, -1)
ytb = yt.view(N_batch, -1)

print('\n\nBatch Size  = ', batch_size)
print('Num Batches = ', N_batch)
print('\nInput Array Size: ', xtb.size())
print('Target Array Size: ', ytb.size())

# -------- VALIDATION DATA --------
N2 = 200
xrand2 = np.random.uniform(-np.pi, np.pi, N2)
yrand2 = np.sin(xrand2)

# convert numpy arrays to torch tensors
xt2 = torch.from_numpy(xrand2)
xt2 = xt2.type(torch.float)
yt2 = torch.from_numpy(yrand2)
yt2 = yt2.type(torch.float)


# ----------------------------------------------------------
# DEFINE THE NETWORK

# 1 input
# 1 hidden layer with 120 neurons
# 1 output

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # define fully connected layers
        self.fc1 = nn.Linear(in_features = 1,\
                             out_features = 120,\
                             bias=True)
        
        self.fc2 = nn.Linear(in_features = 120,\
                             out_features = 1,\
                             bias=True)

        
    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

# return size of each parameter
print([p.size() for p in params])


# ----------------------------------------------------------
# DEFINE LOSS FUNCTION & OPTIMIZER

criterion = nn.MSELoss()       # define loss function
learning_rate = 0.01
optimizer = optim.SGD(net.parameters(), learning_rate)


# ----------------------------------------------------------
# TRAIN PARAMETERS / LOAD  PARAMETERS

if (yesno('Train neural net')):
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(N_batch):
            # get the inputs; data is a list of [inputs, labels]
            inputs = xtb[i, :]
            inputs = inputs.view(-1, 1)
        
            target = ytb[i, :]
            target = target.view(-1, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    # test
    train_input  = xt.view(-1, 1)
    train_output = net(train_input)
    train_target = yt.view(-1, 1)
    train_error  = torch.sub(train_target, train_output)

    # convert to numpy
    train_input_np = np.squeeze(train_input.detach().numpy())
    train_output_np = np.squeeze(train_output.detach().numpy())

    plt.figure
    plt.scatter(xrand, yrand)
    plt.scatter(train_input_np, train_output_np, 5, 'r')
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    plt.axis('tight')
    plt.show()

    # option to save trained parameters
    if (yesno('Save neural net')):
        torch.save(net.state_dict(), SAVE_PATH)
        
else:
    net = Net()
    net.load_state_dict(torch.load(LOAD_PATH))

# ----------------------------------------------------------
# VALIDATE

# generate prediction w/ trained model
test_input  = xt2.view(-1, 1)
test_output = net(test_input)
test_target = yt2.view(-1, 1)
test_error  = torch.sub(test_target, test_output)

# convert to numpy
test_input_np = np.squeeze(test_input.detach().numpy())
test_output_np = np.squeeze(test_output.detach().numpy())

plt.figure
plt.scatter(xrand2, yrand2, label='True')
plt.scatter(test_input_np, test_output_np, 5, 'r', label='Predicted')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.title('Sine Function Approximation w/ 1-Layer Neural Net')
plt.axis('tight')
plt.legend()
plt.show()
