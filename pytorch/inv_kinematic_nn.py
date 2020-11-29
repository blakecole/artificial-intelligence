# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: kinematic_nn.py                                   #
#    DATE: 27 NOV 2020                                       #
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
LOAD_PATH = './kinematic_nn.pth'
SAVE_PATH = './kinematic_nn.pth'

# ----------------------------------------------------------
# GENERATE TRAINING DATA & VALIDATION DATA

# -------- TRAINING DATA --------
batch_size = 32
N_batch = 100000
N = batch_size * N_batch

q1 = np.random.uniform(-np.pi, np.pi, (batch_size, 1, N_batch))
q2 = np.random.uniform(-np.pi, np.pi, (batch_size, 1, N_batch))

x1 = np.cos(q1) + np.cos(q2)
x2 = np.sin(q1) + np.sin(q2)

# data shape: [batch size x input dim x num batches]
x  = np.concatenate((x1,x2), axis=1)
y  = np.concatenate((q1, q2), axis=1)
print('Input (X) Dimensions: ', x.shape)
print('Output (Q) Dimensions: ', y.shape)

# print('b1: [x1, x2]\n',x[:,:,0])
# print('b2: [x1, x2]\n',x[:,:,1])

# convert numpy arrays to torch tensors
xt = torch.from_numpy(x)
xt = xt.type(torch.float)
yt = torch.from_numpy(y)
yt = yt.type(torch.float)

print('\nBatch Size  = ', batch_size)
print('Num Batches = ', N_batch)
print('---------------------')
print('Total Data  = ', N)

print('\nInput (x) Size: ', xt.size())
print('Target (y) Size: ', yt.size(), '\n')

# -------- VALIDATION DATA --------
N2 = 200
q1_2 = np.random.uniform(-np.pi, np.pi, N2)
q2_2 = np.random.uniform(-np.pi, np.pi, N2)
y2  = np.stack((q1_2, q2_2), axis=1)

x1_2 = np.cos(q1_2) + np.cos(q2_2)
x2_2 = np.sin(q1_2) + np.sin(q2_2)
x2  = np.stack((x1_2,x2_2), axis=1)

# convert numpy arrays to torch tensors
xt2 = torch.from_numpy(x2)
xt2 = xt2.type(torch.float)
yt2 = torch.from_numpy(y2)
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
        self.fc1 = nn.Linear(in_features = 2,\
                             out_features = 120,\
                             bias=True)

        self.fc2 = nn.Linear(in_features = 120,\
                             out_features = 2,\
                             bias=True)

        
    def forward(self, x):
        #x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
    # def num_flat_features(self, x):

        # all dimensions besides batch dimension
        # size = x.size()[1:]

        # num_features = 1
        # for s in size:
          #  num_features *= s
        # return num_features

net = Net()
print(net)


# ----------------------------------------------------------
# QUERY THE NETWORK

params = list(net.parameters())

# return size of each parameter
print([p.size() for p in params], '\n')

# ----------------------------------------------------------
# DEFINE LOSS FUNCTION & OPTIMIZER

criterion = nn.MSELoss()       # define loss function
learning_rate = 0.01
momentum = 0
optimizer = optim.SGD(net.parameters(), learning_rate, momentum,\
                      dampening=0, weight_decay=0, nesterov=False)


# ----------------------------------------------------------
# TRAIN PARAMETERS / LOAD  PARAMETERS
N_epochs = 2;
test_interval = min([N, 1000])    # integer number of batches

# allocate loss arrays
mean_loss  = np.empty([int(N_epochs*N_batch/test_interval), 1])
mean_vloss = np.empty([int(N_epochs*N_batch/test_interval), 1])

bcount = 0

if (yesno('Train neural net')):
    for epoch in range(N_epochs):  # loop over the dataset multiple times
        training_running_loss = 0.0
        validation_running_loss = 0.0
        for i in range(N_batch):
            
            # batch is 32 rows of 1x2 input data [x1, x2]
            inputs = xt[:, :, i]

            # note: target is defined in terms of x, not q
            x_target = xt[:, :, i]

            # zero the parameter gradients
            optimizer.zero_grad()

            # TRAINING: ------------------------------------
            # forward pass
            q_out = net(inputs)    # training

            # feed output (q) through kinematic model to get (x)
            x1_out = torch.add(torch.cos(q_out[:,0]), torch.cos(q_out[:,1]))
            x2_out = torch.add(torch.sin(q_out[:,0]), torch.sin(q_out[:,1]))
            x_out = torch.stack((x1_out, x2_out), dim=1)

            # backward + optimize + loss
            loss = criterion(x_out, x_target)
            loss.backward()
            optimizer.step()

            # VALIDATION: ----------------------------------
            # forward pass
            vq_out = net(xt2)
            
            # feed output (vq) through kinematic model to get (vx)
            vx1_out = torch.add(torch.cos(vq_out[:,0]),\
                                torch.cos(vq_out[:,1]))
            vx2_out = torch.add(torch.sin(vq_out[:,0]),\
                                torch.sin(vq_out[:,1]))
            vx_out = torch.stack((vx1_out, vx2_out), dim=1)

            # loss
            vloss = criterion(vx_out, xt2)

            # CALCULATE MEAN STATS: -----------------------
            # print statistics, and test validation every 1000 batches
            training_running_loss += loss.item()
            validation_running_loss += vloss.item()
            
            if (i % test_interval == test_interval-1):
                # mean training loss
                mean_loss[bcount] = training_running_loss / test_interval
                training_running_loss = 0.0

                # mean validation loss
                mean_vloss[bcount] = validation_running_loss / test_interval
                validation_running_loss = 0.0

                # print stats
                print('[%d, %5d] loss: %.6f / %.6f' %
                      (epoch + 1, i + 1, mean_loss[bcount], mean_vloss[bcount]))
                bcount += 1

    print('Finished Training\n\n')

    # TEST WITH ENTIRE DATASET AND PLOT RESULTS ------------

    print('test on training data...')
    # restructure batched data for one-shot testing
    # split input data into list of batches
    input_batch_list = torch.split(xt, 1, dim=2)

    # merge batches into a single column of (1x2) rows of (x1,x2)
    input_stacked = torch.cat(input_batch_list, dim=0)

    # eliminate the empty batch dimension (dim=2)
    input_stacked = torch.squeeze(input_stacked,dim=2)

    print('reshape 3D batched data -> 2D unbatched...')
    print('    batched input size           : ', xt.size())
    print('    unbatched/stacked input size : ', input_stacked.size())

    # print('1st batch comparison:')
    # print('stacked (1st batch):\n', input_stacked[0:batch_size, :])
    # print('size: ', input_stacked[0:batch_size, :].size())
    
    # print('original (1st batch):\n',  torch.squeeze(xt[:,:,0]))
    # print('size: ', torch.squeeze(xt[:,:,0]).size())

    print('\nconfirm valid reshape...')
    if (torch.equal(input_stacked[0:batch_size,:], torch.squeeze(xt[:,:,0]))):
        print('VALID. (i.e. consistent up to first batch).   :)\n')
    else:
        print('WARNING!! Invalid Reshape.  Good f**king luck.\n')
        
        
    q_out = net(input_stacked)

    # feed output (q) through kinematic model to get (x)
    x1_out = torch.add(torch.cos(q_out[:,0]), torch.cos(q_out[:,1]))
    x2_out = torch.add(torch.sin(q_out[:,0]), torch.sin(q_out[:,1]))
    x_out = torch.stack((x1_out, x2_out), dim=1)
    
    train_MAE = torch.sub(x_out, input_stacked)

    # convert to numpy
    input_stacked_np = input_stacked.detach().numpy()
    x_out_np = x_out.detach().numpy()

    # plot predicted vs. actual
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].scatter(input_stacked_np[:,0], x_out_np[:,0], 1, label='x_1')
    ax[0].set_title("x_1")
    ax[0].set(ylabel='predicted')

    ax[1].scatter(input_stacked_np[:,1], x_out_np[:,1], 1, label='x_2')
    ax[1].set_title("x_2")
    ax[1].set(xlabel='true', ylabel='predicted')
    fig.tight_layout()
    plt.show()

    # plot error for both training and validation
    plt.figure
    plt.plot(range(bcount), mean_loss, label='Training')
    plt.plot(range(bcount), mean_vloss, label='Validation')
    plt.xlabel('Training Progress [%d-batch groups]' % test_interval)
    plt.ylabel('Aggregate-Mean MSE')
    plt.title('Neural Net Training')
    plt.axis('tight')
    plt.legend()
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
test_q_out = net(xt2)

# feed output (q) through kinematic model to get (x)
test_x1_out = torch.add(torch.cos(test_q_out[:,0]), torch.cos(test_q_out[:,1]))
test_x2_out = torch.add(torch.sin(test_q_out[:,0]), torch.sin(test_q_out[:,1]))
test_x_out = torch.stack((test_x1_out, test_x2_out), dim=1)
    
test_MAE = torch.sub(test_x_out, xt2)

# convert predicted x values to numpy
test_x_out_np = test_x_out.detach().numpy()

plt.figure
plt.scatter(x2[:,0], test_x_out_np[:,0], 1, label='x_1')
plt.scatter(x2[:,1], test_x_out_np[:,1], 1, label='x_2')
plt.xlabel('x')
plt.ylabel('\hat{x}')
plt.title('Validation Data')
plt.axis('tight')
plt.legend()
plt.show()
