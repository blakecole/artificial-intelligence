# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: dual_kinematic_nn1.py                             #
#    DATE: 27 NOV 2020                                       #
# ********************************************************** #

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# THIS MODEL USES A STATIC, PRE-TRAINED NN2 to TRAIN NN1


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
FWD_LOAD_PATH = './fwd_kinematic_nn.pth'

DUAL1_LOAD_PATH = './dual_kinematic_nn1.pth'
DUAL1_SAVE_PATH = './dual_kinematic_nn1.pth'

# ----------------------------------------------------------
# GENERATE TRAINING DATA & VALIDATION DATA

# -------- TRAINING DATA --------
batch_size = 32
N_batch = 200000
N = batch_size * N_batch

q1 = np.random.uniform(-np.pi, np.pi, (batch_size, 1, N_batch))
q2 = np.random.uniform(-np.pi, np.pi, (batch_size, 1, N_batch))

x1 = np.cos(q1) + np.cos(q2)
x2 = np.sin(q1) + np.sin(q2)

# data shape: [batch size x input dim x num batches]
x  = np.concatenate((x1,x2), axis=1)
q  = np.concatenate((q1, q2), axis=1)
print('\n')
print('N/N-1 Input (X) Dimensions  : ', x.shape)
print('N/N-1 Output (Q) Dimensions : ', q.shape)

# print('b1: [x1, x2]\n',x[:,:,0])
# print('b2: [x1, x2]\n',x[:,:,1])

# convert numpy arrays to torch tensors
xt = torch.from_numpy(x)
xt = xt.type(torch.float)
qt = torch.from_numpy(q)
qt = qt.type(torch.float)

print('\nBatch Size  = ', batch_size)
print('Num Batches = ', N_batch)
print('---------------------')
print('Total Data  = ', N)

print('\n N/N-1 Input (x_desired) Size: ', xt.size())
print('N/N-1 Output (q_predicted) Size: ', qt.size(), '\n')

# -------- VALIDATION DATA --------
vN = 200
vq1 = np.random.uniform(-np.pi, np.pi, vN)
vq2 = np.random.uniform(-np.pi, np.pi, vN)
vq  = np.stack((vq1, vq2), axis=1)

vx1 = np.cos(vq1) + np.cos(vq2)
vx2 = np.sin(vq1) + np.sin(vq2)
vx  = np.stack((vx1, vx2), axis=1)

# convert numpy arrays to torch tensors
vxt = torch.from_numpy(vx)
vxt = vxt.type(torch.float)
vqt = torch.from_numpy(vq)
vqt = vqt.type(torch.float)


# ----------------------------------------------------------
# DEFINE THE NETWORK

# 1 input
# 1 hidden layer with 120 neurons
# 1 output

# should be fine using the same structure for NN1 and NN2

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

net1 = Net()
print(net1)
params1 = list(net1.parameters())
print([p.size() for p in params1], '\n')

# load pre-trained weights and biases for N/N-2
net2 = Net()
print(net2)
net2.load_state_dict(torch.load(FWD_LOAD_PATH))
params2 = list(net2.parameters())
print([p.size() for p in params2], '\n')

# ----------------------------------------------------------
# DEFINE LOSS FUNCTION & OPTIMIZER

criterion = nn.MSELoss()       # define loss function
learning_rate = 0.01
momentum = 0
optimizer = optim.SGD(net1.parameters(), learning_rate, momentum,\
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
            input_nn1 = xt[:, :, i]

            # note: target is defined in terms of x, not q
            target_nn1 = xt[:, :, i]

            # zero the parameter gradients
            optimizer.zero_grad()

            # TRAINING: ------------------------------------
            # forward pass
            out_nn1 = net1(input_nn1)    # training

            # note: this time we do not have the luxury of feeding
            #       nn output (q) directly through the kinematic
            #       model to get (x_hat); rather, we must pass (q)
            #       into N/N-2 to generate (x_hat)
            out_nn2 = net2(out_nn1)

            # backward + optimize + loss
            loss = criterion(out_nn2, target_nn1)
            loss.backward()
            optimizer.step()

            # VALIDATION: ----------------------------------
            # forward pass
            vout_nn1 = net1(vxt)
            
            # feed output (vq) through N/N-2 to get (vx)
            vout_nn2 = net2(vout_nn1)
            
            # loss
            vloss = criterion(vout_nn2, vxt)

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
    input_nn1_batch_list = torch.split(xt, 1, dim=2)

    # merge batches into a single column of (1x2) rows of (x1,x2)
    input_nn1_stacked = torch.cat(input_nn1_batch_list, dim=0)

    # eliminate the empty batch dimension (dim=2)
    input_nn1_stacked = torch.squeeze(input_nn1_stacked,dim=2)

    print('reshape 3D batched data -> 2D unbatched...')
    print('    batched input size           : ', xt.size())
    print('    unbatched/stacked input size : ', input_nn1_stacked.size())

    # print('1st batch comparison:')
    # print('stacked (1st batch):\n', input_stacked[0:batch_size, :])
    # print('size: ', input_stacked[0:batch_size, :].size())
    
    # print('original (1st batch):\n',  torch.squeeze(xt[:,:,0]))
    # print('size: ', torch.squeeze(xt[:,:,0]).size())

    print('\nconfirm valid reshape...')
    if (torch.equal(input_nn1_stacked[0:batch_size,:],\
                    torch.squeeze(xt[:,:,0]))):
        print('VALID. (i.e. consistent up to first batch).   :)\n')
    else:
        print('WARNING!! Invalid Reshape.  Good f**king luck.\n')
        
        
    out_nn1 = net1(input_nn1_stacked)

    # feed output (q) through kinematic model to get (x)
    out_nn2 = net2(out_nn1)
    
    MAE = torch.sub(out_nn2, input_nn1_stacked)

    # convert to numpy
    input_nn1_stacked_np = input_nn1_stacked.detach().numpy()
    out_nn2_np = out_nn2.detach().numpy()

    # plot predicted vs. actual
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].scatter(input_nn1_stacked_np[:,0], out_nn2_np[:,0], 1, label='x_1')
    ax[0].set_title("x_1")
    ax[0].set(ylabel='predicted')

    ax[1].scatter(input_nn1_stacked_np[:,1], out_nn2_np[:,1], 1, label='x_2')
    ax[1].set_title("x_2")
    ax[1].set(xlabel='true', ylabel='predicted')
    fig.tight_layout()
    plt.show()

    # plot error for both training and validation
    plt.figure
    plt.plot(range(bcount), mean_loss, label='Training')
    plt.plot(range(bcount), mean_vloss, label='Validation')
    plt.title('N/N-1 Neural Net Training with static N/N-2')
    plt.ylabel('Aggregate-Mean MSE per Batch Group')
    plt.xlabel('Training Batch Groups (size = %d batches)' % test_interval)
    plt.axis('tight')
    plt.legend()
    plt.show()
    

    # option to save trained parameters
    if (yesno('Save neural net')):
        torch.save(net1.state_dict(), DUAL1_SAVE_PATH)
        
else:
    net = Net()
    net.load_state_dict(torch.load(DUAL1_LOAD_PATH))

# ----------------------------------------------------------
# VALIDATE

# generate prediction w/ trained model
vout_nn1 = net1(vxt)

# feed output (q) through kinematic model to get (x)
vout_nn2 = net2(vout_nn1)
    
vMAE = torch.sub(vout_nn2, vxt)

# convert predicted x values to numpy
vout_nn2_np = vout_nn2.detach().numpy()

plt.figure
plt.scatter(vx[:,0], vout_nn2_np[:,0], 1, label='x_1')
plt.scatter(vx[:,1], vout_nn2_np[:,1], 1, label='x_2')
plt.xlabel('x')
plt.ylabel('\hat{x}')
plt.title('Validation Data')
plt.axis('tight')
plt.legend()
plt.show()
