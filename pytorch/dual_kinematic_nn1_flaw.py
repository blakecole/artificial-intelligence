# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: MIT                                               #
#    FILE: dual_kinematic_nn1_flaw.py                        #
#    DATE: 27 NOV 2020                                       #
# ********************************************************** #

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# THIS MODEL USES A STATIC, PRE-TRAINED NN2 to TRAIN NN1

# FURTHER THERE IS SOME MODEL UNCERTAINTY WITH RESPECT TO:
#  1) base position, dxb
#  2) arm length, dl
#  3) joint angle, dq

# HOWEVER THERE ARE TWO CASES TO CONSIDER
#  1) NN2 IS TRAINED USING THE NOMINAL MODEL, AND
#     ONLY NN1 IS TRAINED WITH THE ``TRUE (BIASED) MODEL''
#     --> for this scnario set:
#         FWD_LOAD_PATH = './fwd_kinematic_nn.pth'
#
#  2) NN2 IS TRAINED USING TRUE DATA AS WELL
#     --> for this scenario set:
#         FWD_LOAD_PATH = './fwd_kinematic_nn_biased.pth'

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
FWD_LOAD_PATH = './fwd_kinematic_nn_biased.pth'

DUAL1_LOAD_PATH = './dual_kinematic_nn1_flaw.pth'
DUAL1_SAVE_PATH = './dual_kinematic_nn1_flaw.pth'

# ----------------------------------------------------------
# GENERATE TRAINING DATA & VALIDATION DATA

# -------- TRAINING DATA --------
batch_size = 32
N_batch = 200000
N = batch_size * N_batch

# define model biases/offsets -- ``true plant model''
# base position offsets
# dxb1 = 0.1
# dxb2 = 0.1

# large error run
dxb1 = 1
dxb2 = 1

# arm length offsets (5% error)
#dl1 = 0.05
#dl2 = 0.05

# large error run
dl1 = 0.3
dl2 = -0.2

L1 = 1 + dl1
L2 = 1 + dl2

# angle offsets (0.05 rad == 2.9 degree bias)
#dq1 = 0.05
#dq2 = 0.05

# large error run (0.2 rad == 11.5 degree bias)
dq1 = 0.2
dq2 = -0.2

q1 = np.random.uniform(-np.pi, np.pi, (batch_size, 1, N_batch))
q2 = np.random.uniform(-np.pi, np.pi, (batch_size, 1, N_batch))

x1 = np.cos(q1) + np.cos(q2)
x2 = np.sin(q1) + np.sin(q2)

X1 = L1*np.cos(q1+dq1) + L2*np.cos(q2+dq2) + dxb1
X2 = L1*np.sin(q1+dq1) + L2*np.sin(q2+dq2) + dxb2

# data shape: [batch size x input dim x num batches]
x = np.concatenate((x1,x2), axis=1)
X = np.concatenate((X1,X2), axis=1)
q = np.concatenate((q1, q2), axis=1)
print('\n')
print('N/N-1 Input (x) Dimensions   : ', x.shape)
print('N/N-1 Output (q) Dimensions  : ', q.shape)
print('True Position (X) Dimensions : ', X.shape)


# convert numpy arrays to torch tensors
xt = torch.from_numpy(x)
xt = xt.type(torch.float)
Xt = torch.from_numpy(X)
Xt = Xt.type(torch.float)
qt = torch.from_numpy(q)
qt = qt.type(torch.float)

print('\nBatch Size  = ', batch_size)
print('Num Batches = ', N_batch)
print('---------------------')
print('Total Data  = ', N)

print('\n N/N-1 Input (x_desired) Size: ', xt.size())
print('N/N-1 Output (q_predicted) Size: ', qt.size(), '\n')
print('True Position (X) Size: ', Xt.size(), '\n')

# -------- VALIDATION DATA --------
vN = 200
vq1 = np.random.uniform(-np.pi, np.pi, vN)
vq2 = np.random.uniform(-np.pi, np.pi, vN)
vq  = np.stack((vq1, vq2), axis=1)

vx1 = np.cos(vq1) + np.cos(vq2)
vx2 = np.sin(vq1) + np.sin(vq2)
vx  = np.stack((vx1, vx2), axis=1)

vX1 = L1*np.cos(vq1+dq1) + L2*np.cos(vq2+dq2) + dxb1
vX2 = L1*np.sin(vq1+dq1) + L2*np.sin(vq2+dq2) + dxb2
vX  = np.stack((vX1, vX2), axis=1)

# convert numpy arrays to torch tensors
vxt = torch.from_numpy(vx)
vxt = vxt.type(torch.float)
vXt = torch.from_numpy(vX)
vXt = vXt.type(torch.float)
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
net2.load_state_dict(torch.load(FWD_LOAD_PATH))
print(net2)
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

            # note: target is defined in terms of ``true'' position!
            target = Xt[:, :, i]

            # zero the parameter gradients
            optimizer.zero_grad()

            # TRAINING: ------------------------------------
            # forward pass
            out_nn1 = net1(input_nn1)

            # note: this time we do not have the luxury of feeding
            #       nn output (q) directly through the kinematic
            #       model to get (x_hat); rather, we must pass (q)
            #       into N/N-2 to generate (x_hat)
            out_nn2 = net2(out_nn1)

            # backward + optimize + loss
            loss = criterion(out_nn2, target)
            loss.backward()
            optimizer.step()

            # VALIDATION: ----------------------------------
            # forward pass
            vout_nn1 = net1(vxt)
            
            # feed output (vq) through N/N-2 to get (vx)
            vout_nn2 = net2(vout_nn1)
            
            # loss -- note!  compare to ``true'' measured position
            vloss = criterion(vout_nn2, vXt)

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
    x_batch_list = torch.split(xt, 1, dim=2)
    X_batch_list = torch.split(Xt, 1, dim=2)

    # merge batches into a single column of (1x2) rows of (x1,x2)
    x_stacked = torch.cat(x_batch_list, dim=0)
    X_stacked = torch.cat(X_batch_list, dim=0)

    # eliminate the empty batch dimension (dim=2)
    x_stacked = torch.squeeze(x_stacked,dim=2)
    X_stacked = torch.squeeze(X_stacked,dim=2)

    print('reshape 3D batched data -> 2D unbatched...')
    print('    batched input size                     : ', xt.size())
    print('    unbatched/stacked input size           : ', x_stacked.size())

    print('    batched ``true position" size           : ', Xt.size())
    print('    unbatched/stacked ``true position" size : ', X_stacked.size())

    # print('1st batch comparison:')
    # print('stacked (1st batch):\n', input_stacked[0:batch_size, :])
    # print('size: ', input_stacked[0:batch_size, :].size())
    
    # print('original (1st batch):\n',  torch.squeeze(xt[:,:,0]))
    # print('size: ', torch.squeeze(xt[:,:,0]).size())

    print('\nconfirm valid reshape...')
    if (torch.equal(x_stacked[0:batch_size,:],\
                    torch.squeeze(xt[:,:,0]))):
        print('VALID (x)  :)\n')
    else:
        print('WARNING!! Invalid Reshape (x)   :(\n')

    if (torch.equal(X_stacked[0:batch_size,:],\
                    torch.squeeze(Xt[:,:,0]))):
        print('VALID (X)  :)\n')
    else:
        print('WARNING!! Invalid Reshape (X)   :(\n')
        
        
    out_nn1 = net1(x_stacked)

    # feed output (q) through kinematic model to get (x)
    out_nn2 = net2(out_nn1)

    # convert to numpy
    x_stacked_np = x_stacked.detach().numpy()
    X_stacked_np = X_stacked.detach().numpy()
    out_nn2_np = out_nn2.detach().numpy()

    # plot predicted vs. actual
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].scatter(X_stacked_np[:,0], out_nn2_np[:,0], 1, label='x_1')
    ax[0].set_title("x_1 Predicted vs. Observed")
    ax[0].set(ylabel='Predicted Position')

    ax[1].scatter(X_stacked_np[:,1], out_nn2_np[:,1], 1, label='x_2')
    ax[1].set_title("x_2 Predicted vs. Observed")
    ax[1].set(xlabel='Observed Position', ylabel='Predicted Postion')
    fig.tight_layout()
    plt.show()

    # plot error for both training and validation
    plt.figure
    plt.plot(range(bcount), mean_loss, label='Training')
    plt.plot(range(bcount), mean_vloss, label='Validation')
    plt.title('N/N-1 Neural Net Training on Biased Plant with static N/N-2')
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

# convert predicted x values to numpy
vout_nn2_np = vout_nn2.detach().numpy()

plt.figure
plt.scatter(vX[:,0], vout_nn2_np[:,0], 1, label='x_1')
plt.scatter(vX[:,1], vout_nn2_np[:,1], 1, label='x_2')
plt.xlabel('Observed Position [x]')
plt.ylabel('Estimated Position [x]')
plt.title('Validation Data for Biased Plant Model')
plt.axis('tight')
plt.legend()
plt.show()
