import os
import numpy as np
import pandas as pd
from h11 import Data
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
import torch.nn.functional as F
import matplotlib.pyplot as plt

def decorate_axis(ax, remove_left=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def accuracy(output_1, target_1):
    #bitwise and is not supported in pytorch atm so we use multiplication and sum
    output_1 = torch.round(output_1).cpu().numpy().astype(int)
    target_1 = target_1.cpu().numpy().astype(int)
    sum_1 = np.sum(np.bitwise_and(output_1, target_1))

    sum_2 = np.sum(np.bitwise_or(output_1, target_1))
    # print(target_1/, output_1, sum_2)
    accur =  sum_1 / sum_2
    if(sum_2 == sum_1):
        accur = 1

    return accur
def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class CustomDataset(Dataset):
    """ Custom  Dataset. """

    def __init__(self, x, y):
        """
        Args:
            x: np array containing input features
            y: label vectors
        """

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#ANN
class Net(nn.Module):
  def __init__(self,input_size, H_1, H_2, classes):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_size,  H_1)
    self.fc2 = nn.Linear(H_1, H_2)
    self.fc3 = nn.Linear(H_2, classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x


# Configuration options and hyperparameters
k_folds = 5
num_epochs = 1000
lr = 1e-1
m =  0.6
loss_function = nn.CrossEntropyLoss()
# loss_function = nn.MSELoss()

output_s = 20


# For fold results
results = {}

# Set fixed random number seed
torch.manual_seed(42)

# Prepare dataset by concatenating Train/Test part; we split later.

data_train = np.load('data/train.npz',allow_pickle=True)

data_test = np.load('data/test.npz',allow_pickle=True)
# drop columns that corrrespond to insifnificant features (non selected genes from ga)
# read array
selected = np.load("data/8.npy")
ids = np.where(np.array(selected)==1)[0][:-1]
x_train = data_train['X']
x_test =  data_test['X']
print(x_train.shape)
print(x_test.shape)
x_train = np.delete(x_train,ids,1)
x_test = np.delete(x_test,ids,1)
input_s = x_train.shape[1]
hidden_1_s = (output_s + input_s)
hidden_2_s = (2*hidden_1_s)
data_X = np.vstack((x_train, x_test))
data_Y = np.vstack((data_train['Y'], data_test['Y']))


dataset = CustomDataset(data_X, data_Y)

# Define the K-fold Cross Validator
# set shuffle to false to use same kfold in every method
kfold = KFold(n_splits=k_folds, shuffle=False)
# Start print
print('--------------------------------')


all_losses = []

losses = []
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader( dataset, batch_size=256, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader( dataset, batch_size=256, sampler=test_subsampler)

    # Init the neural network
    network = Net(input_s, hidden_1_s, hidden_2_s, output_s).to(device)
    network.apply(reset_weights)

    # Initialize optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum = m)

    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
        # Set current loss value
        current_loss = 0.0
        batches_per_epoch = data_X.shape[0]//256
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass

            outputs = network(inputs.to(device))
            # Compute loss
            loss = loss_function(outputs, targets.to(device))

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
        train_mean_loss = current_loss/batches_per_epoch
        losses.append(train_mean_loss)
        print(" Epoch: ", epoch+1, "Loss: ", train_mean_loss)


    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')

    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)

    # Evaluation for this fold
    acc, total = 0, 0
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):

            # Get inputs
            inputs, targets = data

            # Generate outputs
            outputs = network(inputs.to(device))

            # Set total and correct
            for j in range(0, outputs.shape[0]):

                acc += accuracy(outputs[j], targets[j].to(device))

                total+=1
            #correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * acc / total))
        print('--------------------------------')
        results[fold] = 100.0 * (acc / total)

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum/len(results.items())} %')


losses_2 = np.array(losses).reshape(k_folds,num_epochs)
losses_2 =np.mean(losses_2, axis = 0)
all_losses.append(losses_2)






from matplotlib.ticker import MaxNLocator
x = np.arange(num_epochs)
fig = plt.figure()
ax0 = fig.add_subplot()
ax0.set_title("CE Loss for different lr and momentum")
ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epoch', fontsize=8)
plt.ylabel('Loss', fontsize=8)
decorate_axis(ax0)
h_1 = plt.plot(x,all_losses[0], label = "lr = 1e-3 m = 0.2")
h_2 = plt.plot(x,all_losses[1], label = "lr = 1e-3 m = 0.6")
h_3 = plt.plot(x,all_losses[2], label = "lr = 5e-2 m = 0.6")
h_4 = plt.plot(x,all_losses[3], label = "lr = 1e-1 m = 0.6")
lgnd = ax0.legend(handles=[h_1[0], h_2[0], h_3[0], h_4[0]],loc = "upper right")
fig.savefig('ce_lr.png')