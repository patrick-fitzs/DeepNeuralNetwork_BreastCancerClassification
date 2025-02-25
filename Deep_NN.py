'''
Here we will be using PyTorch to build a simple neural network and using a dataset from the UCIML repository.
The data will be Breast Cancer Wisconsin (Diagnostic) Data Set' which is a binary classification dataset.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
import ucimlrepo
from IPython.display import display
from torch.utils.data import TensorDataset, DataLoader

# Here we will first import the dataset from the UCIML repository and check the structure.

from ucimlrepo import fetch_ucirepo

# breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
# here we extract the data and target from the dataset
# X = breast_cancer_wisconsin_diagnostic.data.features
# y = breast_cancer_wisconsin_diagnostic.data.targets

# downloaded for faster loading
X = pd.read_csv("features.csv")  # Load saved features
y = pd.read_csv("targets.csv")   # Load saved targets
# displays the first 5 rows of the dataset for both the features and the target(M or B)
display(X.head())
display(y.head())

display(f'X shape: {X.shape}')
display(f'y shape: {y.shape}')

# display the counts of the target, i.e. the number of Malignant and Benign
display(y['Diagnosis'].value_counts())

# here we can see the data set is imbalanced
# we will process the data. Randomly choose 200 samples in 'M' and 'B' each

# combine features and targets into a single dataframe for easier manipulation
data = pd.concat([X, y], axis=1)

data_B = data[data['Diagnosis'] == 'B']
data_M = data[data['Diagnosis'] == 'M']

# select 200 from each class
data_B = data_B.sample(n=200, random_state=42)
data_M = data_M.sample(n=200, random_state=42)

# combine
balanced_data = pd.concat([data_B, data_M])
display(balanced_data['Diagnosis'].value_counts())

###### Data preprocessing ######
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# this separates true features and targets
X = balanced_data.drop('Diagnosis', axis=1)
y = balanced_data['Diagnosis']

# turns the targets into binary labels
y = y.map({'B': 0, 'M': 1})

display(X)
display(y)

# Now we split the data into training and testing sets, 80/20 split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42, stratify=y)

display(f'X_train shape: {X_train.shape}')
display(f'y_train shape: {y_train.shape}')
display(f'X_test shape: {X_test.shape}')
display(f'y_test shape: {y_test.shape}')

'''
Now we will standardise the data using StandardScaler from sklearn.  This will scale the data to have a mean of 0 and a standard deviation of 1.
This is important as it will help the model to converge faster.
#1 We use StandardScaler to calculate the mean and the standard deviation of the data. 
#2 We then transform the data using transform which scales the data accordingly
#3 Lastly we apply the same transformation to the test data to ensure both the training and test data are scaled the same way.
'''
scaler = StandardScaler()
# fit the scaler to the training data
X_train = scaler.fit_transform(X_train)
# transform the training data using the same scaler
X_test = scaler.transform(X_test)

# Now convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# create DataLoader for training and test sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

###### Building the model ######
# here we will define our neural network architecture, specify the loss function and optimiser. Then train the model.

import torch.nn as nn

class ClassificationNet(nn.Module):
    def __init__(self, input_units=30, hidden_units=64, output_units=2):
        super(ClassificationNet, self).__init__()
        # below are fully connected layer which means each neuron in one layer is connected to every neuron in the next layer
        # input units are the features in my dataset and hidden units determine the capacity of the network.
        # More hidden units can learn more complex patterns but can also lead to overfitting, Fewer can lead to underfitting
        self.fc1 = nn.Linear(input_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ClassificationNet(input_units=30, hidden_units=64, output_units=2)

# now we define the loss function and the optimiser.
# We will use cross entropy loss and the Adam optimiser(latter is used to update the weights of the neural network during training.)

import torch.optim as optim
# define the loss function and optimiser
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
'''
#### An alternate optimiser. SGD ####
'''
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay=0.0001)




'''
### Training the model ###

# now we will iterate over the training data and update the weights of the model using backpropagation.
During the trainin we will calculate the loss and accuracy of the model. The loss should decrease as the model learns the patterns in the data.
Lastly we evaluate the model on the test data to see how well it generalises to unseen data.
'''
epochs = 10
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Now eval on the test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            test_outputs = model(X_batch)
            loss = criterion(test_outputs, y_batch)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

'''
Now we will visualise the training and test loss to see how the model is learning.
'''
plt.figure(figsize=(10,6))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, epochs+1), test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss curve')
plt.legend()
plt.grid(True)
plt.show()
