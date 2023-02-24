import torch
import numpy as np
import torch.nn as nn
import pandas as pd

from model import Model
from utils import *

NUM_EPOCHS = 100
BATCH_SIZE = 10

# batch size, channels, length
# input_tensor = torch.randn(10, 1, 1000)

model = Model()

dataset = pd.read_csv('dataset.csv').to_numpy()

# Remove any rows with NaN values
dataset = dataset[~np.isnan(dataset).any(axis=1)]

# The last column is the class label
x, y = dataset[:, :-1], dataset[:, -1]

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

x = x.view(-1, 1, 1000)
y = y.view(-1, 1)

# Normalise the input
x = (x - x.mean()) / x.std()

# Split the data into test and train (shuffles the data)
x_test, x_train, y_test, y_train = split_dataset(x, y, test_proportion=0.2)

critereon = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_split_indices = k_fold_split(BATCH_SIZE, len(x_train))

# Test the model before training
y_pred = model(x_test)
accuracy = (y_pred.round() == y_test).sum().item() / len(y_test)
print(f"Accuracy before training: {accuracy}")

for epoch in range(NUM_EPOCHS):
    for i, batch_indices in enumerate(batch_split_indices):
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]
        y_pred = model(x_batch)
        loss = critereon(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

# Test the model after training
y_pred = model(x_test)
accuracy = (y_pred.round() == y_test).sum().item() / len(y_test)
print(f"Accuracy after training: {accuracy}")