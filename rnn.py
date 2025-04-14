import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFeedforwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFeedforwardNet, self).__init__()
        # Input layer to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Hidden layer to output layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass the input through the first fully connected layer and apply ReLU activation
        x = F.sigmoid(self.fc1(x))
        # Pass the output of the hidden layer through the second fully connected layer
        x = self.fc2(x)
        return x
