import torch
import torch.nn as nn


class RecurrentNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RecurrentNet, self).__init__()
        self.hidden_size = hidden_size
        # Input to hidden layer
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Hidden to output layer
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)  # For classification tasks
        self.hidden = self.init_hidden()

    def forward(self, input_tensor, hidden):
        # Combine input and previous hidden state
        combined = torch.cat((input_tensor, hidden), 1)
        # Pass through the input-to-hidden linear layer and apply activation
        hidden = torch.relu(self.i2h(combined))
        # Pass the current hidden state to the output layer
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        # Initialize the hidden state with zeros
        return torch.zeros(batch_size, self.hidden_size)
