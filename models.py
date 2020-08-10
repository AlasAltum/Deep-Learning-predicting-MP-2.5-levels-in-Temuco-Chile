""" This module contains our models."""

import torch
from torch import nn, optim


class SimpleFCModel(nn.Module):
    """ Simple feedforward network of one single layer"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, input):
        predictions = self.fc(input)

        return predictions


class SimpleDoubleModel(nn.Module):
    """ Simple feedforward network of two fully connected layers"""
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input):
        hidden_preds = self.fc1(input)
        predictions = self.fc2(hidden_preds)

        return predictions


class ConvModel(nn.Module):
    """ """

    def __init__(self, input_size, output_size=1):
        super().__init__(self)

    def forward(self, input):
        pass


class LSTMModel(nn.Module):
    """LSTM layer + linear to get output"""

    def __init__(self, input_size, n_hidden=128, n_layers=3,  output_size=1, batch_size=1, dropout=0.2):
        super().__init__()

        self.n_hidden = n_hidden
        self.seq_len = batch_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=output_size)


    def reset_hidden_state(self):
        """Initialize hidden state"""
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
            )

    def forward(self, input):
            lstm_out, self.hidden = self.lstm(
                input.view(len(input), self.seq_len, -1),
                self.hidden
            )
            last_time_step = \
                lstm_out.view(self.seq_len, len(input), self.n_hidden)[-1]
            
            y_pred = self.linear(last_time_step)
            
            return y_pred


