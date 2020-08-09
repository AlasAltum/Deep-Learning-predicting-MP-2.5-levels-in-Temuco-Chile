""" This module contains our models."""

import torch
from torch import nn, optim


class SimpleFCModel(nn.Module):
    """ Simple feedforward network of one single layer"""

    def __init__(self, input_size, output_size):
        super().__init__(self)
        self.fc = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, input):
        predictions = self.fc(input)

        return predictions


class SimpleDoubleModel(nn.Module):
    """ Simple feedforward network of one single layer"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(self)
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, input):
        hidden_preds = self.fc1(input)
        predictions = self.fc2(hidden_preds)

        return predictions


class ConvModel(nn.Module):
    """ """

    def __init__(self, input_size, output_size):
        super().__init__(self)

