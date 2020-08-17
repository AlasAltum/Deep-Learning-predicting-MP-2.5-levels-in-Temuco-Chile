from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, name, sequence_len):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # Batch x time sequence x time features
        self.fc1 = nn.Linear(hidden_size * sequence_len, num_classes)
        self.name = name

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward
        out, _ = self.rnn(x, h0)
        out = F.relu(out.reshape(out.shape[0], -1))
        out = self.fc1(out)
        return out


class MP25Dataset(Dataset):
    """Dataset para el proyecto"""

    def __init__(self, X, y, n, classifier=False):
        if classifier:
            self.X = [torch.from_numpy(x.values.reshape(1, n, 24)) for x in X]
            self.y = [torch.from_numpy(i.reshape(-1)) for i in y]
        else:
            self.X = [torch.from_numpy(x.values.reshape(1, n, 6)) for x in X]
            self.y = [torch.from_numpy(i.reshape(-1)) for i in y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx], self.y[idx]
        return sample

    def __repr__(self):
        return 'MP25Dataset'


def test(model, x_test, y_test, loss_function, batch_size=365 * 24):
    model.eval()
    total_loss = 0
    loss = {}
    loss['model_name'] = model.name
    loss['loss'] = []
    loss['epoch'] = []

    with torch.no_grad():
        for x_i, y_i in zip(x_test, y_test):
            # predict data using the given model
            prediction = model(x_i)
            # Compute loss
            total_loss += loss_function(prediction, y_i).item()

    print(total_loss)

    return total_loss


def train(model, train_set, optimizer, loss_function, device, epochs=5, batch_size=365 * 24, classifier=False):
    model.train()
    total_loss = 0

    for i in range(epochs):
        # each epoch
        epoch_loss = 0
        best_test_loss = float('inf')

        if classifier:
            for j in range(len(train_set)):
                # get the inputs; data is a list of [inputs, labels]
                x_i, y_i = train_set[j]

                x_i = x_i.to(device).float()
                y_i = y_i.to(device).float()
                optimizer.zero_grad()
                y_pred = model(x_i)

                loss = loss_function(y_pred.view(1, 10), torch.argmax(y_i).view(-1))
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
        else:
            for j in range(len(train_set)):
                # get the inputs; data is a list of [inputs, labels]
                x_i, y_i = train_set[j]

                x_i = x_i.to(device).float()
                y_i = y_i.to(device).float()
                optimizer.zero_grad()
                y_pred = model(x_i)

                loss = loss_function(y_pred, y_i)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

        # Save results from the best trained model
        if epoch_loss < best_test_loss:
            best_test_loss = epoch_loss
            torch.save(model.state_dict(), '{}.pt'.format(model.name))

        total_loss += epoch_loss
        print(f'epoch: {i} loss: {epoch_loss:10.8f}')
        loss['epoch'].append(i + 1)
        loss['loss'].append(epoch_loss)

    with open('{}_training.json'.format(model.name), 'w') as f:
        json.dump(loss, f)
    f.close()
    print(f'Average loss: {total_loss / len(train_set):4f}')
    return total_loss
