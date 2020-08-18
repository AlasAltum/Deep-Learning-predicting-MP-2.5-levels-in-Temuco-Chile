import pickle
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
import warnings
from sklearn.cluster import KMeans
from rnn_imports import RNN, train, test, MP25Dataset
from sklearn.preprocessing import KBinsDiscretizer
import sys


warnings.filterwarnings("ignore")
number_of_epochs = int(sys.argv[1])

# getting data
with open('data/data_product_03.pk', 'rb') as f:
    X, y = pickle.load(f)
f.close()

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_train = [i.mp_25 for i in y_train]
y_test = [i.mp_25 for i in y_test]

# making Datasets
training_set = MP25Dataset(X_train, y_train, 128)
test_set = MP25Dataset(X_test, y_test, 128)

# parameters for first rnn
input_size1 = 7
sequence_len1 = 128
num_layers1 = 2
hidden_size1 = 1000

# first rnn
rnn1 = RNN(input_size1, hidden_size1, num_layers1, 1, 'rnn1', sequence_len1)
rnn1.to("cuda")

# loss function and optimizer for first rnn
loss_func1 = torch.nn.MSELoss().to("cuda")
opt1 = optim.Adam(rnn1.parameters())

train(rnn1, training_set, opt1, loss_func1, epochs=number_of_epochs)


# Parameters for rnn 2
input_size2 = 7
sequence_len2 = 128
num_layers2 = 2
hidden_size2 = 1000

rnn2 = RNN(input_size2, hidden_size2, num_layers2, 10, 'rnn2', sequence_len2)
rnn2.to("cuda")

# loss function and optimizer for second rnn
loss_func = torch.nn.CrossEntropyLoss().to("cuda")
opt = optim.Adam(rnn2.parameters())


# preparing data for rnn2
discretizador = KBinsDiscretizer(10)
mp_25_vectors = discretizador.fit_transform(data[['mp_25']]).toarray()

# making discretization
for i, x in enumerate(X):
    yy = discretizador.transform(x[['mp_25']]).toarray()
    x.drop('mp_25', axis=1, inplace=True)
    for j in range(10):
        x['onehot_{}'.format(j)] = yy[:, j]
y = [discretizador.transform(i.mp_25.reshape(-1, 1)).toarray() for i in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# making Datasets for rnn2
training_set2 = MP25Dataset(X_train, y_train, 128, classifier=True)
test_set2 = MP25Dataset(X_test, y_test, 128, classifier=True)

# training rnn2
train(rnn2, training_set2, opt, loss_func, classifier=True, epochs=number_of_epochs)
