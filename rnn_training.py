import pickle
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta
import warnings
from preprocessing import get_consecutive
from sklearn.cluster import KMeans
from rnn_imports import RNN
from sklearn.preprocessing import KBinsDiscretizer
warnings.filterwarnings("ignore")
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# abriendo datos
with open('./data/data_product_02.pk', 'rb') as f:
    data = pickle.load(f)
f.close()

# standarization of the data
data_with_clusters = data.copy()
data_with_clusters['mp_25'] = data['mp_25']

# making clusters
for i in range(2, 11):
    kmeans = KMeans(i)
    kmeans.fit(data_with_clusters)
    data_with_clusters['{}_clusters'.format(i)] = kmeans.labels_

# getting consecutive dataframes of length 128
X, y = get_consecutive(data, 128)

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_train = [i.mp_25 for i in y_train]
y_test = [i.mp_25 for i in y_test]

# making Datasets
training_set = MP25Dataset(X_train, y_train, 128)
test_set = MP25Dataset(X_test, y_test, 128)

# parameters for first rnn
input_size = 6
sequence_len = 128
num_layers = 1
hidden_size = 1000
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# first rnn
rnn1 = RNN(input_size, hidden_size, num_layers, 1, 'rnn1')
rnn1.to(device)

# loss function and optimizer for first rnn
loss_func1 = torch.nn.MSELoss().to(device)
opt1 = optim.Adam(rnn1.parameters())

train(rnn1, training_set, opt1, loss_func1)


# Parameters for rnn 2
input_size2 = 24
sequence_len2 = 128
num_layers2 = 1
hidden_size2 = 1000
learning_rate2 = 0.001
batch_size2 = 64
num_epochs2 = 2

rnn2 = RNN(input_size2, hidden_size2, num_layers2, 10, 'rnn2')
rnn2.to(device)

# loss function and optimizer for second rnn
loss_func = torch.nn.CrossEntropyLoss().to(device)
opt = optim.Adam(rnn2.parameters())


# preparing data for rnn2
discretizador = KBinsDiscretizer(10)
mp_25_vectors = discretizador.fit_transform(data[['mp_25']]).toarray()
X, y = get_consecutive(data_with_clusters, 128)

# making discretization
for i, x in enumerate(X):
    yy = discretizador.transform(x[['mp_25']]).toarray()
    x.drop('mp_25', axis=1, inplace=True)
    for j in range(10):
        x['onehot_{}'.format(j)] = yy[:, j]
y = [discretizador.transform(i.mp_25.reshape(-1, 1)).toarray() for i in y]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# making Datasets for rnn2
training_set2 = MP25Dataset2(X_train, y_train, 128, classifier=True)
test_set2 = MP25Dataset2(X_test, y_test, 128, classifier=True)

# training rnn2
train(rnn2, training_set2, opt, loss_func, classifier=True)
