# import necessary modules
import mxnet as mx
#import mne
import matplotlib.pyplot as plt
import numpy as np
from mxnet import autograd, nd, gluon
import math
from scipy import signal, stats
import math
import scipy
import sys
from time import time


# set context for data and model
# select gpu by default
# opt for cpu only when no gpu is available
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx



def read_files(file):
    """Read a .npy file and convert it to a numpy array."""
    return np.load(file, mmap_mode = 'r')


x_file_list_train = ['/project/6019337/cosmin14/phase_5_pre_processed/x' + f'{i}_{i+200000}' + '.npy' for i in range(0, 2200000, 100000)]
y_file_list_train = ['/project/6019337/cosmin14/phase_5_pre_processed/y' + f'{i}_{i+200000}' + '.npy' for i in range(0, 2200000, 100000)]


def create_array(x, y, n, m):
    """Splits the data into smaller segments for training.

    Parameters:
    -----------
    x : numpy array
        The file containing predictor variable
    y : numpy array
        The file containing target variable
    n : int
        Length of each selected segment
    m : int
        Number of selected segments per patient

    """

    xx = []
    yy = []

    # loop over all patients
    for i in range(len(x)):

        # select current patient
        a = x[i]
        b = y[i]

        # check for null ages
        if not math.isnan(b):
            for j in range(0, m*n, n):

                u = a[:,:,j:(j+n)]

                # normalize each segment
                u = stats.zscore(u, axis=1)

                # add it to the list
                xx.append(u)
                yy.append(b)
                del u
        del a
        del b

    return xx, yy


def expand(x, t = 0):
    """Expand dimension of a list of arrays along a given axis.

    Parameters:
    -----------
    x : list
        Contains arrays whose dimension will be expanded
    t : int
        Dimension along which expansion will be made
    """
    # loop over each array
    for i in range(len(x)):

        # expand dimension
        x[i] = np.expand_dims(x[i], t)

    return x


def train_test(x, y, p = 0.7, batch_size = 50):
    """Split the data into train-test subsets.

    Parameters:
    -----------
    x : list or numpy array
        Contains predictor variable
    y : list or numpy array
        Contains target variable
    p : int, optional
        Percentage of data to be used for training
    batch_size  : int, optional
        Size of each batch
    """
    # total number of data points
    n = len(x)

    # number of data points for training
    n = int(p * n)

    # perform split
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x[:n], y[:n]), batch_size = batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x[n:], y[n:]), batch_size = batch_size, shuffle=True)

    return train_data, test_data


def prediction(data_iterator, n = 0):
    """Display the predictions of the current network for a given batch

    Parameters:
    -----------
    data_iterator : dataloader
                    Contains multiple batches
    n : int, optional
        Position of the batch to be considered for prediction
    """
    for i, (data, label) in enumerate(data_iterator):
        if i == n:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            output = net(data)
            return output
            break


x = []
y = []
x_test = []
y_test = []



for x_file, y_file in zip(x_file_list_train, y_file_list_train):

    p = read_files(x_file)
    q =  read_files(y_file)

    a, b = create_array(p, q, 500, 1)

    x.extend(a)
    y.extend(b)

    del a
    del b


for x_file, y_file in zip(x_file_list_test, y_file_list_test):

    p = read_files(x_file)
    q =  read_files(y_file)

    for i in range(len(p)):

        # select current patient
        a = p[i]
        b = q[i]

        # check for null ages
        if not math.isnan(b):
            for j in range(0, 500, 500):

                u = a[:,:,j:(j+500)]

                # normalize each segment
                u = stats.zscore(u, axis=1)

                # add it to the list
                x_test.append(u)
                y_test.append(b)
                del u

        del a
        del b


train_data, test_data = train_test(x, y)


def evaluate_error(data_iterator, net):
    """Evaluate error of a model for a specific metric.

    Parameters:
    ----------
    data_iterator: dataloader
                   Contains the batches to be evaluated
    net : mxnet neural network
          Model to be assessed
    metric: mx object, optional
          The metric with respect to which error is computed
    """
    err = mx.metric.MAE()

    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        err.update(preds=output, labels=label)
    return err.get()[1]


def f1():

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.ELU())
        net.add(gluon.nn.Conv2D(channels= 16, kernel_size=(3, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(5, 2), strides = (1, 1), activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(3, 5), strides = (1, 1), activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 0))

        #net.add(gluon.nn.BatchNorm())

        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(5, 2), strides = (1, 1), activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(3, 5), strides = (1, 1), activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 0))
        #net.add(gluon.nn.AvgPool2D(pool_size=(2, 2), strides=(1, 1), padding = 1))

        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(300, activation = "relu"))
        net.add(gluon.nn.Dense(200, activation = "relu"))
        net.add(gluon.nn.Dense(100, activation = "relu"))
        net.add(gluon.nn.Dense(50, activation = "relu"))
        net.add(gluon.nn.Dense(25, activation = "relu"))
        net.add(gluon.nn.Dense(10, activation = "relu"))
        net.add(gluon.nn.Dense(5, activation = "relu"))
        net.add(gluon.nn.Dense(1))
        return net


def f2():

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(3, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(3, 3), strides = (1, 1), activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 256, kernel_size=(3, 3), strides = (1, 1), activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(64, activation = None))
        net.add(gluon.nn.Dropout(0.5))
        net.add(gluon.nn.Dense(32, activation = "relu"))
        net.add(gluon.nn.Dense(16, activation = "relu"))
        net.add(gluon.nn.Dense(8, activation = 'relu'))
        net.add(gluon.nn.Dense(1))
        return net


def f3():

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.Conv2D(channels= 16, kernel_size=(3, 3), strides = (1, 1), activation=None))

        net.add(gluon.nn.Conv2D(channels= 32, kernel_size=(3, 3), strides = (1, 1), activation='relu'))

        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(3, 3), strides = (1, 1), activation='relu'))

        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(3, 3), strides = (1, 1), activation='relu'))

        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(2, 2), strides = (1, 1), activation='relu'))

        net.add(gluon.nn.Conv2D(channels= 256, kernel_size=(2, 2), strides = (1, 1), activation='relu'))

        net.add(gluon.nn.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(256, activation = None))
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.Dense(128, activation = 'relu'))
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.Dense(64, activation = "relu"))
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.Dense(32, activation = "relu"))
        net.add(gluon.nn.Dense(16, activation = 'relu'))
        net.add(gluon.nn.Dense(1))
        return net

def f4():

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)
        net.add(gluon.nn.Conv2D(channels= 8, kernel_size=(3, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 16, kernel_size=(3, 3), strides = (1, 1), activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Conv2D(channels= 32, kernel_size=(3, 3), strides = (1, 1), activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(64, activation = None))
        net.add(gluon.nn.Dense(32, activation = "relu"))
        net.add(gluon.nn.Dense(16, activation = "relu"))
        net.add(gluon.nn.Dense(8, activation = 'relu'))
        net.add(gluon.nn.Dense(1))
        return net


def f5():

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.Conv2D(channels= 8, kernel_size=(3, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding = 1))

        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(64, activation = None))
        net.add(gluon.nn.Dense(32, activation = "relu"))
        net.add(gluon.nn.Dense(16, activation = "relu"))
        net.add(gluon.nn.Dense(8, activation = 'relu'))
        net.add(gluon.nn.Dense(1))
        return net


def f6():

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.Conv2D(channels= 8, kernel_size=(3, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 16, kernel_size=(3, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 32, kernel_size=(5, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(2, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(5, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))


        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(300, activation = None))
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(200, activation = None))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(100, activation = None))
        net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(50, activation = None))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(20, activation = None))
        net.add(gluon.nn.Dense(1))
        return net


def f7(i, j):

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.Conv2D(channels= 16, kernel_size=(3, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 2))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 32, kernel_size=(3, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(5, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(2, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 256, kernel_size=(5, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))


        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(300, activation = None))
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(200, activation = None))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(100, activation = None))
        net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(50, activation = None))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(20, activation = None))
        net.add(gluon.nn.Dense(1))
        return net


def f8():

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.Conv2D(channels= 8, kernel_size=(3, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 16, kernel_size=(3, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 32, kernel_size=(5, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(2, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(5, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))


        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(400, activation = None))
        net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(200, activation = None))
        net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(100, activation = None))
        net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(50, activation = None))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(20, activation = None))
        net.add(gluon.nn.Dense(1))
        return net


def f9(i, j):

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.Conv2D(channels= 8, kernel_size=(2, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 2))

        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 16, kernel_size=(2, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 32, kernel_size=(2, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(2, 6), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(2, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(2, 6), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(2, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 256, kernel_size=(5, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 256, kernel_size=(2, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels= 256, kernel_size=(2, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.LeakyReLU(0.4))
        net.add(gluon.nn.Dense(300, activation = None))
        #net.add(gluon.nn.Dropout(0.3)

        net.add(gluon.nn.LeakyReLU(0.4))
        net.add(gluon.nn.Dense(200, activation = None))
        #net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.4))
        net.add(gluon.nn.Dense(100, activation = None))
        #net.add(gluon.nn.Dropout(0.5))

        net.add(gluon.nn.LeakyReLU(0.4))
        net.add(gluon.nn.Dense(50, activation = None))

        net.add(gluon.nn.LeakyReLU(0.4))
        net.add(gluon.nn.Dense(12, activation = None))

        net.add(gluon.nn.Dense(1))
        return net

def f10():

    net = gluon.nn.Sequential()

    with net.name_scope():
        net.cast(np.float32)

        net.add(gluon.nn.Conv2D(channels= 16, kernel_size=(3, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.ELU(0.3))
        net.add(gluon.nn.Conv2D(channels= 32, kernel_size=(3, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.ELU(0.3))
        net.add(gluon.nn.Conv2D(channels= 64, kernel_size=(5, 3), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.ELU(0.3))
        net.add(gluon.nn.Conv2D(channels= 128, kernel_size=(2, 5), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))

        net.add(gluon.nn.ELU(0.3))
        net.add(gluon.nn.Conv2D(channels= 256, kernel_size=(5, 2), strides = (1, 1), activation=None))
        net.add(gluon.nn.MaxPool2D(pool_size=(2, 2), strides=(1, 2), padding = 1))


        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(300, activation = None))
        net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(200, activation = None))
        net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(100, activation = None))
        net.add(gluon.nn.Dropout(0.3))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(50, activation = None))

        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Dense(20, activation = None))
        net.add(gluon.nn.Dense(1))
        return net


l = []
tr_err = []
te_err = []

def my_loss(y_pred, y_test):
    return (((y_pred - y_test)**2).sum() + (nd.abs(y_pred - y_test)).sum())/len(y_test)

for u in [0.01]:

    print(f"wd = {u}")
    net = f1()
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001, 'wd': u})
    net.collect_params()
    loss_fc = gluon.loss.L2Loss()
    epochs = 10
    smoothing_constant = .01


    for e in range(epochs):

        #start = time()

        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = loss_fc(output, label)
            #print(type(output[0][0].asnumpy()[0]))
            #z = data
            #print(loss)
            loss.backward()
            trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                        else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
            del data
            del label
            l.append(moving_loss)

        val_error = evaluate_error(test_data, net)
        train_error = evaluate_error(train_data, net)
        tr_err.append(train_error)
        te_err.append(val_error)

        #end = time()
        #print(f"Time epoch {e+1} : {(end - start)/60}")

        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e+1, moving_loss, train_error,
                                                             val_error))
        print("---------------------------------------------------------------------------------------------------------")

    #print(prediction(test_data))
    #plt.plot(range(2000), loss_list)
    #plt.show()
    print("*********************************************************************************")
    print("*********************************************************************************")
