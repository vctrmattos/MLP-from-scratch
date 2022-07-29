from dense_layer import DenseLayer
from linearAlgebra import Matrix
from network import Network
from nasc import *
import pickle
from keras.datasets import mnist
from keras.utils import np_utils

#It loads a model using pickle

def mnist_acc(y_test:Matrix, predictions:list): 
    cont = 0
    for i in range(len(predictions)):
        if predictions[i].index(max(predictions[i])) == y_test[i].array[0].index(max(y_test[i].array[0])):
            cont += 1
    return cont/len(predictions)

(x_train, y_train), (x_val, y_val) = mnist.load_data()

n_train = 5000
n_val = 1000

x_train, y_train, x_val, y_val = x_train[:n_train], y_train[:n_train], x_val[:n_val], y_val[:n_val]

# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_train = x_train.astype('float32')
x_train /= 255

# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : n_val samples
x_val = x_val.reshape(x_val.shape[0], 28*28)
x_val = x_val.astype('float32')
x_val /= 255
y_val = np_utils.to_categorical(y_val)

x_train, y_train, x_val, y_val = Matrix(x_train.tolist()), Matrix(y_train.tolist()), Matrix(x_val.tolist()), Matrix(y_val.tolist())

model = "models/mnist_all_1616_01.pkl"
file = open(model, "rb")
net = pickle.load(file)

pred_val = net.predict(x_val)
print("validation: ", mnist_acc(y_val, pred_val))

pred_train = net.predict(x_train)
print("train: ", mnist_acc(y_train, pred_train))
