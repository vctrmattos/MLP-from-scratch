from dense_layer import DenseLayer
from linearAlgebra import Matrix
from network import Network
from dataset import *
from nasc import *
import pickle
import time

epochs  = 30
learning_rate = 0.007
#Main file for model training on MNIST dataset

t2 = time.time()


#Download the MNIST dataset
# get_mnist()

# Read 
xy_test = read_dataset("testing",   1)
xy_train = read_dataset("training", 1)

#Shuffle
x_test, y_test = shuffle_dataset(xy_test)
x_train, y_train = shuffle_dataset(xy_train)

x_test  = [i/255 for i in x_test]
x_train = [i/255 for i in x_train]

#to categorical e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

net = Network()
net.add_layer(DenseLayer(28*28, 16, relu, "random", "random"))
net.add_layer(DenseLayer(16, 16, relu, "random", "random"))
net.add_layer(DenseLayer(16, 10, sigmoid, "random", "random"))
net.set_loss(mse)

t0 = time.time()

params = net.fit_and_test(x_train, y_train, epochs=epochs, learning_rate=learning_rate, iter_step=1) 

t1 = time.time()

pred_test = net.predict(x_test)
print("test: ", mnist_acc(y_test, pred_test))
pred_train = net.predict(x_train)
print("train: ", mnist_acc(y_train, pred_train))

file = open('mnist_100p_1616_007_30.pkl', 'wb')
pickle.dump(net, file)

t3 = time.time()

print("Training duration:", t1 - t0,"\nTotal duration:", t3 - t2)
print("Data length:", len(x_train), "Learning rate: ", learning_rate, ", Epochs: ", epochs )

#uncomment the line below if you want to plot the training params
# plot_model(params)