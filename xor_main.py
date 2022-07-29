from dense_layer import DenseLayer
from linearAlgebra import Matrix
from network import Network
from nasc import *

import time

def xor_acc(y_train, predictions):
    acc = 0
    preds_len = len(predictions)
    for i in range(preds_len):
        if predictions[i][0] >= 0.5:
            predictions[i][0] = 1
        else:
            predictions[i][0] = 0
        acc += predictions[i][0] == y_train[i,0]
    return acc/preds_len

t0 = time.time()
#Erros constantes após a 1º epoch
x_train = Matrix([[0,0], [0,1], [1,0], [1,1]])
y_train = Matrix([[0], [1], [1], [0]])
y_test = y_train

net = Network()

net.add_layer(DenseLayer(2, 50, relu, "random", "random"))
net.add_layer(DenseLayer(50, 1, relu, "random", "random"))

net.set_loss(mse)
params = net.fit_and_test(x_train, y_train, x_val=x_train, y_val=y_train, epochs=2000, progression=True, learning_rate=0.01, acc_fun=xor_acc, iter_step=100) 

out = net.predict(x_train)
print(out)

print(time.time() - t0)