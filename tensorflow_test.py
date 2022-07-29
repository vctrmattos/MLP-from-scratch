from tabnanny import verbose
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np

t0 = time.time()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# n = 4000
# x_train, y_train, x_test, y_test = x_train[:n], y_train[:n], x_test[:n], y_test[:n]
x_train = x_train / 255
x_test = x_test / 255

x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)

model = keras.Sequential([
    keras.layers.Dense(32, input_shape=(784,), activation='relu'),
    keras.layers.Dense(16, input_shape=(16,), activation='relu'),
    keras.layers.Dense(10, input_shape=(16,), activation='sigmoid')
])

# Optimizer will help in backproagation to reach better global optima
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Does the training
model.fit(x_train_flattened, y_train, epochs=20, validation_data=(x_test_flattened, y_test))
print("Tempo: ", t0 - time.time())  