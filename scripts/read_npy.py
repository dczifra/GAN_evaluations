import numpy as np
import os

def valami():
    file="/home/doma/model_celeba10000.npy"
    data=np.load(file)
    print(np.shape(data))

def read_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.resize(x_train, (-1,28, 28, 1))
    np.save("models/mnist/test.npy", x_train[5000:7100])

read_mnist()
