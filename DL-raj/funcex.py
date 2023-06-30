import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime, relu, relu_prime
from losses import mse, mse_prime
import matplotlib.pyplot as plt
from data_generator import data_generator

# training data
x_train,y_train =data_generator(500)
# print(x_train)

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(relu, relu_prime))

# train
net.use(mse, mse_prime)

errors = net.fit(x_train, y_train, epochs=1000, learning_rate=0.3)

# test
out = net.predict(np.array([[2,-1]]))
print(out)
