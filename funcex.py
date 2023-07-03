import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
import activations
from losses import mse, mse_prime
import matplotlib.pyplot as plt
from data_generator import data_generator

# training data
x_train,y_train =data_generator(500)
# print(x_train)

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(activations.linear, activations.prime_linear))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(activations.linear, activations.prime_linear))

# train
net.use(activations.mae, activations.prime_mae)

errors = net.fit(x_train, y_train, epochs=100, learning_rate=0.1)

# test
out = net.predict(np.array([[2,-1]]))
print(out)
