import numpy as np

from Network import Network
from fc_layer import FCLayer
from ActivationLayer import ActivationLayer
from activations import tanh, prime_tanh, prime_mse, mse

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, prime_tanh))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, prime_tanh))

# print(type(net.layers[0]))

# train
net.use(mse, prime_mse)
net.train(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)