import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2


def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    return np.where(x>0,1,0)

def mse(y_true, y_predicted):
    return np.mean(np.power(y_true-y_predicted,2))

def mse_prime(y_true, y_predicted):
    return 2*(y_true-y_predicted)/y_true.size

def linear(x):
    return x

def prime_linear(x):
    return np.ones_like(x)

def mae(y_true, y_predicted):
    return np.mean(np.abs(y_true - y_predicted))

def prime_mae(y_true, y_predicted):
    return np.sign(y_predicted - y_true) / y_true.shape[0]

