import numpy as np

def tanh(x):
    return np.tanh(x)

def prime_tanh(x):
    return 1-np.tanh(x)**2

def mse(y_true, y_predicted):
    return np.mean(np.power(y_true-y_predicted,2))

def prime_mse(y_true, y_predicted):
    return 2*(y_true-y_predicted)/y_true.input_size