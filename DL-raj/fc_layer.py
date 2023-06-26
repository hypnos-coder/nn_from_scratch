from layer import Layer
import numpy as np


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weight = np.random.rand(input_size,output_size)-0.5
        self.bias = np.random.rand(1,output_size)-0.5

    def forward_propagation(self,input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weight)+self.bias

        return self.output

    def backward_propagation(self, output_error, learning_rate):#compute dE/dW, dE/dB given dE/dY and eta return dE/dX wich will serve as the input for the next one
        input_error = np.dot(output_error,self.weight.T)
        weight_error = np.dot(self.input.T, output_error)
        bias_error = output_error

        #update parameters

        self.weight-= learning_rate*weight_error
        self.bias-= learning_rate*bias_error

        return input_error

    

