from layer import Layer

class ActivationLayer(Layer):
    def __init__(self,activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output# return the activation of the previously calculated forwrd

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input)*output_error

        