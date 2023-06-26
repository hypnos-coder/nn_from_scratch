class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self,input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):#compute dE/dx for a given dE/dY and update parameters if not last layer
        raise NotImplementedError



    

    