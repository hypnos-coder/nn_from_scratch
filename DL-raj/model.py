import numpy as np

class Network():
    def __init__(self, sizes):
        self.nLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weight = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        for w,b in zip(self.weight,self.biases):
            a = sigmoid(np.dot(w,a)+b)
        return a


    def gradient_descent(self,learning_rate, training_data, test_data=None, ):
        if test_data:
            n=len(test_data)
        n=training_data
        x=training_data
        self.update_hyper_parameters(learning_rate,nn)

    def update_hyper_parameters(self,learning_rate,x,y):
        #x-input, y-output
        gradient_b = [np.zeros(b.shapes) for b in self.biases]
        gradient_w = [np.zeros(w.shapes) for w in self.weight]

        delta_gradient_b, delta_gradient_w = self.backprop(x,y)

        gradient_b = [nb+dgb for nb, dgb in zip(gradient_b,delta_gradient_b)]
        gradient_w = [nw+dgw for nw, dgw in zip(gradient_w,delta_gradient_w)]

        self.weight = [w - (learning_rate*nw) for w,nw in zip(self.weight,gradient_w)]
        self.biases = [b - (learning_rate*nb) for b,nb in zip(self.biases,gradient_b)]

    def backprop(self,x,y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weight]
        activation = x #initialize the first activation as x
        activations = [x]# will keep track of the a's

        zs = []#will keep track of the z's 

        #feedforward
        for w,b in zip(self.weight,self.biases):
            z = sigmoid(np.dot(w,activation)+b)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        error_delta = self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
        gradient_b[-1] = error_delta
        gradient_w[-1] = np.dot(error_delta,activations[-2].transpose())

        for layer in range(self.nLayers-2):
            current_z = zs[layer]
            sp = sigmoid_prime(current_z)
            error_delta = np.dot(self.weight[layer-1].transpose(),error_delta)*sp
            gradient_b[layer] = error_delta
            gradient_w[layer] = np.dot(error_delta,activations[-2].transpose())
        
        return (gradient_b,gradient_w)

    def cost_derivative(self,output_activation, y):#cross entropy cost function
        return( output_activation-y)



def sigmoid(z):
        return 1/(1+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

net = Network([2,3,1])
data = np.array([8,5])
y = np.array([2])
print(net.weight)
# print(net.backprop(np.reshape(data,(2,1)),y)[1])