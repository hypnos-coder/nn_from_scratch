import numpy as np

# Define the activation functions
def tanh(x):
    return np.tanh(x)

def prime_tanh(x):
    return 1 - np.tanh(x) ** 2

def mse(y_true, y_predicted):
    return np.mean(np.power(y_true - y_predicted, 2))

def prime_mse(y_true, y_predicted):
    return 2 * (y_predicted - y_true) / y_true.shape[0]

# Define the Layer class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

# Define the ActivationLayer class
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

# Define the FCLayer class
class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))
        self.momentum_weight = np.zeros_like(self.weight)
        self.momentum_bias = np.zeros_like(self.bias)
        self.velocity_weight = np.zeros_like(self.weight)
        self.velocity_bias = np.zeros_like(self.bias)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weight.T)
        weight_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.momentum_weight = self.beta1 * self.momentum_weight + (1 - self.beta1) * weight_error
        self.momentum_bias = self.beta1 * self.momentum_bias + (1 - self.beta1) * bias_error
        self.velocity_weight = self.beta2 * self.velocity_weight + (1 - self.beta2) * (weight_error ** 2)
        self.velocity_bias = self.beta2 * self.velocity_bias + (1 - self.beta2) * (bias_error ** 2)

        m_weight_hat = self.momentum_weight / (1 - self.beta1)
        v_weight_hat = self.velocity_weight / (1 - self.beta2)
        m_bias_hat = self.momentum_bias / (1 - self.beta1)
        v_bias_hat = self.velocity_bias / (1 - self.beta2)

        self.weight -= (learning_rate * m_weight_hat) / (np.sqrt(v_weight_hat) + self.epsilon)
        self.bias -= (learning_rate * m_bias_hat) / (np.sqrt(v_bias_hat) + self.epsilon)

        return input_error

# Define the Network class
class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def train(self, x_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                output = x
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                error = self.loss_prime(y, output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            if (epoch + 1) % 100 == 0:
                loss = self.calculate_loss(x_train, y_train)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def calculate_loss(self, x, y):
        total_loss = 0
        for i in range(len(x)):
            output = self.predict(x[i])
            loss = self.loss(y[i], output)
            total_loss += loss
        return total_loss / len(x)

# Create the XOR training dataset
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Create the network
network = Network()

# Add layers to the network
network.add(FCLayer(2, 3))
network.add(ActivationLayer(tanh, prime_tanh))
network.add(FCLayer(3, 1))
network.add(ActivationLayer(tanh, prime_tanh))

# Specify the loss function and its derivative
network.use(mse, prime_mse)

# Train the network
network.train(x_train, y_train, learning_rate=0.1, epochs=1000)

# Test the network
for x, y in zip(x_train, y_train):
    output = network.predict(x)
    print(f"Input: {x}, Predicted: {output}")
