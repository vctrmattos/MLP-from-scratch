from layer import Layer
from linearAlgebra import Matrix
from activation import Activation

# inherit from base class Layer
class DenseLayer(Layer, Activation):
    '''
    Fully connected layer class.
    '''
    def __init__(self, input_size, output_size, activation, weights_initializer, bias_initializer):

        Activation.__init__(self, activation)
        if weights_initializer == "random":
            self.weights = Matrix.random((input_size, output_size)) - 0.5 
        elif isinstance(weights_initializer, (int, float)):
            self.weights = Matrix.fill(weights_initializer, (input_size, output_size)) - 0.5
        else:
            raise(TypeError("weights_initializer must be 'random', int or float."))
        
        if bias_initializer == "random":
            self.bias = Matrix.random((1, output_size)) - 0.5
            # self.bias = Matrix.fill(bias_initializer, (1, output_size)) - 0.5 
        else:
            raise(TypeError("bias_initializer must be 'random', int or float."))
        
        self.activation = activation
        self.activation_d = self.activation_dict[self.activation]

    # returns output for a given input
    def forward_propagation(self, input_data):

        self.input = input_data
        self.output = (self.input @ self.weights + self.bias).apply(self.activation)
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        output_error = self.output.apply(self.activation_d) * output_error
        input_error = output_error @ self.weights.T()

        weights_error = self.input.T() @ output_error
        
        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error