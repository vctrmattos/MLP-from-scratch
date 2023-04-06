from nasc import *
from layer import Layer 


class Activation(Layer):
    '''
    Activation function class.
    '''
    def __init__(self, activation):
        #Creates a correspondence between a activation function and its derivative
        self.activation_dict = {
            relu:    step,
            sigmoid: sigmoid_d,
            tanh:    tanh_d 
        }

        self.activation = activation
        self.activation_d = self.activation_dict[self.activation]
    
    def forward_propagation(self, input):
        self.input = input
        self.output = self.input.apply(self.activation)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.input.apply(self.activation_d)*output_error