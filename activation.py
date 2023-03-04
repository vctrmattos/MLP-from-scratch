from nasc import *


class Activation:
    '''
    Activation function class.
    '''
    def __init__(self, activation):
        self.activation = None
        self.activation_d = None
        
        self.activation_dict = {
            relu:    step,
            sigmoid: sigmoid_d,
            tanh:    tanh_d
        }