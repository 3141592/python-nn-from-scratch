from math import exp

class Activation:
    """
    This class represents a Neural Network Node
    """
    @staticmethod 
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def sigmoid_derivative(z):
        s = Activation.sigmoid(z)
        return s * (1 - s)

    @staticmethod 
    def relu(x):
        return max(0, x)

    @staticmethod 
    def tanh(x):
        return 2 / (1 + exp(-2*x)) - 1


