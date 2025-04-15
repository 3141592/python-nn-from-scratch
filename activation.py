from math import exp

class Activation:
    """
    This class represents a Neural Network Node
    """
    @staticmethod 
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod 
    def relu(x):
        return max(0, x)

    @staticmethod 
    def tanh(x):
        return 2 / (1 + exp(-2*x)) - 1


