from math import exp

class Activation:
    """
    This class represents a Neural Network Node
    """
    @staticmethod 
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    @staticmethod 
    def relu(self, x):
        return max(0, x)

    @staticmethod 
    def tanh(self, x):
        return 2 / (1 + exp(-2*x)) - 1


