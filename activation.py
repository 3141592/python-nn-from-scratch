from dataclasses import dataclass
from typing import Callable
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
    def relu_derivative(z):
        return 1 if z > 0 else 0

    @staticmethod 
    def tanh(x):
        return 2 / (1 + exp(-2*x)) - 1

    @staticmethod
    def tanh_derivative(z):
        t = math.tanh(z)
        return 1 - t * t

@dataclass
class ActivationFunction:
    func: Callable[[float], float]
    derivative: Callable[[float], float]
