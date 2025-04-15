import random
import numpy as np
from dataclasses import dataclass, field
from activation import Activation

@dataclass
class Node:
    """
    This class represents a Neural Network Node
    """
    value: float = 0.0
    weights: list = field(default_factory=list)
    bias: float = 0.0
    debug: bool = False

    def __post_init__(self):
        if self.debug: print("Initialized Node")

    def __str__(self):
        return f"Node(value={self.value:.4f}, bias={self.bias:.4f}, weights={len(self.weights) if self.weights else 0})"

    def initialize_weights(self, x):
        """
        Argument: x is an example list
        """
        if len(self.weights) == 0:
            self.weights = [random.uniform(0, 1) for _ in range(len(x))]

    def initialize_bias(self):
        """
        Argument: x is an example list
        """
        if len(self.weights) == 0:
            self.bias = random.uniform(0, 1)

    def forward_prop(self, x):
        """
        One step of forward propagation.
        """
        if len(self.weights) == 0:
            self.initialize_weights(x)
            self.initialize_bias()

        z = np.dot(self.weights, x) + self.bias
        self.value = Activation.sigmoid(z)
        print(self.weights)
        return self.value

    def print_weights(self):
        print(self.weights)


