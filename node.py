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

