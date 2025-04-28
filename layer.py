from dataclasses import dataclass, field

@dataclass
class Layer:
    """
    This class represents a Neural Network Layer
    """
    nodes: list = field(default_factory=list)
    values: list = field(default_factory=list)
    debug: bool = False

    def __post_init__(self):
        if self.debug: print("Initialized Layer.")

    def __str__(self):
        layer_str = "  Layer:\n"
        for node in self.nodes:
            layer_str += f"    {node}\n"
        return layer_str

    def add_node(self, node):
        self.nodes.append(node)
        if self.debug: print("Adding a Node to the Layer.")

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)
        if self.debug: print(f"Adding {len(nodes)} Nodes to the Layer.")

    def forward(self, x):
        """
        One step of forward propagation.
        """
        self.values = []
        for node in self.nodes:
            value = node.forward_prop(x)
            self.values.append(value)
        return self.values

    def backward(self, x):
        """
        One step of backward propagation.
        """
        self.values = []
        for node in self.nodes:
            value = node.backward_prop(x)
            self.values.append(value)
        return self.values

    def print_weights(self):
        """
        Print weights.
        """
        for node in self.nodes:
            node.print_weights()
        



