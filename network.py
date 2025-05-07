from dataclasses import dataclass, field

@dataclass
class Network:
    """
    This class represents a Neural Network
    """
    layers: list = field(default_factory=list)
    debug: bool = False
    learning_rate: float = 0.1

    def __post_init__(self):
        self.layeArs = []
        if self.debug: print("Initializing Network object.")

    def __str__(self):
        net_str = "Network:\n"
        for i, layer in enumerate(self.layers):
            net_str += f" Layer {i + 1}:\n{layer}"
        return net_str

    def add_layer(self, layer):
        self.layers.append(layer)
        if self.debug: print("Adding a Layer to the Network.")

    def add_layers(self, layers):
        self.layers.extend(layer)
        if self.debug: print(f"Adding {len(layers)} Layers to the Network.")

    def forward_prop(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward_prop(self, x, y):
        # Forward pass values must be saved in the layers (activations, z)
        reversed_layers = list(reversed(self.layers))
        next_grad = None  # Start with None

        for i, layer in enumerate(reversed_layers):
            if i == 0:
                # Output layer: needs true y
                next_grad = layer.backward(self.learning_rate, y_true=y)
            else:
                # Hidden layer: use gradient passed from layer ahead
                next_grad = layer.backward(self.learning_rate, grad_from_next=next_grad)

    def print_weights(self):
        for layer in self.layers:
            x = layer.print_weights()



