class Network:
    """
    This class represents a Neural Network
    """
    def __init__(self):
        self.layers = []
        print("Initializing Network object.")

    def add_layer(self, layer):
        self.layers.append(layer)
        print("Adding a Layer to the Network.")

    def add_layers(self, layers):
        self.layers.layers(layers)
        print(f"Adding {len(layers)} Layers to the Network.")


