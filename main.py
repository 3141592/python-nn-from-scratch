from network import Network
from layer import Layer
from node import Node

# Example data
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]

# Define network
network = Network()

# Define nodes
for layer_index in range(4):
    # Define layer
    layer = Layer()

    for node_index in range(3):
        node = Node()
        #print("="*18)
        #print(node.weights)
        #print(node.bias)
        layer.add_node(node)

    # Build network
    network.add_layer(layer)

# Print network
#print("="*18)
#print(network)

# Forward propagation
network.forward_prop(x)
network.print_weights()


