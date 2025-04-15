from network import Network
from layer import Layer
from node import Node

# Example data
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]

# Define network
network = Network()

#
# Define layer 1
layer = Layer()

for node_index in range(3):
    node = Node()
    layer.add_node(node)

# Build network
network.add_layer(layer)

#
# Define layer 2
layer = Layer()

for node_index in range(5):
    node = Node()
    layer.add_node(node)

# Build network
network.add_layer(layer)

#
# Define layer 3
layer = Layer()

for node_index in range(1):
    node = Node()
    layer.add_node(node)

# Build network
network.add_layer(layer)

# Forward propagation
network.forward_prop(x)

# Print network
print("="*18)
print(network)

print("="*18)
network.print_weights()


