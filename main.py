from network import Network
from layer import Layer
from node import Node

def mean_squared_error(y_pred, y_true):
    return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / len(y_true)

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
output = network.forward_prop(x)

# Compute and print the loss
loss = mean_squared_error(output, y)
errors = [i - j for i, j in zip(output, y)]

print(f"x: {x}")
print(f"y: {y}")
print(f"Prediction: {output}")
print(f"Error: {errors}")
print(f"Loss: {loss:.4f}")

# Print network
#print("="*18)
#print(network)

#print("="*18)
#network.print_weights()


