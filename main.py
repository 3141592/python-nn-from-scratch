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
network.learning_rate = 0.1

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

# Set epochs
epochs = 3

for epoch in range(1, epochs + 1):
    # Forward propagation
    output = network.forward_prop(x)

    # Compute and print the loss
    loss = mean_squared_error(output, y)
    network.backward_prop(x, y)

    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch}: loss = {loss:.4f}")


print(f"x: {x}")
print(f"y: {y}")
print(f"Prediction: {output}")
#print(f"Error: {errors}")
print(f"Loss: {loss:.4f}")

# Print network
#print("="*18)
#print(network)

#print("="*18)
#network.print_weights()


