class Layer:
    """
    This class represents a Neural Network Layer
    """
    def __init__(self):
        self.nodes = []
        print("Initializing Layer object.")

    def add_node(self, node):
        self.nodes.append(node)
        print("Adding a Node to the Layer.")

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)
        print(f"Adding {len(nodes)} Nodes to the Layer.")


