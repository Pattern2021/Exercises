import numpy as np

class Network:
    def __init__(self, shape=np.array([2, 2, 1])):
        self.shape = shape
        self.layers = [Layer(self, nodes, layer_index) for layer_index, nodes in enumerate(self.shape)]
        self.w = []
        
        # looping through layer except the first one.
        for layer in self.layers[1:]:
            self.w.append(layer.w_mat)

class Layer(Network):
    def __init__(self, network, nodes, index):
        self.network = network
        self.index = index

        self.is_input = False
        self.is_output = False
        self.is_hidden = False

        if self.index == 0:
            self.is_input = True
        elif len(self.network.shape) == index + 1:
            self.is_output = True
        else:
            self.is_hidden = True

        self.nodes = [Node(self, index) for index in range(nodes)]

        # Fixes correct size of weight matrix of layer
        if self.index > 0:
            self.w_mat = np.array([node.w for node in self.nodes])
            self.w_mat = self.w_mat.reshape(self.w_mat.shape[0], self.w_mat.shape[2])
        
    def calc_v(self, y_prev):
        self.v = self.w_mat @ y_prev

    def update_weights(self, delta_w):
        self.w_mat = self.w_mat + delta_w

class Node(Network):
    def __init__(self, layer, index):
        self.layer = layer
        self.index = index

        prev_layer_nodes = self.layer.network.shape[self.layer.index - 1]

        if self.layer.is_input:
            self.w = None
        else:
            # weights at this node including bias. i.e. number of nodes in previous layer plus bias
            self.w = np.random.uniform(size=(1, prev_layer_nodes + 1))

def main():
    n1 = Network()
    # n1.layers[1].calc_v(np.array([1, 0.66, 0.5]).reshape(3,1))

if __name__ == "__main__":
    main()