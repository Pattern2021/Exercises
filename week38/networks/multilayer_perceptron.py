import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from neural_network import *

class Multilayer_perceptron(NeuralNetwork):
    """ 
    Multilayer network implementation of perceptron algorithm. Child which inherits from neural network base class.

    Parameters:
        Xtr: array_like
            Training data structured as an matrix with N-rows corresponing to datapoint and l-columns representing features.
        Ytr: array_like
            Training data labels. Structured as l-columns with 1 row. Values contained should have class belonging information.
        network_structure: array_like
            Array with information of the structure of the network. len(structure) represents the number of layers, whilst values represents neurons within.
    """
    def __init__(self, Xtr, Ytr, network_structure):
        super().__init__(Xtr, Ytr, network_structure)

    def train(self, mu=1):
        J = 0
        # For each datapoint
        for i, x in enumerate(self.Xtr[:1]):

            # for each layer
            y_prev = x.reshape(3, 1)
            
            y_prev_arr = []
            v_arr = []
            # loops through all layers which has a weight defined.
            for layer in self.network.layers[1:]:
                layer.calc_v(y_prev)

                # Temporary fix, but i think its not right. Inserting a noter row of only a one in v as we need this to get the correct shape.
                if layer.v.shape != x.reshape(3, 1).shape:
                    one = np.ones((1, 1))
                    layer.v = np.concatenate((layer.v, one))
                
                y_prev = self.sigmoid(layer.v)
                last_v = layer.v
                v_arr.append(layer.v)
                y_prev_arr.append(y_prev)

            y = y_prev
            # cost function
            error = np.sum((y - self.Ytr[i]))
            J += error
            
            # backpropagation
            e = np.sum(y - self.Ytr[i], axis=1)
            last_delta = e @ self.sigmoid_derivative(last_v)
            e_arr = [e]
            deltas = [last_delta]
            
            e_arr.insert(0, deltas[-1] @ self.w[-1])
            deltas.insert(0, e_arr[-2] @ self.sigmoid_derivative(v_arr[-2]))

            # print(y_prev_arr)
            # print(deltas)
            # exit()

            for i, w in enumerate(self.w):
                # print(y_prev_arr[i].T.shape)
                # print(np.atleast_2d(deltas[i]).shape)
                # print((np.atleast_2d(deltas[i]) @ y_prev_arr[i].T).shape)
                # # exit()
                # print(w.shape)
                delta_w = - mu * np.atleast_2d(deltas[i]) @ y_prev_arr[i].T
                self.network.layers[i + 1].update_weights(delta_w[::-1])