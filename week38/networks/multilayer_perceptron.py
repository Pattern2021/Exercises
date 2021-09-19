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
        # For each datapoint
        for i, x in enumerate(self.Xtr[:3]):

            # for each layer
            a = x.reshape(3, 1)
            z_arr = []
            activations = []

            for i, w_in_layer in enumerate(self.w):
                z = w_in_layer @ a

                if z.shape != x.reshape(3,1).shape:
                    one = np.ones((1,1))
                    z = np.concatenate((z, one))

                a = self.sigmoid(z)

                # store theese for backpropagation
                if len(a) > 0:
                    activations.append(a)
                    z_arr.append(z)
            
            # Cost function
            error = np.sum((a - self.y) ** 2, axis=0)


            self.w = self.backprop(self.w, activations, z, error)