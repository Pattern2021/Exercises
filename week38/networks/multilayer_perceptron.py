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
        self.prev_delta_w = []

    def forward_propagate(self, Xtr, Ytr):
        J = 0

        missclassifications = 0

        y_prev = np.atleast_2d(Xtr)
        self.v_arr = []
        self.y_arr = [Xtr]
        
        # loops through all layers which has a weight defined.
        for r, layer in enumerate(self.network.layers[1:]):
            v = y_prev @ layer.w_mat.T

            # add column of ones to v-matrix
            if layer.index < len(layer.network.shape) - 1:
                v = np.append(v, np.ones((len(v), 1)), axis=1)

            self.v_arr.append(v)
            self.y = self.sigmoid(v)
            self.y_arr.append(self.y)

            y_prev = self.y

        y_hat = self.y

        self.error = (self.y.squeeze() - Ytr).reshape(len(self.y), 1)

        return np.round(y_hat)


    def backward_propagate(self, error):
        last_delta = error * self.sigmoid_derivative(self.v_arr[-1]) #[:, None]

        deltas = [last_delta]

        # iterates backwards through layers but does not count over input layer
        for layer in self.network.layers[:0:-1]:
            r = layer.index
            if r == 1:
                break

            e = deltas[-1] @ layer.w_mat[:, :-1]

            deltas_in_layer = []

            if r >= 2:
                der = self.sigmoid_derivative(self.v_arr[r - 2][:, :-1])

                # Calculate delta from elementwise multiplication with error with sigmoid derivative
                for error_node, der_node in zip(e.T, der.T):
                    
                    # For each node calculate deltas
                    delta_node = error_node * der_node
                    deltas_in_layer.append(delta_node)

                deltas.append(deltas_in_layer)

        self.update_weights(deltas)
        
    def update_weights(self, deltas):

        # loop through each layer except input layer.
        for r, layer in enumerate(self.network.layers[:0:-1]):
            opposite_index = len(self.network.layers[:0:-1]) - r
            print(r, opposite_index - 1)
            if layer.is_output:
                delta_w = - self.learning_rate * deltas[r].T @ self.y_arr[opposite_index - 1]
                print(delta_w.shape)
            else:
                delta_w = np.zeros((len(deltas[r]), 1))
                for i, delta in enumerate(deltas[r]):
                    # se over shapen til y_arr at denne lager riktig shape på delta_w_node
                    print("delta: ", delta.shape, self.y_arr[opposite_index - 1][:, i].shape)
                    delta_w_node = - self.learning_rate * delta @ self.y_arr[opposite_index - 1][:, i]
                    print(delta_w_node)
                    delta_w[i] = delta_w_node.reshape((1,1))

            for i, neuron in enumerate(layer.nodes):
                if layer.is_output:
                    neuron.change_class_weights(neuron.w + delta_w)
                else:
                    neuron.change_class_weights(neuron.w + delta_w[i])
        exit()

    def train(self, mu=1, epochs=1000, alpha=0, ax=None):
        self.alpha = alpha

        self.learning_rate = mu
        epochs = np.arange(epochs)

        errors_arr = []
        for epoch in epochs:
            y_hat = self.forward_propagate(self.Xtr, self.Ytr)
            
            self.backward_propagate(self.error)
            
            self.Ytr = np.atleast_2d(self.Ytr)

            # print(y_hat.shape, self.Ytr.shape)

            # predictions = (np.round(y_hat.T - self.Ytr) == 1)
            predictions = y_hat.T != self.Ytr
            errors = len(predictions[predictions])
            if errors == 0:
                break
            errors_arr.append(errors)
        print(errors, self.network.w)

        ax.plot(epochs[0:len(errors_arr)], errors_arr)
        ax.set_title("Errors")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Errors")


    def test(self, Xte, Yte):
        y_hat = self.forward_propagate(Xte, Yte)
        return y_hat

    def plot_training(self, ax, lr):

        x1_range = np.linspace(np.min(self.Xtr[:, 0]), np.max(self.Xtr[:, 0]), 50)
        x2_range = np.linspace(np.min(self.Xtr[:, 1]), np.max(self.Xtr[:, 1]), 50)

        xx, yy = np.meshgrid(x1_range, x2_range)
        
        xx, yy = xx.reshape(len(x1_range)**2), yy.reshape(len(x2_range)**2)
        inp = np.transpose(np.vstack((xx, yy)))
        inp = np.c_[inp, np.ones(inp.shape[0])]
        y = np.atleast_2d(np.zeros(len(inp)))

        y_hat = self.test(inp, y)
        y_hat = y_hat.reshape(len(x1_range), len(x1_range))

        # print(np.round(y_hat)[0:10, 0:10])

        ax.contourf(x1_range,x2_range, y_hat, cmap="cool")
        ax.scatter(self.Xtr[:, 0], self.Xtr[:, 1], c=list(self.Ytr))
        ax.set_title("lr = {:.7f}".format(lr))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

