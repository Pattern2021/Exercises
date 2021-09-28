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
        it = 0

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


        MSE = np.sum((self.y - Ytr) ** 2, axis=0) / len(Xtr)

        error = np.sum((self.y - Ytr), axis=0) / len(Xtr)
        # J += np.sum(MSE) / len(MSE)
        # missclassifications += np.sum(np.abs(np.round(y) - Ytr), axis=0)
        # print(J)
        self.backward_propagate(error)

    def backward_propagate(self, error):
        last_delta = error[:, None] * self.sigmoid_derivative(self.v_arr[-1])
        
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
        for r, layer in enumerate(self.network.layers[1:]):
            
            if layer.is_output:
                delta_w = -self.learning_rate * np.sum(deltas[r].T @ self.y_arr[r])
            else:
                delta_w = np.zeros((len(deltas[r]), 1))
                for i, delta in enumerate(deltas[r]):
                    # print(len(delta), self.y_arr[r][:, i].T.shape, "delta", r)

                    delta_w_node = -self.learning_rate * np.sum(delta @ self.y_arr[r][:, i])
                    delta_w[i] = delta_w_node.reshape((1,1))

            # delta_w = self.alpha * self.prev_delta_w[r] - self.learning_rate * np.sum(deltas[r] @ y[r].T)

            # store this weight
            print(delta_w)
            # self.prev_delta_w[]
            # self.prev_delta_w[r] = delta_w

            for i, neuron in enumerate(layer.nodes):
                if layer.is_output:
                    neuron.change_class_weights(neuron.w + delta_w)
                else:
                    neuron.change_class_weights(neuron.w + delta_w[i])



    def train(self, mu=1, epochs=1000, alpha=0):
        self.alpha = alpha

        # fig, axs = plt.subplots(2, 1, figsize=(8,8))

        self.learning_rate = mu
        epochs = np.arange(epochs)
        cost_arr = []
        error_arr = []
        for epoch in epochs:
            print("epoch: ", epoch)
            self.forward_propagate(self.Xtr, self.Ytr)



            # cost_arr.append(cost)
            # print(error)
            # error_arr.append(error)
        # axs[0].plot(epochs, error_arr, label="missclassifications")
        # axs[1].plot(epochs, cost_arr, label="cost function (MSE)")
        # axs[0].legend()
        # axs[1].legend()
        # plt.show()

    def test(self, Xte, Yte):
        cost, error, y_pred = self.forward_propagate(Xte, Yte)
        return cost, error, y_pred

    def plot_training(self):

        x1_range = np.linspace(np.min(self.Xtr[:, 0]), np.max(self.Xtr[:, 0]), 50)
        x2_range = np.linspace(np.min(self.Xtr[:, 1]), np.max(self.Xtr[:, 1]), 50)

        X, Y = np.meshgrid(x1_range, x2_range)
        
        y_fakearr = np.zeros_like(x1_range)


        for i in range(len(X)):
            vec = np.c_[X[i], Y[i]]
            vec = self.onecolumn(vec)

            cost, error, y_pred = self.test(vec, y_fakearr)
            class_ind = np.where(y_pred == y_fakearr[i], True, False)
            c1 = vec[class_ind == True]
            c2 = vec[class_ind == False]
            plt.scatter(c1[:, 0], c1[:, 1], c="blue", s=1)
            plt.scatter(c2[:, 0], c2[:, 1], c="red", s=1)
        plt.show()

            
        


