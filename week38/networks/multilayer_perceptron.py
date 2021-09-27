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

    def forward_propagate(self, Xtr, Ytr):
        J = 0
        it = 0
        v_mat_shape = (self.network.shape[:-1] + np.array([0, 1]))
        self.v_mat = np.zeros(v_mat_shape)

        self.y_arr = []
        y_predictions = np.zeros_like(Ytr)

        for i, x in enumerate(Xtr):
            y_prev = np.atleast_2d(x)

            # loops through all layers which has a weight defined.
            for r, layer in enumerate(self.network.layers[1:]):
                # v = y_prev @ layer.w_mat.T
                
                # # Now v has one less column than the input data, therefore concatenate bias.
                # bias = np.random.uniform(size=(1,1))
                # v = np.c_[v, bias]
                v = np.zeros_like(y_prev)

                for k, neuron in enumerate(layer.nodes):
                    v[:, k] = np.sum(neuron.w * y_prev[:, k])

                self.v_mat[r] = v
                
                y = self.sigmoid(v)
                y_prev = y

                self.y_arr.append(y)
            
            # actual prediction
            y_hat = np.where(np.round(y[0][0]) == np.round(y[0][1]), 1, 0)
            y_predictions[i] = y_hat

            MSE = np.sum((y_hat - Ytr[i]) ** 2)
            error = np.sum(y_hat - Ytr[i])  # np.sum(Ytr[i] - y[0])
            J += MSE
            it += 1

        # cast to numpy array, this gives wierd 3d array shape
        self.y_arr = np.array(self.y_arr)

        # reshaping to 2d array
        self.y_arr = self.y_arr.reshape((self.y_arr.shape[0], self.y_arr.shape[2]))
        
        # Pick out every odd item of list
        self.y_arr_prev_layer = self.y_arr[::2]


        return J / it, error, y_predictions

    def backward_propagate(self, error):
        last_delta = error * self.sigmoid_derivative(self.v_mat[-1])
        deltas = np.zeros_like(self.v_mat)
        deltas[-1] = last_delta

        # iterates backwards through layers but does not count over input layer
        for layer in self.network.layers[:0:-1]:
            r = layer.index

            e = np.dot(deltas[r - 1], layer.w_mat.T)
            if r >= 2:
                der = self.sigmoid_derivative(self.v_mat[r - 2])
                deltas[r - 2] = e * der

        return deltas
        
    def update_weights(self):

        # is the input from first node and input from second node to third node. has shape (2, 3, N)
        y = np.array([self.Xtr, self.y_arr_prev_layer])
        
        # loop through each layer except input layer.
        for r, layer in enumerate(self.network.layers[1:]):

            delta_w = - self.learning_rate * np.sum(self.deltas[r] @ y[r].T)

            for neuron in layer.nodes:
                neuron.change_class_weights(neuron.w + delta_w)


    def train(self, mu=1, epochs=1000):

        self.learning_rate = mu
        epochs = np.arange(epochs)
        cost_arr = []
        error_arr = []
        for epoch in epochs:
            cost, error, y_pred = self.forward_propagate(self.Xtr, self.Ytr)
            self.deltas = self.backward_propagate(error)
            self.update_weights()
            cost_arr.append(cost)
            # print(error)
            error_arr.append(error)
        # plt.plot(epochs, error_arr, label="error")
        # plt.plot(epochs, cost_arr, label="cost")
        # plt.legend()
        # plt.show()

    def test(self, Xte, Yte):
        cost, error, y_pred = self.forward_propagate(Xte, Yte)
        return cost, error, y_pred

    def plot_training(self):

        x1_range = np.arange(np.min(self.Xtr[:, 0]), np.max(self.Xtr[:, 0]), 0.1)
        x2_range = np.arange(np.min(self.Xtr[:, 1]), np.max(self.Xtr[:, 1]), 0.1)

        X, Y = np.meshgrid(x1_range, x2_range)
        
        y_fakearr = np.zeros_like(x1_range)

        for i in range(len(X)):
            vec = np.c_[X[i], Y[i]]
            vec = self.onecolumn(vec)

            cost, error, y_pred = self.test(vec, y_fakearr)

