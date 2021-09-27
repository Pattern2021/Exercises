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
        self.prev_delta_w = np.zeros((2, 1))

    def forward_propagate(self, Xtr, Ytr):
        J = 0
        it = 0
        v_mat_shape = (self.network.shape[:-1] + np.array([0, 1]))
        self.v_mat = np.zeros(v_mat_shape)

        y_predictions = np.zeros_like(Ytr)
        missclassifications = 0

        for i, x in enumerate(Xtr):
            y_prev = np.atleast_2d(x)
            
            self.y_arr = []

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

            MSE = np.sum((y - Ytr[i]) ** 2)
            error = (y - Ytr[i])  # np.sum(Ytr[i] - y[0])
            # print(Ytr[i], y_hat)
            J += MSE
            it += 1

            missclassifications += np.abs(np.sum(y_hat - Ytr[i]))

            # ===== backpropogation and updating weights =====
            deltas = self.backward_propagate(error)
            self.update_weights(deltas, x)

        return J / it, missclassifications, y_predictions

    def backward_propagate(self, error):
        last_delta = error @ self.sigmoid_derivative(self.v_mat[-1])
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
        
    def update_weights(self, deltas, x):

        # is the input from first layer and input from second layer to third layer.
        y = np.insert(self.y_arr[:-1], 0, x, axis=1)
        y = y.reshape(y.shape[1], y.shape[2])
        
        # loop through each layer except input layer.
        for r, layer in enumerate(self.network.layers[1:]):

            delta_w = self.alpha * self.prev_delta_w[r] - self.learning_rate * np.sum(deltas[r] @ y[r].T)

            # store this weight
            self.prev_delta_w[r] = delta_w

            for neuron in layer.nodes:
                neuron.change_class_weights(neuron.w + delta_w)


    def train(self, mu=1, epochs=1000, alpha=0):
        self.alpha = alpha

        fig, axs = plt.subplots(2, 1, figsize=(8,8))

        self.learning_rate = mu
        epochs = np.arange(epochs)
        cost_arr = []
        error_arr = []
        for epoch in epochs:
            cost, error, y_pred = self.forward_propagate(self.Xtr, self.Ytr)
            cost_arr.append(cost)
            # print(error)
            error_arr.append(error)
        axs[0].plot(epochs, error_arr, label="missclassifications")
        axs[1].plot(epochs, cost_arr, label="cost function (MSE)")
        axs[0].legend()
        axs[1].legend()
        plt.show()

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

            
        


