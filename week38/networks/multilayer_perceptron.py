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

        MSE = np.sum((self.y - Ytr) ** 2, axis=0) / len(Xtr)

        self.error = np.sum((self.y - Ytr), axis=0) / len(Xtr)

        J += np.sum(MSE) / len(MSE)

        return y_hat


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
        for r, layer in enumerate(self.network.layers[:0:-1]):

            if layer.is_output:
                delta_w = -self.learning_rate * np.sum(deltas[r].T @ self.y_arr[r])
            
            else:
                delta_w = np.zeros((len(deltas[r]), 1))
                for i, delta in enumerate(deltas[r]):

                    delta_w_node = -self.learning_rate * np.sum(delta @ self.y_arr[r][:, i])
                    delta_w[i] = delta_w_node.reshape((1,1))
            
            
            # delta_w = self.alpha * self.prev_delta_w[r] - self.learning_rate * np.sum(deltas[r] @ y[r].T)

            # store this weight
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

        errors_arr = []
        for epoch in epochs:
            y_hat = self.forward_propagate(self.Xtr, self.Ytr)
            self.backward_propagate(self.error)
            
            self.Ytr = np.atleast_2d(self.Ytr)

            # print(y_hat.shape, self.Ytr.shape)

            predictions = (np.round(y_hat.T - self.Ytr) == 1)
            errors = len(predictions[predictions])
            errors_arr.append(errors)
        print(errors)
        self.test(self.Xtr, self.Ytr)
        # plt.plot(epochs, errors_arr)
        # plt.show()


    def test(self, Xte, Yte):
        y_hat = self.forward_propagate(Xte, Yte)
        return y_hat

    def plot_training(self):

        x1_range = np.linspace(np.min(self.Xtr[:, 0]), np.max(self.Xtr[:, 0]), 50)
        x2_range = np.linspace(np.min(self.Xtr[:, 1]), np.max(self.Xtr[:, 1]), 50)

        xx, yy = np.meshgrid(x1_range, x2_range)
        
        xx, yy = xx.reshape(len(x1_range)**2), yy.reshape(len(x2_range)**2)
        inp = np.transpose(np.vstack((xx, yy)))
        inp = np.c_[inp, np.ones(inp.shape[0])]
        y = np.atleast_2d(np.zeros(len(inp)))

        y_hat = self.test(inp, y)
        y_hat = y_hat.reshape(len(x1_range), len(x1_range))

        print(np.round(y_hat))

        # plt.contourf(x1_range,x2_range, y_hat, cmap="cool")
        # plt.show()


    def plot_canvas(self):
        # x is the 2-dimensional input data.
        x1 = x2 = np.arange(np.min(x)-0.5, np.max(x)+0.5, 0.1)


        xx, yy = np.meshgrid(x1, x2)  # Create a grid of points.
        # Reshape into a 2-dimensional data-matrix.
        xx, yy = xx.reshape(len(x1)**2), yy.reshape(len(x1)**2)
        inp = np.transpose(np.vstack((xx, yy)))
        inp = np.c_[inp, np.ones(inp.shape[0])]  # Augmented space

        # Prediction of model. Change this line to fit your implementation.
        y_hat, z2, z1 = model.forward_pass(inp)
        # Reshape into a grid for countourf function.
        y_hat = y_hat.reshape(len(x1), len(x1))

        # Prediction of model. Change this line to fit your implementation.
        y_lab, z2, z1 = model.forward_pass(x)

        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.plot(model.loss)
        plt.plot(model.acc)
        plt.subplot(2, 2, 2)
        plt.contourf(x1, x2, y_hat, cmap="cool")
        plt.scatter(x[:, 0], x[:, 1], c=list(np.round(y_lab)))
        plt.subplot(2, 2, 3)
        plt.scatter(z2[:, 0], z2[:, 1], c=list(y))
        plt.subplot(2, 2, 4)
        plt.scatter(y_lab, np.zeros((400)), c=list(np.round(y_lab)))
        plt.tight_layout()
        plt.show()
            
        


