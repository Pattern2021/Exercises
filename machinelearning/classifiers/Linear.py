import numpy as np
import matplotlib.pyplot as plt
from time import time

class LinearClassifier:
    """
    Base class for all self made linear classifiers
    
    Parameters:
        Xtr1: array_like
            Trainingdata for class 1
        Xtr2: array_like
            Trainingdata for class 2
    """

    def __init__(self, Xtr1, Xtr2, weights=None):
        self.Xtr1, self.Xtr2 = Xtr1, Xtr2
        self.Ytr = np.concatenate((np.ones(len(Xtr1)), np.ones(len(Xtr2)) * -1))
        self.Xtr = np.concatenate((Xtr1, Xtr2))
        self.Xtr = np.append(self.Xtr, np.ones((len(self.Xtr), 1)), axis=1)
        self.Xtr, self.Ytr = self.shuffle(self.Xtr, self.Ytr)

        self.trained = False
        self.tested = False

        # If list is not none, init accepts weight.
        if not weights is None and isinstance(weights, np.ndarray):
            self.w = weights
        else:
            self.w = np.random.uniform(size=(3, 1))

    def test(self, Xte1, Xte2):
        """
        Test method for all supervised linear learning methods using weights

        Parameters:
            Xte1: array_like
                Test data from class 1
            Xte2: array_like
                Test data from class 2
        """
        self.tested = True

        class_label1 = 1
        class_label2 = -1

        # Fixing test data
        self.Xte1 = np.append(Xte1, np.ones((len(Xte1), 1)), axis=1)
        self.Xte2 = np.append(Xte2, np.ones((len(Xte2), 1)), axis=1)
        
        Yte1 = np.ones(len(Xte1)) * class_label1
        Yte2 = np.ones(len(Xte2)) * class_label2

        
        self.Xte = np.concatenate((Xte1, Xte2))
        self.Yte = np.atleast_2d(np.concatenate((Yte1, Yte2)))
        
        indice1 = np.arange(len(Xte1))
        indice2 = np.arange(len(Xte2))

        mul1 = np.atleast_2d(self.Xte1 @ self.w).reshape(Yte1.shape)
        mul2 = np.atleast_2d(self.Xte2 @ self.w).reshape(Yte2.shape)

        # I think there is a bug in here: sometimes points are not correctly classified!
        self.correct_c1 = indice1[mul1 > 0]
        self.correct_c2 = indice2[mul2 <= 0]
        self.false_c1 = indice1[mul1 <= 0]
        self.false_c2 = indice2[mul2 > 0]

        self.accuracy = (len(self.correct_c1) + len(self.correct_c2)) / len(self.Yte[0])

    def is_trained(self):
        """ Method for checking and handling errors if user is inappropriatly trying to plot training or test before training. """
        # if not self.trained:
        #     print("Model not been trained! Use class.train(rho)")
        #     exit()
        try:
            if not self.trained:
                raise Exception("Model must be trained!")
        except Exception as e:
            print(e)
            exit()

    def is_tested(self):
        """ Method for checking and handling errors if user is inappropriatly trying to plot the test before testing. """
        try:
            if not self.tested:
                raise Exception("Model must be tested!")
        except Exception as e:
            print(e)
            exit()

    def decision_boundary_to_plot(self):
        """ Adds the decision boundary to current plot. """
        self.is_trained()

        self.w = self.w.flatten()
        w2 = self.w[0]
        w1 = self.w[1]
        w0 = self.w[2]
        slope = (-w0 / w2) / (w0 / w1)
        intercept = -w0 / w2
        xarr = np.arange(np.min(self.Xtr[:, 0]), np.max(self.Xtr[:, 0]), 0.1)

        self.axs.plot(xarr, slope * xarr + intercept, label="Decision Boundary")
    
    def plot_training(self):
        self.is_trained()

        self.fig, self.axs = plt.subplots(figsize=(7,7))

        self.decision_boundary_to_plot()
        self.test(self.Xtr1, self.Xtr2)

        self.axs.scatter(self.Xtr1[:, 0], self.Xtr1[:, 1], label="Class 1")
        self.axs.scatter(self.Xtr2[:, 0], self.Xtr2[:, 1], label="Class 2")
        self.axs.set_xlabel("x1")
        self.axs.set_ylabel("x2")
        self.axs.legend(loc="upper left")
        self.axs.set_title("Training results using {0:} algorithm\nUsing N = {1:} training samples.\nTraining has accuracy {2:.3f}".format(self.name, len(self.Xtr), self.accuracy))
        self.fig.tight_layout()
        plt.show()

    def plot_testing(self):
        self.is_tested()

        self.fig, self.axs = plt.subplots(figsize=(7,7))

        self.decision_boundary_to_plot()

        x_correct_c1 = self.Xte1[:len(self.correct_c1), 0]
        y_correct_c1 = self.Xte1[:len(self.correct_c1), 1]
        x_correct_c2 = self.Xte2[:len(self.correct_c2), 0]
        y_correct_c2 = self.Xte2[:len(self.correct_c2), 1]
        x_false_c1 = self.Xte1[:len(self.false_c1), 0]
        y_false_c1 = self.Xte1[:len(self.false_c1), 1]
        x_false_c2 = self.Xte2[:len(self.false_c2), 0]
        y_false_c2 = self.Xte2[:len(self.false_c2), 1]

        self.axs.scatter(x_correct_c1, y_correct_c1, label="Correct classified c1")
        self.axs.scatter(x_correct_c2, y_correct_c2, label="Correct classified c2")

        if len(x_false_c1) > 0:
            self.axs.scatter(x_false_c1, y_false_c1, label="False classified c1")
        if len(x_false_c2) > 0:
            self.axs.scatter(x_false_c2, y_false_c2, label="False classified c2")
        
        self.axs.set_xlabel("x1")
        self.axs.set_ylabel("x2")
        self.axs.legend(loc="upper left")
        self.axs.set_title("Test results using {0:} algorithm\nUsing N = {1:} test samples.\nTest has accuracy of {2:.3f}".format(self.name, len(self.Xte[:, 0]), self.accuracy))
        self.fig.tight_layout()
        plt.show()

    @staticmethod
    def shuffle(data, labels):
        ind = np.arange(len(data))
        np.random.shuffle(ind)
        data = data[ind]
        labels = labels[ind]
        return data, labels
