from Linear import *


class LeastMeanSquares(LinearClassifier):
    """
    Linear Clasifier using Least means squares criterion. Inherits from Linear Classifier base class.

    Args:
        Xtr1: array_like
            Trainingdata for class 1
        Xtr2: array_like
            Trainingdata for class 2
    """

    def __init__(self, Xtr1, Xtr2):
        super().__init__(Xtr1, Xtr2)

        self.name = "Least Mean Squares"

    def train(self, rho=0.01, threshold=1e-5):
        self.trained = True

        # Setting previous weights as a zero vector
        prev_w = np.zeros_like(self.w)

        #counting amount of times weights are iterated and adjusted with the complete dataset
        it = 0

        # Checking if the change of  thevalues of the weights are less than a threshold
        while np.any(np.where(np.abs(prev_w - self.w) > threshold, True, False) == True):

            # Setting the previous weights as the newly calculated one.
            prev_w = self.w
            for k in range(len(self.Xtr)):

                x = self.Xtr[k].reshape(3, 1)
                y = self.Ytr[k]
                rho_ = rho / (k+1)

                self.w = self.w + rho_ * x @ (y - x.T @ self.w)
            it += 1
