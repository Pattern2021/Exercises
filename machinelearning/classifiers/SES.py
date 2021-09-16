from Linear import *


class SumOfErrorSquares(LinearClassifier):
    """
    Linear Clasifier using Sum of error squares criterion. Inherits from Linear Classifier base class.

    Args:
        Xtr1: array_like
            Trainingdata for class 1
        Xtr2: array_like
            Trainingdata for class 2
    """

    def __init__(self, Xtr1, Xtr2):
        super().__init__(Xtr1, Xtr2)

        self.name = "Sum of Error Squares"

    def train(self):
        self.trained = True

        self.w = np.linalg.inv(self.Xtr.T @ self.Xtr) @ self.Xtr.T @ self.Ytr
