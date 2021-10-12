import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from time import time

sys.path.insert(0, os.path.abspath(r"C:\Users\Christian Salomonsen\OneDrive - UiT Office 365\Desktop\UIT\FYS-3012\exercises"))

from LinearClassifiers import LinearClassifier

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

def main():
    mu1 = np.array([1, 1])
    mu2 = np.array([1.5, 1.5])
    sigma1 = .2
    sigma2 = sigma1

    data1 = np.random.normal(mu1, sigma1, size=(100, 2))
    data2 = np.random.normal(mu2, sigma2, size=(100, 2))

    ins = SumOfErrorSquares(data1, data2)
    ins.train()
    ins.plot_training()

    test1 = np.random.normal(mu1, sigma1, size=(30, 2))
    test2 = np.random.normal(mu2, sigma2, size=(30, 2))

    ins.test(test1, test2)
    ins.plot_testing()

if __name__ == "__main__":
    main()
