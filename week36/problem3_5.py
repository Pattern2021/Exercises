import numpy as np
import matplotlib.pyplot as plt

from classifiers.Linear import LinearClassifier

class Perceptron(LinearClassifier):
    def __init__(self, Xtr1, Xtr2):
        super().__init__(Xtr1, Xtr2)

        # storing indices of data array
        self.indices = np.arange(len(self.Xtr))

        self.name = "Perceptron"

    def train(self, rho, max_epochs=1000):
        self.trained = True

        # Set previous weights as array of zeros
        prev_w = np.zeros_like(self.w)

        # counter for iterations within for-loop
        t = 0

        # counter for when we approach the max epochs
        epochs = 0

        # looping through while counter is less than lenght of data and total iterations is less than maximum epochs
        while t < len(self.Xtr) and epochs < max_epochs:
            t = 0

            for i in range(len(self.Xtr)):

                # reshape data to correct form
                x = self.Xtr[i].reshape(3, 1)

                # Multiplying weights with data
                mult = np.matmul(self.w.T, x)

                # current label
                lab = self.Ytr[i]

                # missclassified of class1 from current weights, update weights
                if lab == 1 and mult <= 0:
                    self.w = self.w + rho * x
                
                # missclassified of class2 from current weights, update weights
                elif lab == -1 and mult >= 0:
                    self.w = self.w - rho * x
                
                # correctly classified, do nothing
                else:
                    t += 1
            epochs += 1

def main():
    mu1 = np.array([1, 1])
    mu2 = np.array([0, 0])
    sigma1 = .2
    sigma2 = sigma1

    data1 = np.random.normal(mu1, sigma1, size=(100, 2))
    data2 = np.random.normal(mu2, sigma2, size=(100, 2))

    ins = Perceptron(data1, data2)
    ins.train(1)
    ins.plot_training()
    test1 = np.random.normal(mu1, sigma1, size=(30, 2))
    test2 = np.random.normal(mu2, sigma2, size=(30, 2))
    ins.test(test1, test2)
    ins.plot_testing()


if __name__ == '__main__':
    main()
