import numpy as np
import matplotlib.pyplot as plt
from time import time

class LMS:
    def __init__(self, Xtr1, Xtr2):
        self.Xtr1, self.Xtr2 = Xtr1, Xtr2
        self.Ytr = np.concatenate((np.ones(len(Xtr1)), np.ones(len(Xtr2)) * -1))
        self.Xtr = np.concatenate((Xtr1, Xtr2))
        self.Xtr = np.append(self.Xtr, np.ones((len(self.Xtr), 1)), axis=1)
        self.Xtr, self.labels = self.shuffle(self.Xtr, self.Ytr)

        self.trained = False
        self.tested = False

        self.w = np.random.uniform(size=(3, 1))

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

                self.w = self.w + rho * x @ (y - x.T @ self.w)
            it += 1
        print(it)

        self.w = self.w.flatten()
        w2 = self.w[0]
        w1 = self.w[1]
        w0 = self.w[2]
        slope = (-w0 / w2) / (w0 / w1)
        intercept = -w0 / w2
        xarr = np.arange(np.min(self.Xtr[:, 0]), np.max(self.Xtr[:, 0]), 0.1)
        plt.plot(xarr, slope * xarr + intercept)
        plt.scatter(self.Xtr1[:, 0], self.Xtr1[:, 1])
        plt.scatter(self.Xtr2[:, 0], self.Xtr2[:, 1])
        plt.show()

    def test(self):
        self.tested = True
        pass

    def is_trained(self):
        if not self.trained:
            print("Not yet trained")
            exit()

    def is_tested(self):
        if not self.tested:
            print("Not yet tested")
            exit()

    @staticmethod
    def shuffle(data, labels):
        ind = np.arange(len(data))
        np.random.shuffle(ind)
        data = data[ind]
        labels = labels[ind]
        return data, labels        

def main():
    np.random.seed(1)

    mu1 = np.array([1, 1])
    mu2 = np.array([3, 3])
    sigma1 = .2
    sigma2 = sigma1

    data1 = np.random.normal(mu1, sigma1, size=(100, 2))
    data2 = np.random.normal(mu2, sigma2, size=(100, 2))

    ins = LMS(data1, data2)
    ins.train()

if __name__ == '__main__':
    main()