import numpy as np
import matplotlib.pyplot as plt
from time import time

from classifiers.Linear import LinearClassifier

class Perceptron(LinearClassifier):
    def __init__(self, Xtr1, Xtr2):
        super().__init__(Xtr1, Xtr2)

        # storing indices of data array
        self.indices = np.arange(len(self.Xtr))

    def train(self, rho, max_epochs=1000):
        self.start_time = time()
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
        self.end_time = time()
        return self.w
    
    def test(self, Xte1, Xte2):
        self.tested = True

        # Test data
        Xte1 = np.append(Xte1, np.ones((len(Xte1), 1)), axis=1)
        Xte2 = np.append(Xte2, np.ones((len(Xte2), 1)), axis=1)
        lab1 = np.ones(len(Xte1))
        lab2 = np.ones(len(Xte2)) * -1
        Xte = np.concatenate((Xte1, Xte2))
        lab = np.concatenate((lab1, lab2))
        Xte, lab = self.shuffle(Xte, lab)

        # indices of classified test data
        i1 = []
        i2 = []
        for i, x in enumerate(Xte):
            mul = self.w.T @ x
            if mul > 0:
                i1.append(i)
            else:
                i2.append(i)
        i1 = np.asarray(i1)
        i2 = np.asarray(i2)

        classified_c1 = lab[i1]
        classified_c2 = lab[i2]
        correct_c1 = classified_c1[classified_c1 == 1]
        correct_c2 = classified_c2[classified_c2 == -1]
        false_c1 = classified_c1[classified_c1 == -1]
        false_c2 = classified_c2[classified_c2 == 1]
        c1 = Xte[i1]
        c2 = Xte[i2]

        # Calculating the accuracy of the model
        self.accuracy = (len(correct_c1) + len(correct_c2)) / len(Xte)
        plt.scatter(c1[:, 0], c1[:, 1])
        plt.scatter(c2[:, 0], c2[:, 1])

    def plot_results(self):
        self.is_trained()
        x1_vals = np.append(self.Xtr1[:, 0], self.Xtr2[:, 0])
        xarr = np.arange(np.min(x1_vals), np.max(x1_vals), 0.1)
        w0 = self.w[2][0]
        w1 = self.w[1][0]
        w2 = self.w[0][0]
        slope = (-w0 / w2) / (w0 / w1)
        intercept = -w0/w2
        plt.plot(xarr, slope * xarr + intercept)
        #plt.scatter(self.data1[:, 0], self.data1[:, 1])
        #plt.scatter(self.data2[:, 0], self.data2[:, 1])
        title = "Perceptron algorithm decision boundary for N = {0:} datapoints,\ntraining took ={1:.3f} s".format(len(self.Xtr), self.end_time - self.start_time)
        if self.tested:
            title += "\nAccuracy = {:.3f}".format(self.accuracy)
            
        plt.title(title)
        plt.show()



def main():
    # np.random.seed(1)

    mu1 = np.array([1, 1])
    mu2 = np.array([0, 0])
    sigma1 = .2
    sigma2 = sigma1

    data1 = np.random.normal(mu1, sigma1, size=(100, 2))
    data2 = np.random.normal(mu2, sigma2, size=(100, 2))

    ins = Perceptron(data1, data2)
    ins.train(1)
    test1 = np.random.normal(mu1, sigma1, size=(30, 2))
    test2 = np.random.normal(mu2, sigma2, size=(30, 2))
    ins.test(test1, test2)
    ins.plot_results()

if __name__ == '__main__':
    main()
