import numpy as np
import matplotlib.pyplot as plt
from time import time

class Perceptron:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        label1 = np.ones(len(data1))
        label2 = np.ones(len(data2)) * -1
        labels = np.concatenate((label1, label2))
        data1 = np.append(data1, np.ones((len(data1), 1)), axis=1)
        data2 = np.append(data2, np.ones((len(data2), 1)), axis=1)
        data = np.concatenate((data1, data2))
        self.indices = np.arange(len(data))
        self.data, self.labels = self.shuffle(data, labels)
        self.w = np.random.uniform(size=(3, 1))
        self.trained = False

    def train(self, rho, epochs=1000):
        self.start_time = time()
        self.trained = True
        prev_w = np.zeros_like(self.w)
        c1 = 0
        c2 = 0
        c1_err = 0
        c2_err = 0
        t = 0
        iters = 0
        while t < len(self.data) and iters < epochs:
            t = 0
            for i in range(len(self.data)):
                x = self.data[i].reshape(3, 1)

                mult = np.matmul(self.w.T, x)
                lab = self.labels[i]

                # missclassified of class1, update weights
                if lab == 1 and mult <= 0:
                    self.w = self.w + rho * x
                
                # missclassified of class2, update weights
                elif lab == -1 and mult >= 0:
                    c2_err += 1
                    self.w = self.w - rho * x
                
                # correctly classified, do nothing
                else:
                    c1 += 1
                    c2 += 1
                    t += 1
            iters += 1
        self.end_time = time()
        return self.w 

    def plot_results(self):
        if not self.trained:
            print("Model not been trained! Use class.train(rho)")
        else:
            x1_vals = np.append(self.data1[:, 0], self.data2[:, 0])
            xarr = np.arange(np.min(x1_vals), np.max(x1_vals), 0.1)
            w0 = self.w[2][0]
            w1 = self.w[1][0]
            w2 = self.w[0][0]
            slope = (-w0 / w2) / (w0 / w1)
            intercept = -w0/w2
            plt.plot(xarr, slope * xarr + intercept)
            plt.scatter(self.data1[:, 0], self.data1[:, 1])
            plt.scatter(self.data2[:, 0], self.data2[:, 1])
            plt.title("Perceptron algorithm decision boundary for N = {0:} datapoints,\ntraining took ={1:.3f} s".format(len(self.data), self.end_time - self.start_time))
            plt.show()

    @staticmethod
    def shuffle(data, labels):
        ind = np.arange(len(data))
        np.random.shuffle(ind)
        data = data[ind]
        labels = labels[ind]
        return data, labels



def main():
    #np.random.seed(1)

    mu1 = np.array([1, 1])
    mu2 = np.array([0, 0])
    sigma1 = .2
    sigma2 = sigma1

    data1 = np.random.normal(mu1, sigma1, size=(50, 2))
    data2 = np.random.normal(mu2, sigma2, size=(50, 2))

    ins = Perceptron(data1, data2)
    weights = ins.train(0.01)
    ins.plot_results()

if __name__ == '__main__':
    main()
