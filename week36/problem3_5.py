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
        self.tested = False

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
    
    def test(self, Xte1, Xte2):
        self.tested = True
        Xte1 = np.append(Xte1, np.ones((len(Xte1), 1)), axis=1)
        Xte2 = np.append(Xte2, np.ones((len(Xte2), 1)), axis=1)
        lab1 = np.ones(len(Xte1))
        lab2 = np.ones(len(Xte2)) * -1
        Xte = np.concatenate((Xte1, Xte2))
        lab = np.concatenate((lab1, lab2))
        Xte, lab = self.shuffle(Xte, lab)

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
        self.accuracy = (len(correct_c1) + len(correct_c2)) / len(Xte)
        plt.scatter(c1[:, 0], c1[:, 1])
        plt.scatter(c2[:, 0], c2[:, 1])


    def is_trained(self):
        if not self.trained:
            print("Model not been trained! Use class.train(rho)")
            exit()
    
    def is_tested(self):
        if not self.tested:
            print("Model has not been tested! Use class.test()")
            exit()

    def plot_results(self):
        self.is_trained()
        x1_vals = np.append(self.data1[:, 0], self.data2[:, 0])
        xarr = np.arange(np.min(x1_vals), np.max(x1_vals), 0.1)
        w0 = self.w[2][0]
        w1 = self.w[1][0]
        w2 = self.w[0][0]
        slope = (-w0 / w2) / (w0 / w1)
        intercept = -w0/w2
        plt.plot(xarr, slope * xarr + intercept)
        #plt.scatter(self.data1[:, 0], self.data1[:, 1])
        #plt.scatter(self.data2[:, 0], self.data2[:, 1])
        title = "Perceptron algorithm decision boundary for N = {0:} datapoints,\ntraining took ={1:.3f} s".format(len(self.data), self.end_time - self.start_time)
        if self.tested:
            title += "\nAccuracy = {:.3f}".format(self.accuracy)
            
        plt.title(title)
        plt.show()

    @staticmethod
    def shuffle(data, labels):
        ind = np.arange(len(data))
        np.random.shuffle(ind)
        data = data[ind]
        labels = labels[ind]
        return data, labels



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
