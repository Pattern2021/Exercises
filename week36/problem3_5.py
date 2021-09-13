import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        label1 = np.zeros(len(data1))
        label2 = np.ones(len(data2))
        labels = np.concatenate((label1, label2))
        data1 = np.append(data1, np.ones((len(data1), 1)), axis=1)
        data2 = np.append(data2, np.ones((len(data2), 1)), axis=1)
        data = np.concatenate((data1, data2))
        self.indices = np.arange(len(data))
        self.data, self.labels = self.shuffle(data, labels)
        self.w = np.random.uniform(size=(3, 1))

    def train(self, rho):
        t = 0
        prev_w = np.array([0, 0, 0])
        run = True
        c1 = 0
        c2 = 0
        while run and t < len(self.data):
            print(np.any(np.where(prev_w == self.w, True, False) == False), self.w, prev_w)
            if not np.any(np.where(prev_w == self.w, True, False) == False):
                break
            prev_w = self.w
            x = self.data[t].reshape(3, 1)
            mult = np.matmul(self.w.T, x)
            lab = self.labels[t]
            if lab == 0 and mult <= 0:
                print("c1")
                c1 += 1
                self.w = self.w + rho * x
            elif lab == 1 and mult >= 0:
                print("c2")
                c2 += 1
                self.w = self.w - rho * x
            else:
                print("None", mult, lab)
                prev_w = np.array([0,0,0])

            t += 1
            print(t, self.w)
            xarr = np.arange(-1, 2, 0.01)
            # print(xarr, self.w[0][2] / self.w[0][0] * xarr + self.w[0][2] / self.w[0][1])

            plt.plot(xarr, -self.w[2][0] / self.w[0][0]*xarr - self.w[2][0] / self.w[1][0])
        plt.plot(xarr, -self.w[2][0] / self.w[0][0] * xarr - self.w[2][0] / self.w[1][0], "o-")

        plt.scatter(self.data1[:, 0], self.data1[:, 1])
        plt.scatter(self.data2[:, 0], self.data2[:, 1])
        print(c1,c2)
        plt.show()
        return self.w  

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
    mu2 = np.array([0, 0])
    sigma1 = .2
    sigma2 = sigma1

    data1 = np.random.normal(mu1, sigma1, size=(50, 2))
    data2 = np.random.normal(mu2, sigma2, size=(50, 2))

    ins = Perceptron(data1, data2)
    weights = ins.train(0.01)
    
    # xarr = np.arange(-1, 2, 0.01)
    # print(xarr, weights[0][2] / weights[0][0] * xarr + weights[0][2] / weights[0][1])

    # plt.plot(xarr, -weights[0][2] / weights[0][0]*xarr - weights[0][2] / weights[0][1])
    # plt.scatter(data1[:, 0], data1[:, 1])
    # plt.scatter(data2[:, 0], data2[:, 1])
    # plt.show()

if __name__ == '__main__':
    main()
