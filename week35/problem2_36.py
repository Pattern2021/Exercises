import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)


def unknown_pdf(x):
    if x > 0 and x < 2:
        return 0.5
    else:
        return 0


x_vals = np.arange(-1, 3, .1)
x_data = np.array(list(map(unknown_pdf, x_vals)))


class ParzenWindow:
    def __init__(self, x_i, b):
        self.x_i = x_i
        self.b = b
        self.N = len(self.x_i)

    def prob(self, x):
        pulsenum = self.gaussian_pulse((self.x_i - x) / self.b)
        print(pulsenum[0])
        return 1 / (self.b * self.N) * np.sum(pulsenum, axis=1)

    @staticmethod
    def pulse(x):
        return np.where(np.abs(x) <= .5, 1, 0)

    def gaussian_pulse(self, x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(- 0.5 * np.dot(x.T, x))


c1 = ParzenWindow(x_data, 1)
x = np.random.uniform(-5, 6, size=(3, 1))

b1 = c1.prob(x)
# print(b1)
# plt.plot(x_vals, b1)
# plt.plot(x_vals, x_data)
# plt.show()
