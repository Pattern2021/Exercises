import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

np.random.seed(2)

def unknown_pdf(x):
    if x > 0 and x < 2:
        return 0.5
    else:
        return 0


x = np.random.uniform(0, 2, size=(5000,1))
y = np.array(list(map(unknown_pdf, x)))


class ParzenWindow:
    def __init__(self, x, b):
        self.x = x
        self.b = b
        self.N = len(self.x)
 
    def prob(self):
        dist = squareform(pdist(self.x))
        pulsenum = self.gaussian_pulse(dist / (self.b ** 2))
        
        return 1 / self.N * np.sum(pulsenum, axis=1)

    def gaussian_pulse(self, x):
        return 1 / (np.sqrt(2 * np.pi) * self.b) * np.exp(- 0.5 * x)


c1 = ParzenWindow(x, .2)

b1 = c1.prob()

plt.scatter(x, b1)
plt.plot(x, y)
plt.show()
