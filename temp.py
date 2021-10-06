import numpy as np
import matplotlib.pyplot as plt

mu1 = np.array([0, 0])
cov = np.diag([0.1, 0.4])
N = 100

data = np.random.multivariate_normal(mu1, cov, N)
X, Y = data[:, 0], data[:, 1]

for i, _ in enumerate(X, start=1):

    x = np.arange(X.min(), X.max(), 0.1)

    w1, w0 = np.polyfit(X[:i], Y[:i], deg=1)

    y = w1 * x + w0

    if i % 5 == 0:
        plt.clf()
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.scatter(X[:i], Y[:i])
        plt.plot(x, y, color="red")
        plt.pause(0.01)
plt.show()