import numpy as np
import matplotlib.pyplot as plt

mu1 = np.array([0, 0])
cov = np.diag([0.1, 0.3])
N = 250

data1 = np.random.multivariate_normal(mu1, cov, N)
data2 = np.random.multivariate_normal(mu1 + np.array([1, 0]), cov, N)
X1, Y1 = data1[:, 0], data1[:, 1]
X2, Y2 = data2[:, 0], data2[:, 1]

X = np.concatenate((X1, X2))
Y = np.concatenate((Y1, Y2))
ind = np.arange(len(X))
s_ind = np.random.shuffle(ind)
X = X[ind]
Y = Y[ind]

for i, _ in enumerate(X, start=1):

    x = np.arange(X.min(), X.max(), 0.1)

    w1, w0 = np.polyfit(X[:i], Y[:i], deg=1)

    y = w1 * x + w0

    if i % 5 == 0:
        plt.clf()
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.scatter(X[:i], Y[:i], s=5)
        plt.plot(x, y, color="red")
        plt.pause(0.01)
plt.show()