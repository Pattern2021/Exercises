from networks import Multilayer_perceptron
import numpy as np
import matplotlib.pyplot as plt

est_func = lambda x: 0.3 + 0.2 * np.cos(2 * np.pi * x)

x = np.arange(0, 1, 0.01)

xx, yy = np.meshgrid(x, x)

xx, yy = xx.reshape(len(x)**2), yy.reshape(len(x)**2)
Xtrain = np.c_[xx, yy]
ytrain = np.where(est_func(Xtrain[:, 0]))
# plt.plot(x, est_func(x))
# plt.show()