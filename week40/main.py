from networks import Multilayer_perceptron
import numpy as np
import matplotlib.pyplot as plt

est_func = lambda x: 0.3 + 0.2 * np.cos(2 * np.pi * x)

x = np.arange(0, 1, 0.01)

xx, yy = np.meshgrid(x, x)

xx, yy = xx.reshape(len(x)**2), yy.reshape(len(x)**2)
Xtrain = np.c_[xx, yy]
ytrain = np.where(est_func(Xtrain[:, 0]) > Xtrain[:, 1], 1, 0).reshape(1, -1)

# network_shape = np.array([2, 2, 2, 1])
# lr = 0.005
network_shape = np.array([2, 4, 4, 1])
lr = 0.003

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

clf = Multilayer_perceptron(Xtrain, ytrain, network_shape)
error = clf.train(lr, alpha=0.2, epochs=5000)
clf.plot_training(axs[1], lr)
axs[1].plot(x, est_func(x), color="black", label="func to be estimated")
axs[1].legend()
axs[0].plot(clf.epochs, error)
plt.show()
