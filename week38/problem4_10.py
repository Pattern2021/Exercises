import numpy as np
import matplotlib.pyplot as plt
from networks import Multilayer_perceptron
from time import time


mu11 = np.array([0.4, 0.9]).T
mu12 = np.array([2.0, 1.8]).T
mu13 = np.array([2.3, 2.3]).T
mu14 = np.array([2.6, 1.8]).T
mu21 = np.array([1.5, 1.0]).T
mu22 = np.array([1.9, 1.0]).T
mu23 = np.array([1.5, 3.0]).T
mu24 = np.array([3.3, 2.6]).T

N = 200
S = N // 4

variance = 0.08
covariance_matrices = np.diag((variance, variance))

cls1_vec1 = np.random.multivariate_normal(mu11, covariance_matrices, S)
cls1_vec2 = np.random.multivariate_normal(mu12, covariance_matrices, S)
cls1_vec3 = np.random.multivariate_normal(mu13, covariance_matrices, S)
cls1_vec4 = np.random.multivariate_normal(mu14, covariance_matrices, S)
cls2_vec1 = np.random.multivariate_normal(mu21, covariance_matrices, S)
cls2_vec2 = np.random.multivariate_normal(mu22, covariance_matrices, S)
cls2_vec3 = np.random.multivariate_normal(mu23, covariance_matrices, S)
cls2_vec4 = np.random.multivariate_normal(mu24, covariance_matrices, S)

cls1 = np.array([cls1_vec1, cls1_vec2, cls1_vec3, cls1_vec4]).reshape(N, 2)
cls2 = np.array([cls2_vec1, cls2_vec2, cls2_vec3, cls2_vec4]).reshape(N, 2)

y_cls1 = np.ones(len(cls1))
y_cls2 = np.zeros(len(cls2))

Xtraining = np.concatenate((cls1, cls2), axis=0)
Ytraining = np.atleast_2d(np.concatenate((y_cls1, y_cls2), axis=0))

# network_shape = np.array([2, 3, 1]) # actually kinda works better than using two hidden layers.
network_shape = np.array([2, 3, 2, 1])

learning_rates = np.logspace(-1, -4, 7)
fig, axs = plt.subplots(2, len(learning_rates), figsize=(16, 8), sharey="row")

epochs = 5000

for i, lr in enumerate(learning_rates):
    ins = Multilayer_perceptron(Xtraining, Ytraining, network_shape)
    start = time()
    error = ins.train(lr, epochs=epochs, alpha=0.3)
    stop = time()
    ins.plot_training(axs[1, i], lr, stop - start)
    axs[1, i].scatter(Xtraining[:, 0], Xtraining[:, 1], c=list(Ytraining), s=5)
    axs[0, i].plot(np.arange(epochs), error)
    axs[0, i].set_xlabel("Epochs")
    axs[0, i].set_ylabel("Error")
fig.tight_layout()
plt.show()