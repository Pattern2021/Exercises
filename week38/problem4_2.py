import numpy as np
import matplotlib.pyplot as plt
from networks.multilayer_perceptron import Multilayer_perceptron


# np.random.seed(6969)

mu1 = np.array([0, 0])
mu2 = np.array([1, 1])
mu3 = np.array([0, 1])
mu4 = np.array([1, 0])

covariance = np.diag((0.01, 0.01))

N = 100

cls1_vec1 = np.random.multivariate_normal(mu1, covariance, N)
cls1_vec2 = np.random.multivariate_normal(mu2, covariance, N)
cls2_vec1 = np.random.multivariate_normal(mu3, covariance, N)
cls2_vec2 = np.random.multivariate_normal(mu4, covariance, N)


cls1 = np.concatenate((cls1_vec1, cls1_vec2), axis=0)
cls2 = np.concatenate((cls2_vec1, cls2_vec2), axis=0)
y1 = np.ones(len(cls1[:, 0]))
y2 = np.zeros(len(cls2[:, 0]))


Xtraining = np.concatenate((cls1, cls2), axis=0)
Ytraining = np.atleast_2d(np.concatenate((y1, y2), axis=0))
hidden_neurons = 6
network_shape = np.array([2, hidden_neurons, 1])

learning_rates = np.logspace(-1.5, -2, 6)
fig, axs = plt.subplots(2, len(learning_rates), figsize=(16, 8), sharey="row")

for i, lr in enumerate(learning_rates):
    ins = Multilayer_perceptron(Xtraining, Ytraining, network_shape)
    ins.train(lr, epochs=1000, alpha=0, ax=axs[0, i])
    ins.plot_training(axs[1, i], lr)
fig.suptitle("2-class neural network with one hidden layer of {} neurons".format(hidden_neurons))
fig.tight_layout()
plt.show()
