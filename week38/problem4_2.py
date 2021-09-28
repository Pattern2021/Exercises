import numpy as np
import matplotlib.pyplot as plt
from networks.multilayer_perceptron import Multilayer_perceptron


def main():
    # np.random.seed(1)

    mu1 = np.array([0, 0])
    mu2 = np.array([1, 1])
    mu3 = np.array([0, 1])
    mu4 = np.array([1, 0])

    covariance = np.diag((0.01, 0.01))

    N = 100
    cls1_vec1 = np.random.normal(mu1, np.diag(covariance), size=(N, 2))
    cls1_vec2 = np.random.normal(mu2, np.diag(covariance), size=(N, 2))
    cls2_vec1 = np.random.normal(mu3, np.diag(covariance), size=(N, 2))
    cls2_vec2 = np.random.normal(mu4, np.diag(covariance), size=(N, 2))

    # cls1_vec1 = np.random.multivariate_normal(mu1, covariance, size=(N, 2))
    # cls1_vec2 = np.random.multivariate_normal(mu2, covariance, size=(N, 2))
    # cls2_vec1 = np.random.multivariate_normal(mu3, covariance, size=(N, 2))
    # cls2_vec2 = np.random.multivariate_normal(mu4, covariance, size=(N, 2))

    cls1 = np.concatenate((cls1_vec1, cls1_vec2), axis=0)
    cls2 = np.concatenate((cls2_vec1, cls2_vec2), axis=0)
    y1 = np.ones(len(cls1[:, 0]))
    y2 = np.zeros(len(cls2[:, 0]))

    # plt.plot(cls1[:, 0], cls1[:, 1], "b.")
    # plt.plot(cls2[:, 0], cls2[:, 1], "r.")
    # plt.show()

    Xtraining = np.concatenate((cls1, cls2), axis=0)
    Ytraining = np.atleast_2d(np.concatenate((y1, y2), axis=0))
    network_shape = np.array([2, 2, 1])

    ins = Multilayer_perceptron(Xtraining, Ytraining, network_shape)
    ins.train(0.1, epochs=1, alpha=0.75)
    # ins.plot_training()
    # Xtest, Ytest = ins.shuffle(Xtraining, Ytraining)
    # Xtest = ins.onecolumn(Xtest)
    # Xtest = Xtest[:30]
    # Ytest = Ytest[:30]
    # ins.test(Xtest, Ytest)

if __name__ == "__main__":
    main()
