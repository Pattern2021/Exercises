import numpy as np
import matplotlib.pyplot as plt

class BayesianClassifier2D:
    def __init__(self, mu0, mu1, sigma1, sigma2, risk=None):
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.risk = risk
        
    def classify(self, x):
        """ Decides whether two dimensional datavector [x1, x2] belongs to class1 or class2
        Args:
            (array) - Containing the datapoint to be classified.
        Returns:
            (Boolean) True for class1, False for class2.
        """

        boundary0 = self.decision_boundary(x[0], x[1], self.mu0[0], self.mu0[1], self.sigma1)
        boundary1 = self.decision_boundary(x[0], x[1], self.mu1[0], self.mu1[1], self.sigma2)

        # If risk is not included execute normally.
        if not np.any(self.risk):
            if boundary0 - boundary1 >= 0:
                return 0
            else:
                return 1
        # else include risk throught given loss matrix.
        else:
            if self.risk[0][1] * boundary0 - self.risk[1][0] * boundary1 >= 0:
                return 0
            else:
                return 1

    @staticmethod
    def decision_boundary(x1, x2, mu0, mu1, sigma):
        return - 1 / (2 * sigma ** 2) * (x1 ** 2 + x2 ** 2) + 1 / (sigma ** 2) * (mu0 * x1 + mu1 * x2) \
            - 1 / (2 * sigma ** 2) * (mu0 ** 2 + mu1 ** 2)

    def test(self, n=1000):
        N = n // 2
        true_class0 = np.random.normal(self.mu0, self.sigma1, size=(N, 2))
        true_class1 = np.random.normal(self.mu1, self.sigma2, size=(N, 2))
        result_class0 = np.array(list(map(self.classify, true_class0)))
        result_class1 = np.array(list(map(self.classify, true_class1)))
        error_class0 = len(result_class0[result_class0 == 1])
        error_class1 = len(result_class1[result_class1 == 0])

        # Accuracy of classifier given by 1 - total error / total datapoints.
        accuracy = 1 - (error_class0 + error_class1) / n

        # Coordinates of misclassified datapoints.
        error_point0 = true_class0[np.where(result_class0 == 1)]
        error_point1 = true_class1[np.where(result_class1 == 0)]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(true_class0[:, 0], true_class0[:, 1], c="blue", label="Class 0")
        ax.scatter(true_class1[:, 0], true_class1[:, 1], c="red", label="Class 1")
        ax.scatter(error_point0[:, 0], error_point0[:, 1], c="cyan", label="True 0 classified as 1")
        ax.scatter(error_point1[:, 0], error_point1[:, 1], c="magenta", label="True 1 classified as 0")
        ax.scatter(np.array([[self.mu0[0]], [self.mu1[0]]]), np.array([[self.mu0[1]], [self.mu1[1]]]), c="green", label="True mean")
        ax.legend(loc="upper left")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"Test with accuracy of {accuracy:.3f}, of n = {n} points. With loss matrix $\Lambda = [[0 \; 1] \; [0.5 \; 0]]$")
        fig.tight_layout()
        plt.show()
        # plt.savefig("test_result_b_new_mean2.png")



if __name__ == "__main__":
    mu0 = np.array([1, 1])
    mu1 = np.array([1.5, 1.5])
    sigma1 = .2
    sigma2 = .2
    loss_matrix = np.array([[0, 1], [0.5, 0]])
    cl = BayesianClassifier2D(mu0, mu1, sigma1, sigma2, loss_matrix)
    cl.test()
    # mu2 = np.array([3, 3])
    # cl2 = BayesianClassifier2D(mu0, mu2, sigma1, sigma2, loss_matrix)
    # cl2.test()
