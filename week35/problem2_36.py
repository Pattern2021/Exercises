import numpy as np
import matplotlib.pyplot as plt

class ParzenWindow:
    def __init__(self, radius, h, class0, class1):
        self.R = radius
        self.class0 = class0
        self.class1 = class1
        self.N0 = len(self.class0[:, 0])
        self.N1 = len(self.class1[:, 0])
        self.h = h

    def est_pdf(self):
        # pdf0 = np.sum(np.array(list(map(self.gaussian_pulse, self.class0))))
        # pdf1 = np.sum(np.array(list(map(self.gaussian_pulse, self.class1))))
        a = np.random.uniform(0, 2, size=(self.N0, 2))
        b = np.array(list(map(self.pdf, a)))
        c = np.linspace(-1, 3, 50)
        plt.plot(c, b)
        plt.show()
        plt.scatter(self.class0[:, 0], self.class0[:, 1])
        plt.show()
        

    def pdf(self, x):
        return 1 / (self.h * self.N1) * np.sum(np.array(list(map(self.gaussian_pulse, (self.class0 - x) / self.h))))

    @staticmethod
    def gaussian_pulse(x):
        return np.sqrt(2 * np.pi) * np.exp(-0.5 * np.matmul(x.T, x))


    def classify(self, point):
        """ 
        Classifies a point to either class 0 or class 1 based on the amount of points within a constant radius of the point.

        Args:
             (array or array-like) - point to be classified.
        Returns:
             (int) - either 0 for first class or 1 for second class.
        """
        # Find distances from new vector of the classes vectors.
        dist_c0 = self.calc_dist(point, self.class0)
        dist_c1 = self.calc_dist(point, self.class1)

        # Find all points withing radius
        within_r_c0 = dist_c0[dist_c0 < self.R]
        within_r_c1 = dist_c1[dist_c1 < self.R]

        # Count all points within R
        n_within_r_c0 = len(within_r_c0)
        n_within_r_c1 = len(within_r_c1)

        print(n_within_r_c0, n_within_r_c1)

    @staticmethod
    def calc_dist(x1, x2):
        return np.linalg.norm(x1 - x2, axis=1)

def main():
    np.random.seed(3)
    mu1, mu2 = np.array([1, 1]), np.array([1.5, 1.5])
    sigma1 = .2
    sigma2 = sigma1
    N = 100
    class1 = np.random.normal(mu1, sigma1, size=(N // 2, 2))
    class2 = np.random.normal(mu2, sigma2, size=(N // 2, 2))
    c1 = ParzenWindow(5, 0.25, class1, class2)
    c1.est_pdf()

if __name__ == "__main__":
    main()
