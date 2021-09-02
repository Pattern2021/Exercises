import numpy as np
import matplotlib.pyplot as plt

class kNearestNeighbor2Class:
    def __init__(self, k, class0, class1):
        # identify number of classes
        self.k = k
        self.class0 = class0
        self.class1 = class1


    def classify(self, point):
        """
        Classifies point to either class 0 or class 1 based on euclidean distance between point and k nearest points.
        
        Args: 
             (array or array-like) - point to be classified.
        Returns:
             (int) - either 0 for first class or 1 for second class.
        """
        
        # Find distances from new vector to each of the classes vectors.
        dist_c0 = self.calc_dist(point, self.class0)
        dist_c1 = self.calc_dist(point, self.class1)

        # Combine distances
        dist = np.array([dist_c0, dist_c1])
        
        # Sort in an increasing order the two arrays by indices
        sorted_ind_coord = np.array(np.unravel_index(np.argsort(dist, axis=None), dist.shape)).T
        
        # Find k nearest neighbors
        k_nearest_class = sorted_ind_coord[:self.k, 0]

        # Assign class to given point based on the amount of points of k nearest
        classification = np.where(len(k_nearest_class[k_nearest_class == 0]) > len(k_nearest_class[k_nearest_class == 1]), 0, 1)
        
        return classification

    def test(self, mu0, mu1, sigma0, sigma1, n):

        # Create n sample points of each distribution
        c0 = np.random.normal(mu0, sigma0, size=(n, 2))
        c1 = np.random.normal(mu1, sigma1, size=(n, 2))

        # Classify the sample points
        test_c0 = np.array(list(map(self.classify, c0)))
        test_c1 = np.array(list(map(self.classify, c1)))

        # Compare correct classifications 
        correct_c0 = len(test_c0[test_c0 == 0])
        correct_c1 = len(test_c1[test_c1 == 1])

        # Find the points which was missclassified.
        missed_c0 = c0[np.where(test_c0 != 0)]
        missed_c1 = c1[np.where(test_c1 != 1)]

        # Calculate accuracy of algorithm
        accuracy = (correct_c0 + correct_c1) / (n*2)

        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.scatter(c0[:, 0], c0[:, 1], c="blue", label="Class 0")
        ax.scatter(c1[:, 0], c1[:, 1], c="red", label="Class 1")
        ax.scatter(missed_c0[:, 0], missed_c0[:, 1], c="cyan", label="Class 0 classified as 1")
        ax.scatter(missed_c1[:, 0], missed_c1[:, 1], c="magenta", label="Class 1 classified as 0")
        ax.scatter(np.array([mu0, mu1])[:, 0], np.array([mu0, mu1])[:, 1], c="green", label="True mean")
        ax.set_title(f"Test of 3 Nearest Neighbor algorithm with {n} samples of each class.\nAccuracy = {accuracy:.3f}")
        ax.legend(loc="upper left")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        # plt.savefig("test_result_3nn.png")
        plt.show()

    @staticmethod
    def calc_dist(x1, x2):
        return np.linalg.norm(x1 - x2, axis=1)




def main():
    # np.random.seed(1)
    mu1, mu2 = np.array([1, 1]), np.array([1.5, 1.5])
    sigma1 = .2
    sigma2 = sigma1
    N = 100

    class1 = np.random.normal(mu1, sigma1, size=(N // 2, 2))
    class2 = np.random.normal(mu2, sigma2, size=(N // 2, 2))

    c1 = kNearestNeighbor2Class(3, class1, class2)
    c1.test(mu1, mu2, sigma1, sigma2, N*10)
    
if __name__ == "__main__":
    main()
