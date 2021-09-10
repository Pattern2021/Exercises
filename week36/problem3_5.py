import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(1)

    mu1 = np.array([1, 1])
    mu2 = np.array([0, 0])
    sigma1 = .2
    sigma2 = sigma1

    data1 = np.random.normal(mu1, sigma1, size=(50, 2))
    data2 = np.random.normal(mu2, sigma2, size=(50, 2))

if __name__ == '__main__':
    main()