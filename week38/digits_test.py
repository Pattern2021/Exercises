import numpy as np
import matplotlib.pyplot as plt

from networks import Multilayer_perceptron
from sklearn.datasets import load_digits

X, y = load_digits(n_class=2, return_X_y=True)

y = y.reshape(1, -1)

network_structure = np.array([64, 16, 1])

clf = Multilayer_perceptron(X, y, network_structure)
clf.train()

clf.test(X)