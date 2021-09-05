import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
xwidth = 20

kernel_x = np.arange(-xwidth, xwidth, 0.1)
bw_manual = 1

def gauss_const(h):
    return 1 / (h * np.sqrt(np.pi * 2))

def gauss_exp(ker_x, xi, h):
    num = - 0.5 * np.square((xi - ker_x))
    den = h * h
    return num / den

def kernel_function(h, ker_x, xi):
    const = gauss_const(h)
    gauss_val = const * np.exp(gauss_exp(ker_x, xi, h))
    return gauss_val

def res(input):
    res = []
    for i in input:
        res.append(kernel_function(bw_manual, kernel_x, i))
    res = np.array(res)
    res = np.sum(res, axis=0)
    return res

input1 = np.random.uniform(-20, 10, size=(10,))
# input2 = np.random.uniform(-10, 20, size=(10,))
res1 = res(input1)
# res2 = res(input2)
plt.plot(kernel_x, res1, c="blue")
# plt.plot(kernel_x, res2, c="red")
plt.show()