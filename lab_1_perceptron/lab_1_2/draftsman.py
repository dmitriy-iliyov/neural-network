import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def print_sigmoid():
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    plt.plot(x, y, label='sigmoid', color='red')
    plt.title("sigmoid")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()
