import numpy as np


def get_rmse(y, y_pred):
    n = len(y)
    rmse = np.sqrt((y-y_pred)**2/n)
    return rmse


def mse(y, y_pred, diff=False):
    if diff:
        return 2 * (y_pred - y)
    else:
        n = len(y)
        return np.sum((y-y_pred)**2/n)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def diff_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(z, 0)


def diff_relu(z):
    return np.heaviside(z, 1)