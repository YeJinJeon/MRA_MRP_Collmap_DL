import pylab as plt
import numpy as np


def norm_range(x, a, b):
    return (b - a) * ((x - x.min()) / (x.max() - x.min())) + a


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def take_root(x, root=1.2):
    return np.power(x, 1 / root)


def linear_decay(x, x_max, x_min):
    return (x_min - x_max) * x + x_max


def exp_decay(x, alpha=1.5, beta=10):
    # z = np.linspace(0, 1, n)
    y = alpha ** (-beta * x) + 1
    y = norm_range(y, 1, alpha)

    return y


def decayed_root(x, max_root=1.2, method='exp'):
    """

    """
    if method == 'exp':
        root_nums = exp_decay(x, max_root, 10)
    else:
        root_nums = linear_decay(x, max_root, 1)
    y = np.power(x, 1 / root_nums)
    return y


def decayed_discontinued(x):
    # shape = x.shape
    # x = x.flatten()
    y = np.zeros_like(x)
    thr = 0.2
    for i, _ in enumerate(x):
        if x[i] < thr:
            y[i] = 1.2 * x[i]
        else:
            y[i] = 0.95 * x[i] + 0.05
    return y


def decayed_discontinued_2d(x):
    thr = 0.2
    mask = x < thr
    x[mask == 1] *= 1.2
    x[mask == 0] *= 0.95
    x[mask == 0] += 0.05
    return x


def draw1():
    root = 1.15
    x = np.linspace(0, 1, 100)
    # y0 = take_root(x, root)
    y1 = decayed_root(x, root)
    y2 = decayed_discontinued(x)
    plt.plot(x, x)
    # plt.plot(x, y0, 'orange')
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'g')
    plt.show()


def draw2():
    max_root = 1.3
    n = 100
    x = np.linspace(0, 1, n)
    y0 = take_root(x, max_root)
    y1 = decayed_root(x, max_root, 'linear')
    y2 = decayed_root(x, max_root * 1.5, 'exp')
    plt.plot(x, x)
    plt.plot(x, y0, 'orange')
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'g')
    plt.legend(['Linear (Normal)', 'Root', 'Linear Decayed Root', 'Exponential Decayed Root'])
    plt.show()


def draw2():
    max_root = 1.2
    n = 100
    x = np.linspace(0, 1, n)
    y0 = take_root(x, max_root)
    y1 = decayed_root(x, max_root, 'linear')
    y2 = decayed_root(x, max_root, 'exp')
    plt.plot(x, x)
    plt.plot(x, y0, 'orange')
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'g')
    plt.legend(['Linear (Normal)', 'Root', 'Linear Decayed Root', 'Exponential Decayed Root'])
    plt.show()


if __name__ == '__main__':
    draw2()
