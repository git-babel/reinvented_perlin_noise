from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np


def generateRandomGradientForGrid(grid):
    shape = grid.shape
    dim = len(shape)

    rand_nums = 1
    for n in shape:
        rand_nums *= n

    grads = np.random.rand(rand_nums * dim).reshape((*shape, dim))
    grads = grads / np.linalg.norm(grads, ord=2, axis=dim, keepdims=True)
    return grads


def dotGrid(grid, grads):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            pivot = np.random.rand(2)
            grid[i][j] = np.dot(grads[i][j], pivot)


def main(n, dim):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    grid = np.zeros(tuple(n for _ in range(dim)))
    grads = generateRandomGradientForGrid(grid)
    dotGrid(grid, grads)

    xs = np.arange(n)
    ys = np.arange(n)
    xs, ys = np.meshgrid(xs, ys)
    zs = grid

    ax.plot_surface(xs, ys, zs)
    plt.show()


main(10, 2)
