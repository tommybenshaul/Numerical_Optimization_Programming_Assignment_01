# src/utils.py

import matplotlib.pyplot as plt
import numpy as np

def plot_contour(f, x_range, y_range, title, paths=None, file_name=None):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([xx, yy])) for xx in x] for yy in y])

    plt.figure()
    cp = plt.contour(X, Y, Z, levels=np.logspace(-2, 3, 20))
    plt.colorbar(cp)
    if paths:
        for method, path in paths.items():
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], marker='o', linestyle='--', label=method)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    if file_name:
        plt.savefig(file_name)
    plt.close()

def plot_function_values(iterations, values, title, file_name=None):
    plt.figure()
    for method, vals in values.items():
        # Ensure the lengths of iterations and values match
        iterations_to_plot = iterations[:len(vals)]
        plt.plot(iterations_to_plot, vals, marker='o', linestyle='--', label=method)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    if file_name:
        plt.savefig(file_name)
    plt.close()
