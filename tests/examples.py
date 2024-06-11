import numpy as np
from src.unconstrained_min import minimize
from src.utils import plot_contour, plot_function_values

# Quadratic Circles: f(x) = 0.5 * x^T Q x, where Q = [[1, 0], [0, 1]]
def quadratic_circles(x, eval_hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    f_value = 0.5 * x.T @ Q @ x
    grad = Q @ x
    hess = Q if eval_hessian else None
    return f_value, grad, hess

# Quadratic Axis-Aligned Ellipses: f(x) = 0.5 * x^T Q x, where Q = [[1, 0], [0, 100]]
def quadratic_axis_aligned_ellipses(x, eval_hessian=False):
    Q = np.array([[1, 0], [0, 100]])
    f_value = 0.5 * x.T @ Q @ x
    grad = Q @ x
    hess = Q if eval_hessian else None
    return f_value, grad, hess

# Quadratic Rotated Ellipses: f(x) = 0.5 * x^T Q x, where Q is a rotated matrix
def quadratic_rotated_ellipses(x, eval_hessian=False):
    Q = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]]).T @ np.array([[100, 0], [0, 1]]) @ np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    f_value = 0.5 * x.T @ Q @ x
    grad = Q @ x
    hess = Q if eval_hessian else None
    return f_value, grad, hess

# Rosenbrock Function: f(x) = 100 * (x[1] - x[0]^2)^2 + (1 - x[0])^2
def rosenbrock(x, eval_hessian=False):
    f_value = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    grad = np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])
    hess = np.array([
        [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ]) if eval_hessian else None
    return f_value, grad, hess

# Linear Function: f(x) = a^T x, where a = [1, 2]
def linear(x, eval_hessian=False):
    a = np.array([1, 2])
    f_value = a.T @ x
    grad = a
    hess = np.zeros((len(x), len(x))) if eval_hessian else None
    return f_value, grad, hess

# Exponential Function: f(x) = exp(x[0] + 3*x[1] - 0.1) + exp(x[0] - 3*x[1] - 0.1) + exp(-x[0] - 0.1)
def exp_function(x, eval_hessian=False):
    f_value = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)
    grad = np.array([
        np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1),
        3 * np.exp(x[0] + 3*x[1] - 0.1) - 3 * np.exp(x[0] - 3*x[1] - 0.1)
    ])
    hess = None if not eval_hessian else np.array([
        [np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1), 3 * (np.exp(x[0] + 3*x[1] - 0.1) - np.exp(x[0] - 3*x[1] - 0.1))],
        [3 * (np.exp(x[0] + 3*x[1] - 0.1) - np.exp(x[0] - 3*x[1] - 0.1)), 9 * (np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1))]
    ])
    return f_value, grad, hess
