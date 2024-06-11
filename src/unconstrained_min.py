# src/unconstrained_min.py

import numpy as np

# Debug flag
DEBUG = False


def log_debug(message):
    if DEBUG:
        print(message)


def backtracking_line_search(f, grad_f, x, p, c1=0.01, beta=0.5, c2=0.9, max_iterations=100):
    t = 1
    iteration = 0
    log_debug(f"Backtracking line search start: x={x}, p={p}, f(x)={f(x)[0]}, grad_f(x)={grad_f(x)}")

    while iteration < max_iterations and f(x + t * p)[0] > f(x)[0] + c1 * t * np.dot(grad_f(x), p):
        t *= beta
        iteration += 1
        log_debug(f"Backtracking line search (sufficient decrease): iteration={iteration}, t={t}")

    iteration = 0
    while iteration < max_iterations and np.dot(grad_f(x + t * p), p) < c2 * np.dot(grad_f(x), p):
        t *= beta
        iteration += 1
        log_debug(f"Backtracking line search (curvature condition): iteration={iteration}, t={t}")

    log_debug(f"Backtracking line search end: final t={t}")
    return t


def gradient_descent(f, grad_f, x0, obj_tol, param_tol, max_iter, use_wolfe=True, c1=0.01, beta=0.5, c2=0.9,
                     fixed_step_length=0.1):
    x = x0
    obj_values = [f(x)[0]]
    iteration_path = [x]
    for i in range(max_iter):
        grad = grad_f(x)
        log_debug(f"Gradient descent iteration {i}: x={x}, grad={grad}")
        if use_wolfe:
            step_length = backtracking_line_search(f, grad_f, x, -grad, c1, beta, c2)
        else:
            step_length = fixed_step_length
        x_new = x - step_length * grad
        iteration_path.append(x_new)
        obj_values.append(f(x_new)[0])
        print(f"Iteration {i}: x_new = {x_new}, f(x_new) = {f(x_new)[0]}, step_length = {step_length}")
        if np.linalg.norm(x_new - x) < param_tol or abs(f(x_new)[0] - f(x)[0]) < obj_tol:
            print(f"Converged on iteration {i} with x = {x_new}, f(x) = {f(x_new)[0]}, success = True")
            return x_new, f(x_new)[0], True, iteration_path, obj_values
        x = x_new
    print(f"Failed to converge after {max_iter} iterations with x = {x}, f(x) = {f(x)[0]}, success = False")
    return x, f(x)[0], False, iteration_path, obj_values


def newtons_method(f, grad_f, hess_f, x0, obj_tol, param_tol, max_iter, use_wolfe=True, c1=0.01, beta=0.5, c2=0.9,
                   fixed_step_length=0.1):
    x = x0
    obj_values = [f(x)[0]]
    iteration_path = [x]
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        log_debug(f"Newton's method iteration {i}: x={x}, grad={grad}, hess={hess}")
        try:
            delta_x = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            print(f"Iteration {i}: Singular matrix encountered, falling back to gradient descent step.")
            delta_x = -grad
        if use_wolfe:
            step_length = backtracking_line_search(f, grad_f, x, delta_x, c1, beta, c2)
        else:
            step_length = fixed_step_length
        x_new = x + step_length * delta_x
        iteration_path.append(x_new)
        obj_values.append(f(x_new)[0])
        print(f"Iteration {i}: x_new = {x_new}, f(x_new) = {f(x_new)[0]}, step_length = {step_length}")
        if np.linalg.norm(delta_x) < param_tol or abs(f(x_new)[0] - f(x)[0]) < obj_tol:
            print(f"Converged on iteration {i} with x = {x_new}, f(x) = {f(x_new)[0]}, success = True")
            return x_new, f(x_new)[0], True, iteration_path, obj_values
        x = x_new
    print(f"Failed to converge after {max_iter} iterations with x = {x}, f(x) = {f(x)[0]}, success = False")
    return x, f(x)[0], False, iteration_path, obj_values


def minimize(f, x0, obj_tol, param_tol, max_iter, method="gradient_descent", use_wolfe=True, c1=0.01, beta=0.5, c2=0.9,
             fixed_step_length=0.1):
    log_debug(f"Starting minimization with method: {method} (use_wolfe={use_wolfe})")
    grad_f = lambda x: f(x)[1]
    if method == "gradient_descent":
        return gradient_descent(f, grad_f, x0, obj_tol, param_tol, max_iter, use_wolfe, c1, beta, c2, fixed_step_length)
    elif method == "newtons_method":
        hess_f = lambda x: f(x, True)[2]
        return newtons_method(f, grad_f, hess_f, x0, obj_tol, param_tol, max_iter, use_wolfe, c1, beta, c2,
                              fixed_step_length)
    else:
        raise ValueError("Invalid method specified. Use 'gradient_descent' or 'newtons_method'.")
