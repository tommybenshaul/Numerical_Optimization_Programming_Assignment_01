# tests/test_unconstrained_min.py

import unittest
import numpy as np
import os
from src.unconstrained_min import minimize
from src.utils import plot_contour, plot_function_values
from tests.examples import quadratic_circles, quadratic_axis_aligned_ellipses, quadratic_rotated_ellipses, rosenbrock, linear, exp_function

# Constants for tolerances
OBJ_TOL = 1e-12
PARAM_TOL = 1e-8

# Constants for maximum iterations
MAX_ITER_DEFAULT = 100
MAX_ITER_ROSENBROCK = 10000

# Constants for initial points
INITIAL_POINTS = {
    "default": np.array([1.0, 1.0]),
    "rosenbrock": np.array([-1.0, 2.0])
}

# Constants for plot ranges
PLOT_RANGE_X_DEFAULT = (-3, 3)
PLOT_RANGE_Y_DEFAULT = (-3, 3)
PLOT_RANGE_X_ROSENBROCK = (-2, 2)
PLOT_RANGE_Y_ROSENBROCK = (-1, 3)
PLOT_RANGE_X_EXP = (-2, 2)
PLOT_RANGE_Y_EXP = (-2, 2)

# Constants for function names
QUADRATIC_CIRCLES = "1 Quadratic Circles"
QUADRATIC_AXIS_ALIGNED_ELLIPSES = "2 Quadratic Axis-Aligned Ellipses"
QUADRATIC_ROTATED_ELLIPSES = "3 Quadratic Rotated Ellipses"
ROSENBROCK_FUNCTION = "4 Rosenbrock Function"
LINEAR_FUNCTION = "5 Linear Function"
EXPONENTIAL_FUNCTION = "6 Exponential Function"

# Constants for Wolfe conditions
C1 = 0.01
BETA = 0.5
C2 = 0.9
FIXED_STEP_LENGTH = 0.1

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        # List of examples with their specific initial points and maximum iterations
        self.examples = [
            (QUADRATIC_CIRCLES, quadratic_circles, PLOT_RANGE_X_DEFAULT, PLOT_RANGE_Y_DEFAULT, INITIAL_POINTS["default"], MAX_ITER_DEFAULT),
            (QUADRATIC_AXIS_ALIGNED_ELLIPSES, quadratic_axis_aligned_ellipses, PLOT_RANGE_X_ROSENBROCK, PLOT_RANGE_X_ROSENBROCK, INITIAL_POINTS["default"], MAX_ITER_DEFAULT),
            (QUADRATIC_ROTATED_ELLIPSES, quadratic_rotated_ellipses, PLOT_RANGE_X_DEFAULT, PLOT_RANGE_Y_DEFAULT, INITIAL_POINTS["default"], MAX_ITER_DEFAULT),
            (ROSENBROCK_FUNCTION, rosenbrock, PLOT_RANGE_X_ROSENBROCK, PLOT_RANGE_Y_ROSENBROCK, INITIAL_POINTS["rosenbrock"], MAX_ITER_ROSENBROCK),
            (LINEAR_FUNCTION, linear, PLOT_RANGE_X_DEFAULT, PLOT_RANGE_Y_DEFAULT, INITIAL_POINTS["default"], MAX_ITER_DEFAULT),
            (EXPONENTIAL_FUNCTION, exp_function, PLOT_RANGE_X_EXP, PLOT_RANGE_Y_EXP, INITIAL_POINTS["default"], MAX_ITER_DEFAULT)
        ]

        self.methods = [
            ("gradient_descent", True),
            ("newtons_method", True)
            # ("gradient_descent", False),
            # ("newtons_method", False)
        ]
        self.final_results = []

    def test_minimization(self):
        for name, func, x_range, y_range, x0, max_iter in self.examples:
            print(f"\n------ Testing {name} ------")
            paths = {}
            values = {}
            for method, use_wolfe in self.methods:
                print(f"### Testing {name} using method: {method} (use_wolfe={use_wolfe}) ###")
                x_opt, f_opt, success, path, value = minimize(
                    func, x0, OBJ_TOL, PARAM_TOL, max_iter, method=method, use_wolfe=use_wolfe,
                    c1=C1, beta=BETA, c2=C2, fixed_step_length=FIXED_STEP_LENGTH
                )
                paths[f"{method}_use_wolfe={use_wolfe}"] = path
                values[f"{method}_use_wolfe={use_wolfe}"] = value
                self.final_results.append((name, method, use_wolfe, x_opt, f_opt, success))

            plot_contour(lambda x: func(x)[0], x_range, y_range, title=f"{name} Contour Plot",
                         paths=paths, file_name=f"plots/{name.replace(' ', '_')}_contour.png")
            if values:
                num_iterations = max(len(v) for v in values.values())
                plot_function_values(list(range(num_iterations)), values, title=f"{name} Function Values",
                                     file_name=f"plots/{name.replace(' ', '_')}_values.png")

    def tearDown(self):
        print("\n------ Final Results ------")
        with open("final_results.txt", "w") as file:
            for result in self.final_results:
                name, method, use_wolfe, x_opt, f_opt, success = result
                file.write(f"{name} ({method}, use_wolfe={use_wolfe}) - Final iterate: {x_opt}, Final objective value: {f_opt}, Success: {success}\n")
                print(f"{name} ({method}, use_wolfe={use_wolfe}) - Final iterate: {x_opt}, Final objective value: {f_opt}, Success: {success}")

if __name__ == '__main__':
    unittest.main()
