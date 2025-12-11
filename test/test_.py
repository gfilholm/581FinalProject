import numpy as np
import pytest

# ------------------------------------------------------
# Test objective: convex quadratic with known minimum
# f(x) = (x1 - 3)^2 + (x2 + 1)^2
# Minimum at (3, -1)
# ------------------------------------------------------

def f_test(x):
    return (x[0] - 3)**2 + (x[1] + 1)**2

def grad_f_test(x):
    return np.array([
        2 * (x[0] - 3),
        2 * (x[1] + 1)
    ])

# ------------------------------------------------------
# Import your optimizers
# (adjust the import path depending on your project)
# ------------------------------------------------------
from src.optimizers import (
    AR_gradient_descent,
    SW_gradient_descent,
    BFGS_armijo,
    BFGS_strong_wolfe,
    CG_armijo,
    CG_strong_wolfe,
    wolfe_line_search
)

# ------------------------------------------------------
# Parameterized pytest: test ALL methods automatically
# ------------------------------------------------------

optimizers = [
    AR_gradient_descent,
    SW_gradient_descent,
    BFGS_armijo,
    BFGS_strong_wolfe,
    CG_armijo,
    CG_strong_wolfe
]

@pytest.mark.parametrize("optimizer", optimizers)
def test_optimizer_convergence(optimizer):
    x0 = np.array([0.0, 0.0])
    
    # run optimizer
    best_x, hist = optimizer(f_test, grad_f_test, x0)
    
    # expected solution
    expected = np.array([3.0, -1.0])
    
    # convergence test
    assert np.allclose(best_x, expected, atol=1e-3), f"{optimizer.__name__} failed to converge"



# Gradient and function for Wolfe line search test

def grad_f(x):
    return np.array([2 * (x[0] - 3), 2 * (x[1] + 1)])


def test_wolfe_line_search():
    x0 = np.array([0.0, 0.0])
    p = -grad_f(x0)            # search direction = steepest descent
    fx0 = f_test(x0)
    grad0 = grad_f(x0)

    # Run line search
    alpha, x_new, grad_new = wolfe_line_search(f_test, grad_f, x0, p, fx0, grad0)

    # Compute values for assertions
    phi0 = fx0
    dphi0 = grad0 @ p
    phi = f_test(x_new)
    dphi = grad_new @ p

    # --- Armijo (sufficient decrease) condition ---
    c1 = 1e-4
    assert phi <= phi0 + c1 * alpha * dphi0, "Armijo condition failed"

    # --- Strong Wolfe curvature condition ---
    c2 = 0.9
    assert abs(dphi) <= -c2 * dphi0, "Curvature condition failed"

    # --- Step size should be positive ---
    assert alpha > 0, "Line search returned non-positive step size"

    # --- Function value should decrease ---
    assert phi < phi0, "Function did not decrease"
