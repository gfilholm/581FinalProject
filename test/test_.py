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
