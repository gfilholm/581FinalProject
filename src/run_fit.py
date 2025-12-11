#!/usr/bin/env python3
from optimizers import BFGS_strong_wolfe
from data.process import t_k, v_k, model_function, objective_function, gradient_objective
import numpy as np

# -----------------------------
# Initial guess
# -----------------------------
x0 = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

# -----------------------------
# Run optimizer
# -----------------------------
best_params, history = BFGS_strong_wolfe(objective_function, gradient_objective, x0, args=(t_k, v_k))

# -----------------------------
# Output
# -----------------------------
print("Optimized parameters:", best_params)
print("Final objective value:", objective_function(best_params, t_k, v_k))

# Save fitted curve
fitted_W = model_function(best_params, t_k)
np.savetxt('data/fitted_curve.csv', np.column_stack([t_k, fitted_W]), delimiter=',')
print("Fitted curve saved to data/fitted_curve.csv")
