import numpy as np

data = np.loadtxt('data/FFD.csv', delimiter=',')

t_k = data[:, 0]
v_k = data[:, 1]


# 1. Model function: just computes W(t)
def model_function(params, t):
    A0, A, tau, omega, alpha, phi = params
    return A0 + A * np.exp(-t / tau) * np.sin((omega + alpha * t) * t + phi)


def objective_function(params, t, V):
    A0, A, tau, omega, alpha, phi = params
    W = A0 + A * np.exp(-t / tau) * np.sin((omega + alpha * t) * t + phi)
    residuals = W - V
    return np.sum(residuals**2)

def gradient_objective(params, t, V):
    A0, A, tau, omega, alpha, phi = params
    E = np.exp(-t / tau)
    psi = (omega + alpha * t) * t + phi
    S = np.sin(psi)
    C = np.cos(psi)

    W = A0 + A * E * S
    residuals = W - V

    dA0 = 2 * np.sum(residuals)
    dA = 2 * np.sum(residuals * E * S)
    dTau = 2 * np.sum(residuals * A * S * E * (t / tau**2))
    dOmega = 2 * np.sum(residuals * A * E * C * t)
    dAlpha = 2 * np.sum(residuals * A * E * C * t**2)
    dPhi = 2 * np.sum(residuals * A * E * C)

    return np.array([dA0, dA, dTau, dOmega, dAlpha, dPhi])

