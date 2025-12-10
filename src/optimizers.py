import numpy as np

# ====================================
# Wolfe line search (used in several optimizers)
# ====================================
def wolfe_line_search(f, grad_f, x, p, fx, grad, c1=1e-4, c2=0.9, alpha_init=1.0, alpha_max=50.0, bounds=None):
    alpha = alpha_init
    alpha_lo = 0
    alpha_hi = alpha_max

    phi0 = fx
    dphi0 = grad @ p
    if dphi0 >= 0:
        p = -p
        dphi0 = grad @ p

    while True:
        x_new = x + alpha * p
        if bounds is not None:
            x_new = np.clip(x_new, bounds[:,0], bounds[:,1])
        phi = f(x_new)
        grad_new = grad_f(x_new)
        dphi = grad_new @ p

        if phi > phi0 + c1 * alpha * dphi0:
            alpha_hi = alpha
        elif abs(dphi) <= -c2 * dphi0:
            return alpha, x_new, grad_new
        elif dphi >= 0:
            alpha_hi = alpha
        else:
            alpha_lo = alpha

        alpha = 0.5 * (alpha_lo + alpha_hi)
        if alpha_hi - alpha_lo < 1e-10:
            return alpha, x_new, grad_new

# ====================================
# Gradient-based optimizers with bounds
# ====================================
def AR_gradient_descent(f, grad_f, x0, args=(), tol=1e-6, max_iter=10, alpha=0.5, beta=0.8, c1=1e-4, bounds=None):
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x, *args)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"GD with Armijo Converged at iteration {i}")
            break

        t = 1.0
        fx = f(x, *args)
        while f(x - t * grad, *args) > fx - c1*alpha * grad_norm**2:
            t *= beta
        x = x - t * grad

        if bounds is not None:
            x = np.clip(x, bounds[:,0], bounds[:,1])

        # --- Enforce bounds ---
        if bounds is not None:
            lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
            x = np.clip(x, lower, upper)

        history.append(x.copy())

    return x, history

def SW_gradient_descent(f, grad_f, x0, args=(), tol=1e-6, max_iter=1000, c1=1e-4, c2=0.9, bounds=None):
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x, *args)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"GD with Strong Wolfe Converged at iteration {i}")
            break

        p = -grad
        fx = f(x, *args)
        alpha, x_new, grad_new = wolfe_line_search(
            f=lambda z: f(z, *args),
            grad_f=lambda z: grad_f(z, *args),
            x=x, p=p, fx=fx, grad=grad,
            c1=c1, c2=c2, bounds=bounds
        )

        # --- Enforce bounds ---
        if bounds is not None:
            lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
            x = np.clip(x, lower, upper)
        
        x = x_new
        history.append(x.copy())

    return x, history

def BFGS_armijo(f, grad_f, x0, args=(), tol=1e-6, max_iter=1000, alpha=1.0, beta=0.8, c1=1e-4, bounds=None):
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x, *args)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"[BFGS-Armijo] Converged at iteration {i}")
            break

        p = -H @ grad
        fx = f(x, *args)
        t = alpha
        while f(x + t * p, *args) > fx + c1 * t * grad @ p:
            t *= beta

        x_new = x + t * p
        if bounds is not None:
            x_new = np.clip(x_new, bounds[:,0], bounds[:,1])

        s = x_new - x
        x = x_new
        grad_new = grad_f(x, *args)
        y = grad_new - grad
        ys = y @ s
        if ys > 1e-10:
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        # --- Enforce bounds ---
        if bounds is not None:
            lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
            x = np.clip(x, lower, upper)

        history.append(x.copy())

    return x, history

def BFGS_strong_wolfe(f, grad_f, x0, args=(), tol=1e-6, max_iter=1000, c1=1e-4, c2=0.9, bounds=None):
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x, *args)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"[BFGS-Wolfe] Converged at iteration {i}")
            break

        p = -H @ grad
        fx = f(x, *args)
        alpha, x_new, grad_new = wolfe_line_search(
            f=lambda z: f(z, *args),
            grad_f=lambda z: grad_f(z, *args),
            x=x, p=p, fx=fx, grad=grad,
            c1=c1, c2=c2, bounds=bounds
        )

        s = x_new - x
        x = x_new
        y = grad_new - grad
        ys = y @ s
        if ys > 1e-10:
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        # --- Enforce bounds ---
        if bounds is not None:
            lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
            x = np.clip(x, lower, upper)

        history.append(x.copy())

    return x, history

def CG_armijo(f, grad_f, x0, args=(), tol=1e-6, max_iter=1000, alpha=1.0, beta=0.8, c1=1e-4, bounds=None):
    x = np.array(x0, dtype=float)
    grad = grad_f(x, *args)
    d = -grad
    history = [x.copy()]

    for i in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"[CG-Armijo] Converged at iteration {i}")
            break

        fx = f(x, *args)
        t = alpha
        while f(x + t * d, *args) > fx + c1 * t * grad @ d:
            t *= beta

        x_new = x + t * d
        if bounds is not None:
            x_new = np.clip(x_new, bounds[:,0], bounds[:,1])

        grad_new = grad_f(x_new, *args)
        y = grad_new - grad
        beta_pr = max(0, (grad_new @ y) / (grad @ grad))
        d = -grad_new + beta_pr * d

        x = x_new
        grad = grad_new
        history.append(x.copy())

        # --- Enforce bounds ---
        if bounds is not None:
            lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
            x = np.clip(x, lower, upper)

    return x, history

def CG_strong_wolfe(f, grad_f, x0, args=(), tol=1e-6, max_iter=1000, c1=1e-4, c2=0.9, bounds=None):
    x = np.array(x0, dtype=float)
    grad = grad_f(x, *args)
    d = -grad
    history = [x.copy()]

    for i in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"[CG-Wolfe] Converged at iteration {i}")
            break

        fx = f(x, *args)
        alpha, x_new, grad_new = wolfe_line_search(
            f=lambda z: f(z, *args),
            grad_f=lambda z: grad_f(z, *args),
            x=x, p=d, fx=fx, grad=grad,
            c1=c1, c2=c2, bounds=bounds
        )
        y = grad_new - grad
        beta_pr = max(0, (grad_new @ y) / (grad @ grad))
        d = -grad_new + beta_pr * d

        x = x_new
        grad = grad_new
        history.append(x.copy())

        # --- Enforce bounds ---
        if bounds is not None:
            lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
            x = np.clip(x, lower, upper)

    return x, history
