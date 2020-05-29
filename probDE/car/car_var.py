import numpy as np

from probDE.car.car_mou import car_mou

def car_var(tseq, roots, sigma=1.):
    """
    Computes the variance function for the CAR(p) process :math:`var(X_t)`
    
    Args:
        tseq (ndarray(n_timesteps)): Time points at which :math:`x_t` is evaluated. 
        roots (ndarray(n_dim_roots)): Roots to the p-th order polynomial of the car(p) process.
        sigma (float): Parameter in mOU volatility matrix.

    Returns:
        (ndarray(n_timesteps, n_dim_roots, n_dim_roots)): Evaluates :math:`var(X_t)`.

    """
    p = len(roots)
    Sigma_tilde, Q = car_mou(roots, sigma)
    var = np.zeros((p, p, len(tseq)), order='F')
    for t in range(len(tseq)):
        V_tilde = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                V_tilde[i, j] = Sigma_tilde[i, j] / (roots[i] + roots[j]) * (
                    1.0 - np.exp(-(roots[i] + roots[j]) * tseq[t]))  # V_tilde
                V_tilde[j, i] = V_tilde[i, j]

        var[:, :, t] = np.linalg.multi_dot([Q, V_tilde, Q.T])  # V_deltat

    return var
