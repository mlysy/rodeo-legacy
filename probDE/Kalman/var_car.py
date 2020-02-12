"""
.. module:: var_car

Variance function for the CAR(p) process:

.. math:: var(X_t)

"""
import numpy as np

import probDE.Kalman._mou_car as mc

def var_car(tseq, roots, sigma=1.):
    """
    Computes the variance function for the CAR(p) process :math: `var(X_t)`
    
    Parameters
    ----------
    tseq: ndarray(n_timesteps)
        Time points at which :math:`x_t` is evaluated. 
    roots: ndarray(n_dim_roots)
        Roots to the p-th order polynomial of the car(p) process (roots must be negative)
    sigma: float
        Parameter in mOU volatility matrix

    Returns
    -------
    var: ndarray(n_timesteps, n_dim_roots, n_dim_roots)
        Evaluates :math:`var(X_t)`.
    """
    p = len(roots)
    Sigma_tilde, Q = mc._mou_car(roots, sigma)
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
