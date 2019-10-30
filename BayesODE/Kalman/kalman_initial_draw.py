"""
.. module:: kalman_initial_draw

Computes the initial draw X0 for the kalman process. 

"""
import numpy as np

from BayesODE.Kalman.cov_car import cov_car
from BayesODE.utils.utils import mvncond

def kalman_initial_draw(roots, sigma, x0, p):
    """
    Computes the initial draw X0State for the kalman process :math:`X0 ~ N(c0, 0_{pxp})`
    where :math:`c0 ~ p(N(\lambda, V_{\infty}) | x0 = x0)

    Parameters
    ----------
    roots: ndarray(n_dim_roots)
        Roots to the p-th order polynomial of the car(p) process (roots must be negative)
    sigma: float
        Parameter in mOU volatility matrix
    x0: ndarray(n_dim_ode)
        Initial conditions of the ode
    p: float
        Size of X0State

    Returns
    -------
    X0State: ndarray(n_dim_roots)
        Simulate :math:`Y0 ~ N(0, V_{\infty}) conditioned on x0.

    """
    q = len(x0) - 1
    if p == q+1:
        return x0
        
    X0State = np.zeros(p)    #Initialize p sized initial X0
    V_inf = cov_car([], roots, sigma, v_infinity=True)    #Get V_inf to draw X^{{q+1} ... (p-1)}_0
    icond = np.array([True]*(q+1) + [False]*(p-q-1))   #Conditioned on all but X^{{q+1} ... (p-1)}_0
    A, b, V = mvncond(np.zeros(p), V_inf, icond)    #Get mean and variance for p(z_0 | y_0 = c) where z_0 = X_0 \setminus y_0
    z_0 = np.random.multivariate_normal(A.dot(x0) + b, V)    #Draw X^{(p-1)}_0
    X0State[:q+1] = x0    #Replace x^{(0), (1), (2) ... (q)}}_0 with y_0
    X0State[q+1:] = z_0
    return X0State
