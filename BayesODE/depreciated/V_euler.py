import sys
import os
import numpy as np
import BayesODE.Kalman._mou_car as mc

def V_euler(roots, sigma, delta_t, N, B, X0=None):
    """
    Stochastic Euler approximation to the mOU variance.
    
    Parameters
    ----------
    Gamma : [p x p] numpy.ndarray
        mOU mean reversion matrix.
    Sigma : [p x p] numpy.ndarray
        mOU volatility matrix (positive-definite).
    roots: [p] :obj:`numpy.ndarray` of float
        Roots to the p-th order polynomial of the car(p) process (roots must be negative)
    sigma: float
        Parameter in mOU volatility matrix
    delta_t : float
        Time interval between simulation points.
    N : int
        Number of simulation points.
    B : int
        Number of time series to calculate the variance from.
    X0 : [B x p] numpy.ndarray
        Optional starting point for the simulations.
    
    Returns
    -------
    V_t : [p x p] numpy.ndarray
        Monte Carlo estimate of `var(X_t | X_0 = x_0)`.
    """
    Gamma, Sigma = mc._mou_car(roots, sigma, True)

    p = Gamma.shape[0]
    GammaT = Gamma.T
    X = np.zeros((N+1,B,p))
    if X0 is not None: X[0,:,:] = X0
    # simulate random draws
    Z = np.random.multivariate_normal(mean=np.zeros(p), 
                                      cov=delta_t*Sigma,
                                      size=(N,B))
    # step through
    for n in range(N):
        X[n+1] = X[n] - X[n].dot(GammaT)*delta_t + Z[n]
    return np.cov(X[N], rowvar=False)