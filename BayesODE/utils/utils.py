"""
.. module:: utils

Helpful functions used in kalman.
"""
from math import exp
import numpy as np
import scipy.linalg as scl

def mvncond(mu, Sigma, icond):
    """
    Calculates A, b, and V such that :math:`y[~icond] | y[icond] ~ N(A y[icond] + b, V)`.

    Parameters
    ----------
    mu: ndarray(2*n_dim)
        Mean of y
    Sigma: ndarray(2*n_dim, 2*n_dim)
        Covariance of y 
    icond: ndarray(2*nd_dim)
        Conditioning on the terms given
    
    Returns
    -------
    A : ndarray(n_dim, n_dim)
        For :math:`y ~ N(\mu, \Sigma)` such that :math:`y[~icond] | y[icond] ~ N(A y[icond] + b, V)` Calculate A.
    b : ndarray(n_dim)
        For :math:`y ~ N(\mu, \Sigma)` such that :math:`y[~icond] | y[icond] ~ N(A y[icond] + b, V)` Calculate b.
    V : ndarray(n_dim, n_dim)
        For :math:`y ~ N(\mu, \Sigma)` such that :math:`y[~icond] | y[icond] ~ N(A y[icond] + b, V)` Calculate V.

    """
    # if y1 = y[~icond] and y2 = y[icond], should have A = Sigma12 * Sigma22^{-1}
    A = np.dot(Sigma[np.ix_(~icond, icond)],scl.cho_solve(scl.cho_factor(Sigma[np.ix_(icond,icond)]), np.identity(sum(icond))))
    b = mu[~icond] - np.dot(A, mu[icond]) # mu1 - A * mu2
    V = Sigma[np.ix_(~icond,~icond)] - np.dot(A, Sigma[np.ix_(icond,~icond)]) # Sigma11 - A * Sigma21
    return A, b, V

def solveV(V, B):
    """
    Computes :math:`X = V^{-1}B` where V is a variance matrix.

    Parameters
    ----------
    V : ndarray(n_dim1, n_dim1)
        Variance matrix V in :math:`X = V^{-1}B`.
    B : ndarray(n_dim1, n_dim2)
        Matrix B in :math:`X = V^{-1}B`.
    
    Returns
    -------
    X : ndarray(n_dim1, n_dim2)
        Matrix X in :math:`X = V^{-1}B`

    """
    L, low = scl.cho_factor(V)
    return scl.cho_solve((L, low), B)

def root_gen(tau, p):
    """
    Creates p geometrically decaying CAR model roots.

    Parameters
    ----------
    tau : int
        Decorrelation parameter
    p : int
        Number of roots to generate

    Returns
    -------
    roots : ndarray(p)
        Vector of roots generated using p geometrically decaying roots

    """
    # return -np.append(1/tau, np.linspace(1, 1.1, p-1))
    roots = np.zeros(p)
    for k in range(p):
        if k == 0:
            roots[k] = -1/tau
        else:
            roots[k] = -exp((k+1)/tau)

    return roots

def zero_pad(x0, p):
    """
    Pad x0 with p-len(x0) 0s at the end of x0.

    Parameters
    ----------
    x0 : ndarray(n_dim)
        Any vector 
    p : int
        Size of the padded vector

    Returns
    -------
    X0 : ndarray(1, p)
        Padded vector of length p

    """
    q = len(x0)
    X0 = np.array([np.pad(x0, (0, p-q), 'constant', constant_values=(0,0))])
    return X0
