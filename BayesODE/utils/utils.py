"""
.. module:: utils

Helpful functions used in kalman.
"""
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
    X : ndarray(n_dim1, n_dim2)
        Matrix X in :math:`X = V^{-1}B`
    """
    L, low = scl.cho_factor(V)
    return scl.cho_solve((L, low), B)
