"""
.. module:: _mou_car

Calculates parameters for the mOU CAR(p) process.

"""

import numpy as np
from math import exp
        
def _mou_car(roots, sigma=1., test=False):
    """Computes the variance function for the CAR(p) process :math: `var(X_t)`
    
    Parameters
    ----------

    roots: [p] :obj:`numpy.ndarray` of float
        Roots to the p-th order polynomial of the car(p) process (roots must be negative)
    sigma: float
        Parameter in mOU volatility matrix
    test: bool
        If True, return Sigma, and Gamma.

    Returns
    -------
    Gamma: [p, p]  numpy.ndarray
        :math: `\Gamma` in CAR process.

    Sigma: [p, p]  numpy.ndarray
        :math: `\Sigma` in CAR process.
    
    Sigma_tilde: [p, p] numpy.ndarray
        :math: `\widetilde{\Sigma}` in CAR process.

    Q: [p, p] numpy.ndarray
        :math: `Q` in CAR process.
    """

    delta = np.array(-roots)
    D = np.diag(delta)
    p = len(roots)
    Q = np.zeros((p, p))
    row = np.ones(p)
        
    for i in range(p):
        Q[i] = row
        row = row*roots
    Q_inv = np.linalg.pinv(Q)

    if test:
        Sigma = np.zeros((p, p))
        Sigma[p-1, p-1] = sigma*sigma
        Gamma = np.zeros((p,p)) # Q*D*Q^-1
        Gamma[range(p-1),range(1,p)] = -1.
        Gamma[p-1] = np.linalg.multi_dot([Q[p-1], D, Q_inv])
        
        return Gamma, Sigma


    Sigma = np.zeros(p)
    Sigma[p-1] = sigma * sigma
    Sigma_tilde = np.matmul(Q_inv * Sigma, Q_inv.T)
    
    return Sigma_tilde, Q
