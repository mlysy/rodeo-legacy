"""
.. module:: higher_mvCond

Calculates parameters used in kalman_multi_solver.

"""
import numpy as np

from BayesODE.Kalman import higher_mvncond

def multi_mvncond(delta_t, rootlst, sigmalst):
    """
    Calculate wgtState(A), and varState(V) in Y_{n+1} ~ p(Y_{n+1} | Y_n) for
    multivariate processes assuming they are independent.
        
    Parameters
    ----------
    delta_t : ndarray(1)
        A vector containing the step size between simulation points
    rootlst : ndarray(n_dim_var, n_dim_roots)
        A list of roots to n p-th order polynomials of the car(p) process
    sigmalst : ndarray(n_dim_var)
        A list of parameters used in the mOU volatility matrices

    Returns
    -------
    wgtState : ndarray(n_dim_var*n_dim_roots, n_dim_var*n_dim_roots)
        :math:`A = e^{-\Gamma \Delta t}`
    varState : ndarray(n_dim_var*n_dim_roots, n_dim_var*n_dim_roots)
        :math:`V = V_{\Delta t}`
    """
    n = len(rootlst) # number of variables
    p = len(rootlst[0])
    wgtState = np.zeros((n*p, n*p))
    varState = np.zeros((n*p, n*p))
    for i in range(n):
        wgtState[p*i:p*(i+1), p*i:p*(i+1)], varState[p*i:p*(i+1), p*i:p*(i+1)] = higher_mvncond(delta_t, rootlst[i], sigmalst[i])
    return wgtState, varState