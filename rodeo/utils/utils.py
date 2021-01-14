from math import exp
import numpy as np
import scipy.linalg as scl
from timeit import default_timer as timer

def mvncond(mu, Sigma, icond):
    """
    Calculates A, b, and V such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)`.

    Args:
        mu (ndarray(2*n_dim)): Mean of y.
        Sigma (ndarray(2*n_dim, 2*n_dim)): Covariance of y. 
        icond (ndarray(2*nd_dim)): Conditioning on the terms given.

    Returns:
        (tuple):
        - **A** (ndarray(n_dim, n_dim)): For :math:`y \sim N(\mu, \Sigma)` 
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate A.
        - **b** (ndarray(n_dim)): For :math:`y \sim N(\mu, \Sigma)` 
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate b.
        - **V** (ndarray(n_dim, n_dim)): For :math:`y \sim N(\mu, \Sigma)`
          such that :math:`y[!icond] | y[icond] \sim N(A y[icond] + b, V)` Calculate V.

    """
    # if y1 = y[~icond] and y2 = y[icond], should have A = Sigma12 * Sigma22^{-1}
    A = np.dot(Sigma[np.ix_(~icond, icond)], scl.cho_solve(
        scl.cho_factor(Sigma[np.ix_(icond, icond)]), np.identity(sum(icond))))
    b = mu[~icond] - np.dot(A, mu[icond])  # mu1 - A * mu2
    V = Sigma[np.ix_(~icond, ~icond)] - np.dot(A,
                                               Sigma[np.ix_(icond, ~icond)])  # Sigma11 - A * Sigma21
    return A, b, V


def solveV(V, B):
    """
    Computes :math:`X = V^{-1}B` where V is a variance matrix.

    Args:
        V (ndarray(n_dim1, n_dim1)): Variance matrix V in :math:`X = V^{-1}B`.
        B (ndarray(n_dim1, n_dim2)): Matrix B in :math:`X = V^{-1}B`.

    Returns:
        (ndarray(n_dim1, n_dim2)): Matrix X in :math:`X = V^{-1}B`

    """
    L, low = scl.cho_factor(V)
    return scl.cho_solve((L, low), B)

def zero_pad(x, n_deriv, n_deriv_prior):
    """
    Pad x with 0 at the end for each variable.

    Args:
        x0 (ndarray(n_dim1, n_dim2)): Any matrix or vector.
        n_deriv (list): Number of derivatives for each variable in ODE IVP.
        n_deriv_prior (list): Number of derivatives for each variable in Kalman solver.

    Returns:
        (ndarray(n_dim1, n_dim2)): Padded matrix or vector.

    """
    if len(x.shape)==1:
        X = np.zeros(sum(n_deriv_prior), order='F')
    else:
        X = np.zeros((len(n_deriv), sum(n_deriv_prior)), order='F')
    
    n_deriv = [x+1 for x in n_deriv]
    for i in range(len(n_deriv)):
        indx = sum(n_deriv[:i])
        indX = sum(n_deriv_prior[:i])
        if len(x.shape)==1:
            X[indX:indX+n_deriv[i]] = x[indx:indx+n_deriv[i]]
        else:
            X[i, indX:indX+n_deriv[i]] = x[i, indx:indx+n_deriv[i]]
    return X

def indep_init(init, n_deriv_prior):
    """
    Computes the necessary parameters for the Kalman filter and smoother.

    Args:
        init (list(n_var)): Computed initial parameters for each variable.
        n_deriv_prior (int): Number of derivatives for each variable in Kalman solver.
    
    Returns:
        (tuple):
        - **kinit** (dict): Dictionary holding the computed initial parameters for the
          Kalman solver.
        - **wgt_meas** (ndarray(n_var, p)): Transition matrix defining the measure prior.
        - **x0_state** (ndarray(p)): Initial state of the ODE function.

    """
    mu_state = init['mu_state']
    wgt_state_i = init['wgt_state']
    var_state_i = init['var_state']

    n_var = len(var_state_i)
    p = sum(n_deriv_prior)
    wgt_state = np.zeros((p, p), order='F')
    var_state = np.zeros((p, p), order='F')
    ind = 0
    for i in range(n_var):
        wgt_state[ind:ind+n_deriv_prior[i], ind:ind+n_deriv_prior[i]] = wgt_state_i[i]
        var_state[ind:ind+n_deriv_prior[i], ind:ind+n_deriv_prior[i]] = var_state_i[i]
        ind += n_deriv_prior[i]
    kinit = {"wgt_state":wgt_state, "mu_state":mu_state,
            "var_state":var_state}
    
    return kinit

def norm_sim(z, mu, V):
    """
    Simulates from :math:`x \sim N(\mu, V)`.

    Args:
        z (ndarray(n_dim)): Random vector drawn from :math:`N(0,1)`.
        mu (ndarray(n_dim)): Vector mu in :math:`x \sim N(\mu, V)`.
        V (ndarray(n_dim, n_dim)): Matrix V in :math:`x \sim N(\mu, V)`.
    
    Returns:
        (ndarray(n_dim)): Vector x in :math:`x \sim N(\mu, V)`.
    
    """
    L = scl.cholesky(V, True)
    #L = np.linalg.cholesky(V)
    return np.dot(L, z) + mu

def rand_mat(n, p, pd=True):
    """
    Simulate a nxp random matrix from N(0, 1).

    Args:
        n (int): Size of first dimension.
        p (int): Size of second dimension.
        pd (bool): Flag for returning positve definite matrix.
    
    Returns:
        (ndarray(p, n)): Random N(0, 1) matrix.
    
    """
    V = np.zeros((p, n), order='F')
    V[:] = np.random.randn(p, n)
    if (p==n) and pd:
        V[:] = np.matmul(V, V.T)
    return V
