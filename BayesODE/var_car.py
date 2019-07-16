"""
.. module:: var_car

Variance function for the CAR(p) process:

.. math:: var(X_T, X_T)

"""

import numpy as np

def var_car(tseq, roots, sigma=1.):
    """Computes the variance function for the CAR(p) process :math: `var(X_T, X_T)`
    
    Parameters
    ----------
    
    tseq: [N] :obj:`numpy.ndarray` of float
        Time vector tseq
    roots: [p] :obj:`numpy.ndarray` of float
        Root vector roots; roots must be negative
    sigma: float
        Parameter in \Sigma

    Returns
    -------
    
    float
        Evaluates :math:`var(X_T, X_T)`.
    """
    delta = np.array(-roots)
    D = np.diag(delta)
    p = len(roots)
    Q = np.zeros((p, p))

    row = np.ones(p)
    for i in range(p):
        Q[i] = row
        row = row*roots

    # Sigma = np.zeros((p, p))
    # Sigma[p-1, p-1] = sigma**2
    Sigma = np.zeros(p)
    Sigma[p-1] = sigma * sigma

    Q_inv = np.linalg.pinv(Q)
    # Gamma = np.linalg.multi_dot([Q, D, Q_inv])  # Q*D*Q^-1
    # Sigma_tilde = np.linalg.multi_dot(
    #     [Q_inv, Sigma, Q_inv.T])  # Q^-1*Sigma*Q^-1'
    Sigma_tilde = np.matmul(Q_inv * Sigma, Q_inv.T)  # Q^-1*Sigma*Q^-1'

    V = np.zeros((len(tseq), p, p))
    for t in range(len(tseq)):
        V_tilde = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                V_tilde[i, j] = Sigma_tilde[i, j] / (delta[i] + delta[j]) * (
                    1.0 - np.exp(- (delta[i] + delta[j]) * tseq[t]))  # V_tilde
                V_tilde[j, i] = V_tilde[i, j]

        V[t] = np.linalg.multi_dot([Q, V_tilde, Q.T])  # V_deltat

    return V
