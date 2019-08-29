"""
.. module:: kalman_filter

Kalman filter algorithm to estimate the state at :math:`t+1`
give an observation at time :math:`t+1` and the previous estimate for
time :math:`t` given observations from times :math:`[0...t]`.
"""
import numpy as np

def kalman_filter(mu_curr, Sigma_curr, u_star, A, b, V, D, e, F):
    """Update a Kalman Filter state estimate.
    Perform a one-step update to estimate the state at time :math:`t+1`
    give an observation at time :math:`t+1` and the previous estimate for
    time :math:`t` given observations from times :math:`[0...t]`.  This
    method is useful if one wants to track an object with streaming
    observations.

    Parameters
    ----------
    mu_curr : [n_dim_state] :obj:`numpy.ndarray`
        mean estimate for state at time t given observations from times
        [0...t].
    Sigma_curr : [n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        covariance of estimate for state at time t given observations from
        times [0...t]
    u_star : [n_dim_obs] array
        observation from time t+1.
    A : [n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        state transition matrix from time t to t+1.
    b : [n_dim_state] :obj:`numpy.ndarray`
        state offset for transition from time t to t+1.
    V : [n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        state transition covariance from time t to t+1 (CC').
    D : [n_dim_obs, n_dim_state] :obj:`numpy.ndarray`
        observation matrix at time t+1.
    e : [n_dim_obs] :obj:`numpy.ndarray`
        observation offset at time t+1.
    F : [n_dim_obs, n_dim_obs] :obj:`numpy.ndarray`
        observation covariance at time t+1.

    Returns
    -------
    mu_pred : [n_dim_state] :obj:`numpy.ndarray`
        mean of hidden state at time t+1 given observations from 
        times [0...t+1]
    Sigma_pred : [n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        covariance of hidden state at time t+1 given observations from 
        times [0...t+1]
    mu_next : [n_dim_state] :obj:`numpy.ndarray`
        mean of hidden state at time t+1 given observations from 
        times [0...t+1]
    Sigma_next : [n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        covariance of hidden state at time t+1 given observations from 
        times [0...t+1]
    """
    
    #Calculate \mu_{t+1|t} and \Sigma{t+1|t}
    mu_pred = A.dot(mu_curr) + b #\mu_{t+1|t} = A\mu_{t|t} + b
    Sigma_pred = np.linalg.multi_dot([A, Sigma_curr, A.T]) + V #\Sigma_{t+1|t} = A\Sigma_{t|t}A' + CC'
    
    #Calculate \mu_{t|t} and \Sigma{t|t}
    mu_X_pred = D.dot(mu_pred) + e
    Sigma_XZ_pred = D.dot(Sigma_pred)
    Sigma_XX_pred = np.linalg.multi_dot([D, Sigma_pred, D.T]) + F
    Sigma_ZX_pred = Sigma_pred.dot(D.T)
    
    #B = \Sigma^{ZX}_{t+1|t} [\Sigma^{XX}_{t+1|t}]^{-1}
    B = Sigma_ZX_pred.dot(np.linalg.pinv(Sigma_XX_pred))
    #\mu_{t+1|t+1} = \mu_{t+1|t} + B(u_star - \mu^{X}_{t+1|t})
    mu_next = mu_pred + B.dot(u_star - mu_X_pred)
    #\Sigma_{t+1|t+1} = \Sigma_{t+1|t} - B \Sigma^{XZ}_{t+1|t}
    Sigma_next = Sigma_pred - B.dot(Sigma_XZ_pred)
    return (mu_pred, Sigma_pred, mu_next, Sigma_next)
    