"""
.. module:: kalman_smooth

Backward pass algorithm to smooth a Kalman Filter estimate.
"""
import numpy as np

def kalman_smooth(A, mu_currs, Sigma_currs, mu_preds, Sigma_preds):
    """Backward pass to smooth a Kalman Filter estimate.
    Perform smoothing to estimate the state at time :math:`t`
    give an observation at time :math:`[0...N]` for :math: `t=0:N`.  This
    method is useful if one wants to track an object with streaming
    observations.

    Parameters
    ----------
    A : [n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        state transition matrix from time t to t+1.
    mu_currs : [n_timesteps, n_dim_state] :obj:`numpy.ndarray`
        mean estimate for state at time t given observations from times
        [0...t].
    Sigma_currs : [n_timesteps, n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        covariance of estimate for state at time t given observations from
        times [0...t]
    mu_preds : [n_timesteps, n_dim_state] :obj:`numpy.ndarray`
        mean of hidden state at time t+1 given observations from 
        times [0...t+1]
    Sigma_preds : [n_timesteps, n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        covariance of hidden state at time t+1 given observations from 
        times [0...t+1]

    Returns
    -------
    mu_smooth : [n_timesteps, n_dim_state] :obj:`numpy.ndarray`
        mean of hidden state at time t given observations from 
        times [0...N] for t=0:N.
    Sigma_pred : [n_timesteps, n_dim_state, n_dim_state] :obj:`numpy.ndarray`
        covariance of hidden state at time t given observations from 
        times [0...N] for t=0:N.
    """
    n_timesteps, n_dim_state = mu_currs.shape
    mu_smooth = np.zeros((n_timesteps, n_dim_state))
    Sigma_smooth = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    
    #Base Case
    mu_smooth[-1] = mu_currs[-1] #\mu_{t|N} = \mu_{N|N} for t=N
    Sigma_smooth[-1] = Sigma_currs[-1] #\Sigma_{t|N} = \Sigma{N|N} for t=N
    
    for t in reversed(range(n_timesteps-1)):
        #B = \Sigma{t|t}A'_{t+1}[\Sigma_{t+1|t}]^{-1}
        B = np.linalg.multi_dot([Sigma_currs[t], A.T, np.linalg.pinv(Sigma_preds[t+1])])
        #\mu_{t|N} = \mu_{t|t} + B (\mu_{t+1|N} - \mu_{t+1|t})
        mu_smooth[t] = mu_currs[t] + B.dot(mu_smooth[t+1] - mu_preds[t+1])
        #\Sigma_{t+1|N} = \Sigma_{t|t} + B (\Sigma_{t+1|N} - \Sigma_{t+1|t}) B'
        Sigma_smooth[t] = Sigma_currs[t] + np.linalg.multi_dot([B, (Sigma_smooth[t+1] - Sigma_preds[t+1]), B.T])
        
    return (mu_smooth, Sigma_smooth)
