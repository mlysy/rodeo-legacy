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
    Y_tt : [n_timesteps, n_dim_state] :obj:`numpy.ndarray`
        samples at time t given observations from 
        times [0...N] for t=0:N.
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
    h_tt = np.zeros((n_timesteps, n_dim_state))
    W_tt = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    Y_tt = np.zeros((n_timesteps, n_dim_state))
    
    #Base Case Smoothing Sampler
    h_tt[-1] = mu_smooth[-1] = mu_currs[-1] #\mu_{t|N} = \mu_{N|N} for t=N
    W_tt[-1] = Sigma_smooth[-1] = Sigma_currs[-1] #\Sigma_{t|N} = \Sigma{N|N} for t=N
    Y_tt[-1] = np.random.multivariate_normal(mu_smooth[-1], Sigma_smooth[-1])
    
    #Iterative Step
    for t in reversed(range(n_timesteps-1)):
        #B = \Sigma{t|t}A'_{t+1}
        B = Sigma_currs[t].dot(A.T)
        
        #\tilde{B} = \Sigma{t|t}A'_{t+1} \Sigma_{t+1|t}^{-1}
        B_tilde = B.dot(np.linalg.pinv(Sigma_preds[t+1]))
        
        #h_t = \mu_{t|t} + B \Sigma_{t+1|t}^{-1} (Z_{t+1} - \mu{t+1|t})
        h_tt[t] = mu_currs[t] + B_tilde.dot(Y_tt[t+1] - mu_preds[t+1])
        
        #W_t = \Sigma{t|t} - B \Sigma{t+1|t}^{-1} B'
        W_tt[t] = Sigma_currs[t] - B_tilde.dot(B.T)
        
        #Y_t ~ N{h_t, W_t}
        Y_tt[t] = np.random.multivariate_normal(h_tt[t], W_tt[t])
        
        #\mu_{t|N} = \mu_{t|t} + \tilde{B} (\mu_{t+1|N} - \mu_{t+1|t})
        mu_smooth[t] = mu_currs[t] + B_tilde.dot(mu_smooth[t+1] - mu_preds[t+1])
        
        #\Sigma_{t|N} = \Sigma_{t|t} + \tilde{B} (\Sigma_{t+1|N} - \Sigma_{t+1|t}) \tilde{B}'
        Sigma_smooth[t] = Sigma_currs[t] + np.linalg.multi_dot([B_tilde, (Sigma_smooth[t+1] - Sigma_preds[t+1]), B_tilde.T])
        
    return (Y_tt, mu_smooth, Sigma_smooth)
