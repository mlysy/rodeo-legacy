"""
.. module:: kalman_ode_higher

Probabilistic solver for higher order ODE

.. math:: a'X_t = F(X_t, t)

on the time interval :math:`t \in [0, 1]` with initial condition :math:`Y_0 = Y0` and :math: `Y_t = (X_t, Z_t)`.
"""

import numpy as np
from BayesODE.filter_update_full import filter_update_full
from pykalman import standard as pks

def kalman_ode_higher(fun, Y0, N, A, b, V, a):
    """Probabilistic ODE solver based on the Kalman filter and smoother.

    Returns an approximate solution to the higher order ODE

    .. math:: a' x_t = F(x_t, t)
    
    on the time interval :math:`t \in [0, 1]` with initial condition :math:`Y_0 = Y0` and :math: `Y_t = (X_t, Z_t)`.

    Parameters
    ----------

    fun : function 
        Higher order ODE function :math: a' y_t = F(y_t, t) taking arguments `y` and `t`.
    Y0 : float
        Initial value of :math:`x_t` at time :math:`t = 0`.
    N : int
        Number of discretization points of the time interval,
        such that discretization timestep is `dt = 1/N`.
    A : [p, p] :obj:`numpy.ndarray` of float
        Transition matrix defining the solution prior (see below).
    b : [p] :obj:`numpy.ndarray` of float
        transition_offsets defining the solution prior (see below).
    V : [p, p] :obj:`numpy.ndarray`
        Variance matrix defining the solution prior. 
        
        .. math:: y_{n+1} = A y_n + b + V^{1/2} \epsilon_n, \qquad \epsilon_n \stackrel{iid}{\sim} \mathcal N(0, I_2).
    a : [q+1]
        Observation vector
    
    Returns
    -------

    Yn_mean : [N+1, p] :obj:`numpy.ndarray`
        Posterior mean of the solution process :math:`Y_n = (X_n, Z_n)` at times :math:`t = 0,1/N,\ldots,1`.
    Yn_var : [N+1, p, p] :obj:`numpy.ndarray`
        Posterior variance of the solution process at times :math:`t = 0,1/N,\ldots,1`.
    """
    
    # notation consistent with pykalman package
    n_dim_obs = 1
    n_dim_state = len(Y0)
    n_timesteps = N+1

    # allocate memory
    us = np.zeros((n_timesteps,n_dim_obs)) 

    # var(us_n | y_n), to be determined during the interrogation process
    sig2 = np.zeros((n_timesteps, n_dim_obs, n_dim_obs))
    
    # solution process
    # forward mean and variance.
    mu = np.zeros((n_timesteps, n_dim_state)) # E[y_n | us_0:n]
    # var(y_n | us_0:n)
    Sigma = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    
    #a padde with 0s
    p = len(b)
    q = len(a) - 1
    a0 = np.pad(a, (0, p-q-1), 'constant', constant_values=(0,0))

    # arguments to use low-level pykalman functions
    observations = us
    observation_matrix = np.array([a0])
    observation_offset = np.array([0.])
    observation_covariances = sig2 # multidimensional
    transition_matrix = A
    transition_offset = b
    transition_covariance = V # single dimensional
    filtered_state_means = mu
    filtered_state_covariances = Sigma
    predicted_state_means = np.zeros((n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    # initialize things
    mu[0] = Y0
    us[0] = Y0.dot(a0)
    predicted_state_means[0] = mu[0]
    predicted_state_covariances[0] = Sigma[0]

    # forward pass: merging pks._filter to accommodate multiple
    # observation_covariances
    # calculate mu_tt = E[y_t | us_0:t-1] and
    # Sigma_tt = var(y_t | us_0:t-1)

    for t in range(N):
        mu_tt = np.dot(A, mu[t]) + b
        Sigma_tt = np.linalg.multi_dot([A, Sigma[t], A.T]) + V #A*Sigma[n]*A.T + V 
        sig2[t+1] = np.linalg.multi_dot([a0, Sigma_tt, a0.T]) # new observation_covariance
        Z_tt = np.random.multivariate_normal(np.zeros(p), np.eye(p))
        D_tt = np.linalg.cholesky(np.absolute(Sigma_tt, where=np.eye(p, dtype=bool)))
        Yt1 = mu_tt + D_tt.dot(Z_tt) #Y_{n+1} ~ p(Y_{n+1} | Y_n)
        us[t+1] = fun(Yt1,(t+1)/N) #new observation (u_{n+1})

        (predicted_state_means[t+1], predicted_state_covariances[t+1],
                 _, filtered_state_means[t+1],
                 filtered_state_covariances[t+1]) = (
                     filter_update_full(filtered_state_mean = filtered_state_means[t],
                                        filtered_state_covariance = filtered_state_covariances[t],
                                        observation = observations[t+1],
                                        transition_matrix = transition_matrix,
                                        transition_offset = transition_offset,
                                        transition_covariance = transition_covariance,
                                        observation_matrix = observation_matrix,
                                        observation_offset = observation_offset,
                                        observation_covariance = observation_covariances[t+1])
                 )
    # backward pass
    (smoothed_state_means, smoothed_state_covariances, _) = (
        pks._smooth(
            transition_matrix, filtered_state_means,
            filtered_state_covariances, predicted_state_means,
            predicted_state_covariances
        )
    )
    Yn_mean = smoothed_state_means
    Yn_var = smoothed_state_covariances
    return Yn_mean, Yn_var
