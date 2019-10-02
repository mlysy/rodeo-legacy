"""
.. module:: kalman_ode

Probabilistic solver for 1st order ODE

.. math:: v_t = dx_t/dt = f(x_t, t)

on the time interval :math:`t \in [0, 1]` with initial condition :math:`x_0 = x0`.
"""

import numpy as np
from BayesODE.Kalman.Old.kalman_filter import kalman_filter
from BayesODE.Kalman.Old.kalman_smooth import kalman_smooth
from math import sqrt

def kalman_ode(fun, x0, N, A, V, v_star = None):
    """Probabilistic ODE solver based on the Kalman filter and smoother.

    Returns an approximate solution to the ODE

    .. math:: dx_t/dt = f(x_t, t)

    on the interval :math:`t \in [0, 1]`, with initial condition :math:`x_0 = x0`.
    
    Parameters
    ----------

    fun : function 
        ODE function :math:`f(x, t)` taking arguments `x` and `t`.
    x0 : float
        Initial value of :math:`x_t` at time :math:`t = 0`.
    N : int
        Number of discretization points of the time interval,
        such that discretization timestep is `dt = 1/N`.
    A : [2, 2] 
        Transition matrix defining the solution prior (see below).
    V : [2, 2] :obj:`numpy.ndarray`
        Variance matrix defining the solution prior.  Namely,
        if :math:`y_n = (x_n, v_n)` is the solution and its derivative
        at time `t = n/N`, then

        .. math:: y_{n+1} = A y_n + V^{1/2} \epsilon_n, \qquad \epsilon_n \stackrel{iid}{\sim} \mathcal N(0, I_2).
    v_star : [N+1] :obj:`numpy.ndarray`, optional
        Pre-generated model interrogations.  Mainly useful for debugging.

    Returns
    -------
    Yn : [n_timesteps, n_dim_state] :obj:`numpy.ndarray`
        Sample solution at time t given observations from times [0...N] for :math:`t = 0,1/N,\ldots,1`.
    yn_mean : [N+1, 2] :obj:`numpy.ndarray`
        Posterior mean of the solution process and its derivative :math:`y_n = (x_n, v_n)` at times :math:`t = 0,1/N,\ldots,1`.
    yn_var : [N+1, 2, 2] :obj:`numpy.ndarray`
        Posterior variance of the solution process and its derivative at times :math:`t = 0,1/N,\ldots,1`.
    """
    # notation consistent with pykalman package
    n_dim_obs = 1
    n_dim_state = 2
    n_timesteps = N+1
    # allocate memory
    has_vs = v_star is not None
    if has_vs is False:
        # model interrogations vs = v_star
        vs = np.zeros((n_timesteps,n_dim_obs)) 
    else:
        vs = np.array(v_star)
    # var(vs_n | y_n), to be determined during the interrogation process
    sig2 = np.zeros((n_timesteps, n_dim_obs, n_dim_obs))
    # solution process
    # can't actually draw from this (correctly) for now, because of how ks.smooth is implemented...
    yn = np.zeros((n_timesteps, n_dim_state)) 
    # forward mean and variance.
    mu = np.zeros((n_timesteps, n_dim_state)) # E[y_n | vs_0:n]
    # var(y_n | vs_0:n)
    Sigma = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    
    # arguments to use low-level pykalman functions
    """
    observations = vs
    observation_matrix = np.array([[0., 1.]])
    observation_offset = np.array([0.])
    observation_covariances = sig2 # multidimensional
    transition_matrix = np.array(A)
    transition_offset = np.array([0., 0.])
    transition_covariance = np.array(V) # single dimensional
    filtered_state_means = mu
    filtered_state_covariances = Sigma
    predicted_state_means = np.zeros((n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    """

    # argumgents for kalman_filter and kalman_smooth
    b = np.array([0., 0.])
    D = np.array([[0., 1.]])
    e = np.array([0.])
    F = sig2
    mu_currs = mu
    Sigma_currs = Sigma
    mu_preds = np.zeros((n_timesteps, n_dim_state))
    Sigma_preds = np.zeros((n_timesteps, n_dim_state, n_dim_state))

    # initialize things
    y0 = np.array([x0, fun(x0, 0.)]) # initial state
    mu[0] = y0
    vs[0] = y0[1]
    mu_preds[0] = mu[0]
    Sigma_preds[0] = Sigma[0]
    
    # forward pass: merging pks._filter to accommodate multiple
    # observation_covariances
    for t in range(N):
        # calculate mu_tt = E[y_t | vs_0:t-1] and
        # Sigma_tt = var(y_t | vs_0:t-1)
        mu_tt = np.dot(A, mu[t]) # np.array((A*np.matrix(mu[n]).T))
        Sigma_tt = np.linalg.multi_dot([A, Sigma[t], A.T]) + V #A*Sigma[n]*A.T + V
        sig2[t+1] = Sigma_tt[1,1] # new observation_covariance

        # Model Interrogation Step
        if has_vs is False:
            xs = np.random.normal(mu_tt[0], sqrt(Sigma_tt[0,0]))
            vs[t+1] = fun(xs, (t+1)/N)

        # kalman filter update
        (mu_preds[t+1], Sigma_preds[t+1], mu_currs[t+1], Sigma_currs[t+1]) = (
            kalman_filter(mu_curr = mu_currs[t],
                        Sigma_curr = Sigma_currs[t],
                        u_star = vs[t+1],
                        A = A,
                        b = b,
                        V = V,
                        D = D,
                        e = e,
                        F = F[t+1]
                        )
        )
    # backward pass
    (Yn, mu_smooth, Sigma_smooth) = (
        kalman_smooth(
            A = A, 
            mu_currs = mu_currs,
            Sigma_currs = Sigma_currs, 
            mu_preds = mu_preds,
            Sigma_preds = Sigma_preds
        )
    )
    yn_mean = mu_smooth
    yn_var = Sigma_smooth
    return (Yn, yn_mean, yn_var)
