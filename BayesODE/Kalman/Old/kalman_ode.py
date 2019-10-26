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

def kalman_ode(fun, x0State, n_eval, wgtState, varState, obs=None):
    """
    Probabilistic ODE solver based on the Kalman filter and smoother. Returns an approximate solution to the ODE

    .. math:: dx_t/dt = f(x_t, t)

    on the interval :math:`t \in [0, 1]`, with initial condition :math:`x_0 = x0`.
    
    Parameters
    ----------
    fun : function 
        ODE function :math:`f(x, t)` taking arguments :math:`x` and :math:`t`.
    x0State : float
        Initial value of :math:`x_t` at time :math:`t = 0`.
    n_eval : int
        Number of discretization points of the time interval,
        such that discretization timestep is :math:`dt = 1/N`.
    wgtState : ndarray(n_dim_state, n_dim_state) 
        Transition matrix defining the solution prior (see below).
    varState : ndarray(n_dim_state, n_dim_state)
        Variance matrix defining the solution prior.  Namely,
        if :math:`y_n = (x_n, v_n)` is the solution and its derivative
        at time :math:`t = n/N`, then

        .. math:: y_{n+1} = A y_n + V^{1/2} \epsilon_n, \qquad \epsilon_n \stackrel{iid}{\sim} \mathcal N(0, I_2).
    obs : ndarray(n_timesteps), optional
        Pre-generated model interrogations.  Mainly useful for debugging.

    Returns
    -------
    Xn : ndarray(n_timesteps, n_dim_state)
        Sample solution at time t given observations from times [0...N] for :math:`t = 0,1/N,\ldots,1`.
    Xn_mean : ndarray(n_timesteps, n_dim_state)
        Posterior mean of the solution process and its derivative :math:`y_n = (x_n, v_n)` at times :math:`t = 0,1/N,\ldots,1`.
    Xn_var : ndarray(n_timesteps, n_dim_state, n_dim_state)
        Posterior variance of the solution process and its derivative at times :math:`t = 0,1/N,\ldots,1`.

    """
    # notation consistent with pykalman package
    n_dim_obs = 1
    n_dim_state = 2
    n_timesteps = n_eval + 1

    # allocate memory
    has_vs = obs is not None
    if has_vs is False:
        # model interrogations vs = v_star
        vs = np.zeros((n_timesteps,n_dim_obs)) 
    else:
        vs = np.array(obs)

    # var(vs_n | y_n), to be determined during the interrogation process
    sig2 = np.zeros((n_timesteps, n_dim_obs, n_dim_obs))
    # solution process
    # can't actually draw from this (correctly) for now, because of how ks.smooth is implemented...
    yn = np.zeros((n_timesteps, n_dim_state)) 
    # forward mean and variance.
    mu = np.zeros((n_timesteps, n_dim_state)) # E[y_n | vs_0:n]
    # var(y_n | vs_0:n)
    Sigma = np.zeros((n_timesteps, n_dim_state, n_dim_state))

    # argumgents for kalman_filter and kalman_smooth
    muState = np.array([0., 0.])
    wgtMeas = np.array([[0., 1.]])
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
    (Xn, mu_smooth, Sigma_smooth) = (
        kalman_smooth(
            A = A, 
            mu_currs = mu_currs,
            Sigma_currs = Sigma_currs, 
            mu_preds = mu_preds,
            Sigma_preds = Sigma_preds
        )
    )
    Xn_mean = mu_smooth
    Xn_var = Sigma_smooth
    return (Xn, Xn_mean, Xn_var)
