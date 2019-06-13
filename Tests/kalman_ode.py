"""
.. module:: kalman_ode

Probabilistic solver for 1st order ODE
..math::

v_t = dx_t/dt = f(x_t, t)

on the time interval :math:`t \in [0, 1]` with initial condition :math:`x_0 = x0`.

Should eventually be added to module `BayesODE`.
"""

import numpy as np
from pykalman import standard as pks
# from numba import jit
from math import sqrt
import os
os.chdir('../')
from BayesODE.utils import mvCond
from BayesODE.cov_fun import cov_vv_ex, cov_xv_ex, cov_xx_ex
from Tests.test_exp_integrate import cov_yy_ex
#import pdb

def filter_update_full(filtered_state_mean, filtered_state_covariance,
                       observation, transition_matrix,
                       transition_offset, transition_covariance,
                       observation_matrix, observation_offset,
                       observation_covariance):
    """Update a Kalman Filter state estimate.
    Perform a one-step update to estimate the state at time :math:`t+1`
    give an observation at time :math:`t+1` and the previous estimate for
    time :math:`t` given observations from times :math:`[0...t]`.  This
    method is useful if one wants to track an object with streaming
    observations.

    Exactly the same as `pykalman.standard.KalmanFilter.filter_update`,
    except outputs everything needed to use `pykalman.standard._smooth`.

    Parameters
    ----------
    filtered_state_mean : [n_dim_state] array
        mean estimate for state at time t given observations from times
        [0...t].
    filtered_state_covariance : [n_dim_state, n_dim_state] array
        covariance of estimate for state at time t given observations from
        times [0...t]
    observation : [n_dim_obs] array
        observation from time t+1.
    transition_matrix : [n_dim_state, n_dim_state] array
        state transition matrix from time t to t+1.
    transition_offset : [n_dim_state] array
        state offset for transition from time t to t+1.
    transition_covariance : [n_dim_state, n_dim_state] array
        state transition covariance from time t to t+1.
    observation_matrix : [n_dim_obs, n_dim_state] array
        observation matrix at time t+1.
    observation_offset : [n_dim_obs] array
        observation offset at time t+1.
    observation_covariance : [n_dim_obs, n_dim_obs] array
        observation covariance at time t+1.

    Returns
    -------
    predicted_state_mean : [n_dim_state] array
        mean of hidden state at time t+1 given observations from 
        times [0...t+1]
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance of hidden state at time t+1 given observations from 
        times [0...t+1]
    kalman_gain : [n_dim_state, n_dim_obs] array
        Kalman gain matrix for time t+1
    filtered_state_mean : [n_dim_state] array
        mean of hidden state at time t+1 given observations from 
        times [0...t+1]
    filtered_state_covariance : [n_dim_state, n_dim_state] array
        covariance of hidden state at time t+1 given observations from 
        times [0...t+1]
    """
    predicted_state_mean, predicted_state_covariance = (
        pks._filter_predict(
            transition_matrix, transition_covariance,
            transition_offset, filtered_state_mean,
            filtered_state_covariance
        )
    )
    (kalman_gain, filtered_state_mean,
     filtered_state_covariance) = (
         pks._filter_correct(
             observation_matrix, observation_covariance,
             observation_offset, predicted_state_mean,
             predicted_state_covariance, observation
         )
     )
    return (predicted_state_mean, predicted_state_covariance,
            kalman_gain, filtered_state_mean,
            filtered_state_covariance)

def kalman_ode(fun, x0, N, A, V, v_star = None):
    """
    Probabilistic ODE solver based on the Kalman filter and smoother.

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
    A : [2, 2] :obj:`numpy.ndarray`
        Transition matrix defining the solution prior (see below).
    V : [2, 2] :obj:`numpy.ndarray`
        Variance matrix defining the solution prior.  Namely,
        if :math:`y_n = (x_n, v_n)` is the solution and its derivative
        at time `t = n/N`, then
        .. math::

        y_{n+1} = A y_n + V^{1/2} \epsilon_n, \qquad \epsilon_n \stackrel{iid}{\sim} \mathcal N(0, I_2).
        
    v_star : [N+1] :obj:`numpy.ndarray`, optional
        Pre-generated model interrogations.  Mainly useful for debugging.

    Returns
    -------

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
    # initialize things
    y0 = np.array([x0, f(x0, 0.)]) # initial state
    mu[0] = y0
    vs[0] = y0[1]
    predicted_state_means[0] = mu[0]
    predicted_state_covariances[0] = Sigma[0]
    
    # forward pass: merging pks._filter to accommodate multiple
    # observation_covariances
    for t in range(N):
        # calculate mu_tt = E[y_t | vs_0:t-1] and
        # Sigma_tt = var(y_t | vs_0:t-1)
        mu_tt = np.dot(A, mu[t]) # np.array((A*np.matrix(mu[n]).T))
        Sigma_tt = np.linalg.multi_dot([A, Sigma[t], A.T]) + V #A*Sigma[n]*A.T + V
        sig2[t+1] = Sigma_tt[1,1] # new observation_covariance (For some reason 0,0 works a lot better than 1,1)

        # Model Interrogation Step
        if has_vs is False:
            xs = np.random.normal(mu_tt[0], sqrt(Sigma_tt[0,0]))
            vs[t+1] = fun(xs, (t+1)/N)

        # kalman filter update
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
    yn_mean = smoothed_state_means
    yn_var = smoothed_state_covariances
    return (yn_mean, yn_var)

# ok let's check it out

def f(x,t):
    return  3*(t+1/4) - x/(t+1/4)

'''
x0 = 0
gamma = 0.35
alpha = 236
N = 5
tseq = np.linspace(0, 1, N+1)
Sigma = cov_yy_ex(tseq[1:3], tseq[1:3], gamma, alpha)
icond = np.array([True]*2 + [False]*2)
mu = np.array([x0, 0, x0, 0])
A, b, V = mvCond(mu, Sigma, icond)

kalman_ode(f, x0, N, A, V)
'''