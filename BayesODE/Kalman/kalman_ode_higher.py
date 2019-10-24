"""
.. module:: kalman_ode_higher

Model is

.. math:: 

   x_n = c_n + T_n x_n-1 + R_n^{1/2} \epsilon_n

   y_n = d_n + W_n x_n + H_n^{1/2} \eta_n

"""

import numpy as np
from BayesODE.Kalman.KalmanTV import KalmanTV

def kalman_ode_higher(fun, x_0, N, wgtState, muState, varState, wgtMeas, T=1):
    """Probabilistic ODE solver based on the Kalman filter and smoother.

    Returns an approximate solution to the higher order ODE

    .. math:: W x_t = F(x_t, t)
    
    on the time interval :math:`t \in [0, T]` with initial condition :math:`x_0 = x_0`.

    Parameters
    ----------

    fun : function 
        Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
    x_0 : float
        Initial value of :math:`x_t` at time :math:`t = 0`.
    N : int
        Number of discretization points of the time interval,
        such that discretization timestep is :math:`dt = 1/N`.
    wgtState : ndarray(n_dim_state, n_dim_state)
        Transition matrix defining the solution prior; :math:`T_n`.
    muState : ndarray(n_dim_state)
        Transition_offsets defining the solution prior; :math:`c_n`.
    varState : ndarray(n_dim_state, n_dim_state)
        Variance matrix defining the solution prior; :math:`R_n`.
    wgtMeas : ndarray(n_dim_state)
        Transition matrix defining the measure prior; :math:`W_n`.
    T : int
        Last time point required for the solution
    
    Returns
    -------
    xStates : ndarray(n_timesteps, n_dim_state)
        Sample solution at time t given observations from times [0...N] for :math:`t = 0,1/N,\ldots,1`.
    muState_smooths : ndarray(n_timesteps, n_dim_state)
        Posterior mean of the solution process :math: `y_n` at times :math:`t = 0,1/N,\ldots,1`.
    varState_smooths : ndarray(n_timesteps, n_dim_state, n_dim_state)
        Posterior variance of the solution process at times :math:`t = 0,1/N,\ldots,1`.
    """
    
    # Dimensions of state and measure variables
    n_dim_meas = wgtMeas.shape[0]
    n_dim_state = len(muState)
    n_timesteps = N+1

    # allocate memory for observations
    xMeass = np.zeros((n_timesteps,n_dim_meas))

    # argumgents for kalman_filter and kalman_smooth
    muMeas = np.zeros(n_dim_meas)
    varMeass = np.zeros((n_timesteps, n_dim_meas, n_dim_meas))
    muState_filts = np.zeros((n_timesteps, n_dim_state))
    varState_filts = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    muState_preds = np.zeros((n_timesteps, n_dim_state))
    varState_preds = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    muState_smooths = np.zeros((n_timesteps, n_dim_state))
    varState_smooths = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    xStates = np.zeros((n_timesteps, n_dim_state))

    # initialize things
    muState_filts[0] = x_0
    xMeass[0] = x_0.dot(wgtMeas.T)
    muState_preds[0] = muState_filts[0]
    varState_preds[0] = varState_filts[0]

    # forward pass
    # calculate mu_tt = E[X_t | y_0:t-1] and
    # Sigma_tt = var(X_t | y_0:t-1)

    KFS = KalmanTV(n_dim_meas, n_dim_state)
    for t in range(N):
        mu_tt = np.dot(wgtState, muState_filts[t]) + muState
        Sigma_tt = np.linalg.multi_dot([wgtState, varState_filts[t], wgtState.T]) + varState #A*Sigma[t]*A.T + V 
        varMeass[t+1] = np.linalg.multi_dot([wgtMeas, Sigma_tt, wgtMeas.T]) # new observation_covariance
        I_tt = I_tt = np.random.normal(loc=0.0, scale=1.0, size=n_dim_state)
        D_tt = np.linalg.cholesky(Sigma_tt)
        xState_t1 = mu_tt + D_tt.dot(I_tt) #X_{t+1} ~ N(mu_{t+1|t}, Sigma_{t+1|t})
        xMeass[t+1] = fun(xState_t1, T*(t+1)/N) #new observation (y_{t+1})

        (muState_preds[t+1], varState_preds[t+1], muState_filts[t+1], varState_filts[t+1]) = (
            KFS.filter(muState_past = muState_filts[t],
                    varState_past = varState_filts[t],
                    muState = muState,
                    wgtState = wgtState,
                    varState = varState,
                    xMeas = xMeass[t+1],
                    muMeas = muMeas,
                    wgtMeas = wgtMeas,
                    varMeas = varMeass[t+1])
            
        )

    # backward pass
    muState_smooths[-1] = muState_filts[-1]
    varState_smooths[-1] = varState_filts[-1]
    xStates[-1] = np.random.multivariate_normal(muState_smooths[-1], varState_smooths[-1])
    for t in reversed(range(N)):
        (muState_smooths[t], varState_smooths[t], xStates[t]) = (
            KFS.smooth(xState_next = xStates[t+1],
                    muState_next = muState_smooths[t+1],
                    varState_next = varState_smooths[t+1],
                    muState_filt = muState_filts[t],
                    varState_filt = varState_filts[t],
                    muState_pred = muState_preds[t+1],
                    varState_pred = varState_preds[t+1],
                    wgtState = wgtState)
        )
    
    return xStates, muState_smooths, varState_smooths
