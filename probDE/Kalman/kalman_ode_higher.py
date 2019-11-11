"""
.. module:: kalman_ode_higher

Probabilistic ODE solver based on the Kalman filter and smoother. 

"""
import numpy as np

from probDE.Kalman.KalmanTV import KalmanTV

def kalman_ode_higher(fun, x0State, tmin, tmax, n_eval, wgtState, muState, varState, wgtMeas):
    """
    Probabilistic ODE solver based on the Kalman filter and smoother. Returns an approximate solution to the higher order ODE

    .. math:: w' x_t = F(x_t, t)
    
    on the time interval :math:`t \in [a, b]` with initial condition :math:`x_0 = x_0`. The corresponding variable names are

    The specific model we are using to approximate the solution :math:`x_n` is

    .. math::

        X_n = c + T X_{n-1} + R_n^{1/2} \epsilon_n

        y_n = d + W X_n + H_n^{1/2} \eta_n
    
    where :math:`\epsilon_n` and :math:`\eta_n` are independent :math:`N(0,1)` distributions and
    :math:`X_n = (x_n, y_n)` at time n and :math:`y_n` denotes the observation at time n.

    Args:
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        x0State (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, 
            such that discretization timestep is :math:`dt = b/N`.
        wgtState (ndarray(n_dim_state, n_dim_state)): Transition matrix defining the solution prior; :math:`T`.
        muState (ndarray(n_dim_state)): Transition_offsets defining the solution prior; :math:`c`.
        varState (ndarray(n_dim_state, n_dim_state)): Variance matrix defining the solution prior; :math:`R`.
        wgtMeas (ndarray(n_dim_state)): Transition matrix defining the measure prior; :math:`W`.
        
    Returns:
        (tuple):
        - **xStates** (ndarray(n_timesteps, n_dim_state)): Sample solution at time t given observations from times [0...N] for
          :math:`t = 0,1/N,\ldots,1`.
        - **muState_smooths** (ndarray(n_timesteps, n_dim_state)): Posterior mean of the solution process :math:`y_n` at times 
          :math:`t = 0,1/N,\ldots,1`.
        - **varState_smooths** (ndarray(n_timesteps, n_dim_state, n_dim_state)): Posterior variance of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.

    """
    # Dimensions of state and measure variables
    n_dim_meas = wgtMeas.shape[0]
    n_dim_state = len(muState)
    n_timesteps = n_eval + 1

    # argumgents for kalman_filter and kalman_smooth
    muMeas = np.zeros(n_dim_meas)
    varMeass = np.zeros((n_timesteps, n_dim_meas, n_dim_meas))
    xMeass = np.zeros((n_timesteps,n_dim_meas))
    muState_filts = np.zeros((n_timesteps, n_dim_state))
    varState_filts = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    muState_preds = np.zeros((n_timesteps, n_dim_state))
    varState_preds = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    muState_smooths = np.zeros((n_timesteps, n_dim_state))
    varState_smooths = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    xStates = np.zeros((n_timesteps, n_dim_state))

    # initialize things
    muState_filts[0] = x0State
    xMeass[0] = x0State.dot(wgtMeas.T)
    muState_preds[0] = muState_filts[0]
    varState_preds[0] = varState_filts[0]

    # forward pass
    KFS = KalmanTV(n_dim_meas, n_dim_state)
    for t in range(n_eval):
        muState_preds[t+1], varState_preds[t+1] = (
            KFS.predict(muState_past = muState_filts[t],
                        varState_past = varState_filts[t],
                        muState = muState,
                        wgtState = wgtState,
                        varState = varState)
        )

        varMeass[t+1] = np.linalg.multi_dot([wgtMeas, varState_preds[t+1], wgtMeas.T]) 
        I_tt = np.random.normal(loc=0.0, scale=1.0, size=n_dim_state)
        R_tt = np.linalg.cholesky(varState_preds[t+1])
        xState_tt = muState_preds[t+1] + R_tt.dot(I_tt) 
        xMeass[t+1] = fun(xState_tt, tmin + (tmax-tmin)*(t+1)/n_eval)

        muState_filts[t+1], varState_filts[t+1] = (
            KFS.update(muState_pred = muState_preds[t+1],
                       varState_pred = varState_preds[t+1],
                       xMeas = xMeass[t+1],
                       muMeas = muMeas,
                       wgtMeas = wgtMeas,
                       varMeas = varMeass[t+1])
        )

    # backward pass
    muState_smooths[-1] = muState_filts[-1]
    varState_smooths[-1] = varState_filts[-1]
    xStates[-1] = np.random.multivariate_normal(muState_smooths[-1], varState_smooths[-1], tol=1e-6)
    for t in reversed(range(n_eval)):
        muState_smooths[t], varState_smooths[t], xStates[t] = (
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
