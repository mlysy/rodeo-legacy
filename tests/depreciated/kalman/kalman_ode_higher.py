"""
.. module:: kalman_ode_higher

Probabilistic ODE solver based on the Kalman filter and smoother.

"""
import numpy as np

from .KalmanTV import KalmanTV
from probDE.utils import norm_sim
import scipy as sp

def _interrogate_chkrebtii(x_meas, var_meas,
                           fun, t, theta,
                           wgt_meas, mu_state_pred, var_state_pred, z_state):
    """
    Interrogate method of Chkrebtii et al (2016).

    Args:
        x_meas (ndarray(n_meas)): Interrogation variable.
        var_meas (ndarray(n_meas, n_meas)): Interrogation variance.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.

    Returns:
        (tuple):
        - **x_meas** (ndarray(n_meas)): Interrogation variable.
        - **var_meas** (ndarray(n_meas, n_meas)): Interrogation variance.
    """
    var_meas[:] = np.linalg.multi_dot([wgt_meas, var_state_pred, wgt_meas.T])
    x_state = norm_sim(z=z_state,
                       mu=mu_state_pred,
                       V=var_state_pred)
    fun(x_state, t, theta, x_meas)
    return

def _interrogate_probde(x_meas, var_meas,
                        fun, t, theta,
                        wgt_meas, mu_state_pred, var_state_pred):
    """
    Interrogate method of probde.

    Args:
        x_meas (ndarray(n_meas)): Interrogation variable.
        var_meas (ndarray(n_meas, n_meas)): Interrogation variance.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        
    Returns:
        (tuple):
        - **x_meas** (ndarray(n_meas)): Interrogation variable.
        - **var_meas** (ndarray(n_meas, n_meas)): Interrogation variance.
    
    """
    var_meas[:] = np.linalg.multi_dot([wgt_meas, var_state_pred, wgt_meas.T])
    x_state = mu_state_pred
    fun(x_state, t, theta, x_meas)
    return 

def kalman_ode_higher(fun, x0_state, tmin, tmax, n_eval, wgt_state, mu_state, var_state, wgt_meas, z_state, theta=None, smooth_mv=True, smooth_sim=False):
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
        x0_state (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated,
            such that discretization timestep is :math:`dt = b/N`.
        wgt_state (ndarray(n_dim_state, n_dim_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_dim_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_dim_state, n_dim_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_dim_state)): Transition matrix defining the measure prior; :math:`W`.

    Returns:
        (tuple):
        - **x_state_smooths** (ndarray(n_timesteps, n_dim_state)): Sample solution at time t given observations from times [0...N] for
          :math:`t = 0,1/N,\ldots,1`.
        - **mu_state_smooths** (ndarray(n_timesteps, n_dim_state)): Posterior mean of the solution process :math:`X_t` at times
          :math:`t = 0,1/N,\ldots,1`.
        - **var_state_smooths** (ndarray(n_timesteps, n_dim_state, n_dim_state)): Posterior variance of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.

    """
    # Dimensions of state and measure variables
    n_dim_meas = wgt_meas.shape[0]
    n_dim_state = len(mu_state)
    n_timesteps = n_eval + 1

    # argumgents for kalman_filter and kalman_smooth
    mu_meas = np.zeros(n_dim_meas)
    var_meas = np.zeros((n_dim_meas, n_dim_meas))
    x_meas = np.zeros(n_dim_meas)
    mu_state_filts = np.zeros((n_timesteps, n_dim_state))
    var_state_filts = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    mu_state_preds = np.zeros((n_timesteps, n_dim_state))
    var_state_preds = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    mu_state_smooths = np.zeros((n_timesteps, n_dim_state))
    var_state_smooths = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    x_state_smooths = np.zeros((n_timesteps, n_dim_state))

    # initialize things
    mu_state_filts[0] = x0_state
    #x_meass[0] = x0_state.dot(wgt_meas.T)
    mu_state_preds[0] = mu_state_filts[0]
    var_state_preds[0] = var_state_filts[0]
    mu_state_smooths[0] = mu_state_filts[0]
    x_state_smooths[0] = x0_state

    # forward pass
    KFS = KalmanTV(n_dim_meas, n_dim_state)
    for t in range(n_eval):
        mu_state_preds[t+1], var_state_preds[t+1] = (
            KFS.predict(mu_state_past=mu_state_filts[t],
                        var_state_past=var_state_filts[t],
                        mu_state=mu_state,
                        wgt_state=wgt_state,
                        var_state=var_state)
        )

        #var_meas = np.linalg.multi_dot(
        #    [wgt_meas, var_state_preds[t+1], wgt_meas.T])
        #x_state_tt = norm_sim(z=z_state_sim[:, t],
        #                      mu=mu_state_preds[t+1],
        #                      V=var_state_preds[t+1])
        #fun(x_state_tt, tmin + (tmax-tmin)*(t+1)/n_eval, theta, x_meas)
        _interrogate_chkrebtii(x_meas=x_meas,
                               var_meas=var_meas,
                               fun=fun,
                               t=tmin + (tmax-tmin)*(t+1)/n_eval,
                               theta=theta,
                               wgt_meas=wgt_meas,
                               mu_state_pred=mu_state_preds[t+1],
                               var_state_pred=var_state_preds[t+1],
                               z_state=z_state[:, t])
    
        mu_state_filts[t+1], var_state_filts[t+1] = (
            KFS.update(mu_state_pred=mu_state_preds[t+1],
                       var_state_pred=var_state_preds[t+1],
                       x_meas=x_meas,
                       mu_meas=mu_meas,
                       wgt_meas=wgt_meas,
                       var_meas=var_meas)
        )

    # backward pass
    mu_state_smooths[-1] = mu_state_filts[-1]
    var_state_smooths[-1] = var_state_filts[-1]
    #x_states[-1] = np.random.multivariate_normal(
    #    mu_state_smooths[-1], var_state_smooths[-1], tol=1e-6)
    x_state_smooths[-1] = norm_sim(z=z_state[:, n_eval],
                            mu=mu_state_smooths[-1],
                            V=var_state_smooths[-1])
    for t in reversed(range(1, n_eval)):
        if smooth_mv and smooth_sim:
            mu_state_smooths[t], var_state_smooths[t], x_state_smooths[t] = (
                KFS.smooth(x_state_next=x_state_smooths[t+1],
                           mu_state_next=mu_state_smooths[t+1],
                           var_state_next=var_state_smooths[t+1],
                           mu_state_filt=mu_state_filts[t],
                           var_state_filt=var_state_filts[t],
                           mu_state_pred=mu_state_preds[t+1],
                           var_state_pred=var_state_preds[t+1],
                           wgt_state=wgt_state,
                           z_state=z_state[:, (n_eval+1)+t])
            )
        elif smooth_mv:
            mu_state_smooths[t], var_state_smooths[t] = (
                KFS.smooth_mv(mu_state_next=mu_state_smooths[t+1],
                              var_state_next=var_state_smooths[t+1],
                              mu_state_filt=mu_state_filts[t],
                              var_state_filt=var_state_filts[t],
                              mu_state_pred=mu_state_preds[t+1],
                              var_state_pred=var_state_preds[t+1],
                              wgt_state=wgt_state)
            )
        elif smooth_sim:
            x_state_smooths[t] = (
                KFS.smooth_sim(x_state_next=x_state_smooths[t+1],
                               mu_state_filt=mu_state_filts[t],
                               var_state_filt=var_state_filts[t],
                               mu_state_pred=mu_state_preds[t+1],
                               var_state_pred=var_state_preds[t+1],
                               wgt_state=wgt_state,
                               z_state=z_state[:, (n_eval+1)+t])
            )

    if smooth_sim and smooth_mv:
        return x_state_smooths, mu_state_smooths, var_state_smooths
    elif smooth_mv:
        return mu_state_smooths, var_state_smooths
    elif smooth_sim:
        return x_state_smooths