import numpy as np
import jax.numpy as jnp
from jax import jit, partial, lax
from jax.config import config
from math import sin
from kalmantv_jax import *
from kalmantv_jax import _state_sim
config.update("jax_enable_x64", True)

@jit
def _interrogate_rodeo(wgt_meas, mu_state_pred, var_state_pred):
    """
    Interrogate method of rodeo.

    Args:
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        
    Returns:
        (tuple):
        - **x_state** (ndarray(n_state)): Temporary state variable.
        - **var_meas** (ndarray(n_meas, n_meas)): Interrogation variance.
    
    """
    var_meas = jnp.linalg.multi_dot([wgt_meas, var_state_pred, wgt_meas.T])
    x_state = mu_state_pred
    return x_state, var_meas

### kalman_ode does not take function as arguments for now.
### Jax (XLA) cannot set uninitialized arrays so jnp.empty defaults to jnp.zeros.
@partial(jit, static_argnums=(0,2,3,4))
def _solve_filter(fun, x0, tmin, tmax, n_eval, wgt_state, mu_state, 
                  var_state, wgt_meas, z_state, theta=None):
    
    # Dimensions of state and measure variables
    n_meas = wgt_meas.shape[0]
    n_state = len(mu_state)
    n_steps = n_eval + 1

    # arguments for kalman_filter and kalman_smooth
    mu_meas = jnp.zeros(n_meas)
    init_mu_state_filt = x0
    init_var_state_filt = jnp.zeros((n_state, n_state))

    def forward_pass(state_filts, t):
        mu_state_filt, var_state_filt = state_filts
        mu_state_pred, var_state_pred = (
            predict(mu_state_past=mu_state_filt,
                    var_state_past=var_state_filt,
                    mu_state=mu_state,
                    wgt_state=wgt_state,
                    var_state=var_state)
        )

        # model interrogation
        x_state, var_meas = \
            _interrogate_rodeo(wgt_meas=wgt_meas,
                               mu_state_pred=mu_state_pred,
                               var_state_pred=var_state_pred)
        x_meas = fun(x_state, tmin + (tmax-tmin)*(t+1)/n_eval, theta)

        mu_state_filt_next, var_state_filt_next = (
            update(mu_state_pred=mu_state_pred,
                   var_state_pred=var_state_pred,
                   x_meas=x_meas,
                   mu_meas=mu_meas,
                   wgt_meas=wgt_meas,
                   var_meas=var_meas)
        )
        
        new_state_filts = mu_state_filt_next, var_state_filt_next
        filts_and_preds = new_state_filts + (mu_state_pred, var_state_pred)
        return new_state_filts, filts_and_preds

    init_state_filts = (init_mu_state_filt, init_var_state_filt)
    _, scan_out = lax.scan(forward_pass, init_state_filts, jnp.arange(n_eval))

    # Append initial value at time step 0 then move time axis to the end
    init_state_preds = init_state_filts
    init_vals = init_state_filts + init_state_preds
    (mu_state_filt, var_state_filt, mu_state_pred, var_state_pred) = (
        jnp.moveaxis(jnp.concatenate([init[None], out]), 0, -1)
        for init, out in zip(init_vals, scan_out)
    )

    return mu_state_pred, var_state_pred, mu_state_filt, var_state_filt


@partial(jit, static_argnums=(0,2,3,4))
def solve_sim(fun, x0, tmin, tmax, n_eval, wgt_state, mu_state, 
              var_state, wgt_meas, z_state, theta=None):
    
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
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.

    Returns:
        (tuple):
        - **x_state_smooth** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [0...N] for
          :math:`t = 0,1/N,\ldots,1`.
        - **mu_state_smooth** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`X_t` at times
          :math:`t = 0,1/N,\ldots,1`.
        - **var_state_smooth** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.

    """
    # forward pass
    mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = \
        _solve_filter(fun, x0, tmin, tmax, n_eval, wgt_state, mu_state, 
                      var_state, wgt_meas, z_state, theta)
    
    # backward pass
    last_mu_state_smooth = mu_state_filt[:, n_eval]
    last_var_state_smooth = var_state_filt[:, :, n_eval]
    last_x_state_smooth = _state_sim(last_mu_state_smooth,
                                     last_var_state_smooth,
                                     z_state[:, n_eval-1])
    
    def backward_pass(x_state_next, smooth_sim_kwargs):
        x_state_prev = smooth_sim(x_state_next=x_state_next,
                                  wgt_state=wgt_state,
                                  **smooth_sim_kwargs)
        return x_state_prev, x_state_prev

    # lax.scan will efficiently iterate over these arrays for each time step.
    # We slice these arrays so they are aligned.
    # More precisely, for time step t, we want filt[t], pred[t+1], z_state[t-1]
    all_smooth_sim_kwargs = {
        'mu_state_filt': mu_state_filt[:, 1:n_eval],
        'var_state_filt': var_state_filt[:, :, 1:n_eval],
        'mu_state_pred': mu_state_pred[:, 2:n_eval+1],
        'var_state_pred': var_state_pred[:, :, 2:n_eval+1],
        'z_state': z_state[:, :n_eval-1]
    }
    # Move time axis to the front:
    all_smooth_sim_kwargs = {
        k: jnp.moveaxis(v, -1, 0) for k, v in all_smooth_sim_kwargs.items()
    }
    
    (_, x_state_smooth) = lax.scan(backward_pass,
                                   last_x_state_smooth,
                                   all_smooth_sim_kwargs,
                                   reverse=True)

    # Append initial and final values
    x_state_smooth = jnp.concatenate(
        [x0[None], x_state_smooth, last_x_state_smooth[None]]
    )

    return x_state_smooth
