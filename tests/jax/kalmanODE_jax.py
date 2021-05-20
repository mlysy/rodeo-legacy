import numpy as np
import jax.numpy as jnp
from jax import jit, partial
from jax.ops import index, index_update
from jax.config import config
from math import sin
from kalmantv_jax import *
from kalmantv_jax import _state_sim
config.update("jax_enable_x64", True)

# @partial(jit, static_argnums=(1))
# def fun(x, t, theta=None):
#     if x_out is None:
#         x_out = jnp.zeros(1)
#     x_out = index_update(x_out, index[0], sin(2*t) - x[0])
#     return x_out

@partial(jit, static_argnums=(1))
def fun(X_t, t, theta):
    "Fitz ODE written for jax"
    x_out = jnp.zeros(2)
    a, b, c = theta
    V, R = X_t[0], X_t[3]
    x_out = index_update(x_out, index[0], c*(V - V*V*V/3 + R))   
    x_out = index_update(x_out, index[1], -1/c*(V - a + b*R))
    return x_out

# @jit
# def _fzeros(shape):
#     """
#     Create an empty ndarray with the given shape in fortran order.
    
#     """
#     return jnp.zeros(shape[::-1]).T

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
### jnp.zeros does not have order argument to get fortran contiguous. 
def _solve_filter(fun, x0, tmin, tmax, n_eval, wgt_state, mu_state, 
                  var_state, wgt_meas, z_state, theta=None):
    
    # Dimensions of state and measure variables
    n_meas = wgt_meas.shape[0]
    n_state = len(mu_state)
    n_steps = n_eval + 1

    # argumgents for kalman_filter and kalman_smooth
    mu_meas = jnp.zeros((n_meas,))
    mu_state_filt = jnp.zeros((n_state, n_steps))
    var_state_filt = jnp.zeros((n_state, n_state, n_steps))
    mu_state_pred = jnp.zeros((n_state, n_steps))
    var_state_pred = jnp.zeros((n_state, n_state, n_steps))

    # initialize things
    mu_state_filt = index_update(mu_state_filt, index[:, 0], x0)
    mu_state_pred = index_update(mu_state_pred, index[:, 0], x0)

    # forward pass
    for t in range(n_eval):
        mu_state_pred_temp, var_state_pred_temp = (
            predict(mu_state_past=mu_state_filt[:, t],
                    var_state_past=var_state_filt[:, :, t],
                    mu_state=mu_state,
                    wgt_state=wgt_state,
                    var_state=var_state)
        )
        mu_state_pred = index_update(mu_state_pred, index[:, t+1], mu_state_pred_temp)
        var_state_pred = index_update(var_state_pred, index[:, :, t+1], var_state_pred_temp)
        
        # model interrogation
        x_state, var_meas = \
            _interrogate_rodeo(wgt_meas=wgt_meas,
                               mu_state_pred=mu_state_pred[:, t+1],
                               var_state_pred=var_state_pred[:, :, t+1])
        x_meas = fun(x_state, tmin + (tmax-tmin)*(t+1)/n_eval, theta)
        
        mu_state_filt_temp, var_state_filt_temp = (
            update(mu_state_pred=mu_state_pred[:, t+1],
                   var_state_pred=var_state_pred[:, :, t+1],
                   x_meas=x_meas,
                   mu_meas=mu_meas,
                   wgt_meas=wgt_meas,
                   var_meas=var_meas)
        )
        mu_state_filt = index_update(mu_state_filt, index[:, t+1], mu_state_filt_temp)
        var_state_filt = index_update(var_state_filt, index[:, :, t+1], var_state_filt_temp)
        
    return mu_state_pred, var_state_pred, mu_state_filt, var_state_filt

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
    n_state = len(mu_state)
    n_steps = n_eval + 1
    
    # initialize
    mu_state_smooth = jnp.zeros((n_state, n_steps))
    var_state_smooth = jnp.zeros((n_state, n_state, n_steps))
    x_state_smooth = jnp.zeros((n_state, n_steps))
    
    # forward pass
    mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = \
        _solve_filter(fun, x0, tmin, tmax, n_eval, wgt_state, mu_state, 
                      var_state, wgt_meas, z_state, theta)
    
    # backward pass
    mu_state_smooth = index_update(mu_state_smooth, index[:, 0], mu_state_filt[:, 0])
    x_state_smooth = index_update(x_state_smooth, index[:, 0], x0)
    mu_state_smooth = index_update(mu_state_smooth, index[:, n_eval], mu_state_filt[:, n_eval])
    var_state_smooth = index_update(var_state_smooth, index[:, :, n_eval],  var_state_filt[:, :, n_eval])
    x_state_temp = _state_sim(mu_state_smooth[:, n_eval],
                              var_state_smooth[:, :, n_eval],
                              z_state[:, n_eval-1])
    
    x_state_smooth = index_update(x_state_smooth, index[:, n_eval], x_state_temp)
    for t in range(n_eval-1, 0, -1):
        x_state_temp = smooth_sim(x_state_next=x_state_smooth[:, t+1],
                                  mu_state_filt=mu_state_filt[:, t],
                                  var_state_filt=var_state_filt[:, :, t],
                                  mu_state_pred=mu_state_pred[:, t+1],
                                  var_state_pred=var_state_pred[:, :, t+1],
                                  wgt_state=wgt_state,
                                  z_state=z_state[:, t-1])
        x_state_smooth = index_update(x_state_smooth, index[:, t], x_state_temp)
        
    return x_state_smooth.T
