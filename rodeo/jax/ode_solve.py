r"""
Stochastic solver for ODE initial value problems.

The ODE-IVP to be solved is defined as

.. math:: W X_t = F(X_t, t, \theta)

on the time interval :math:`t \in [a, b]` with initial condition :math:`X_a = x_0`.  

The stochastic solver proceeds via Kalman filtering and smoothing of "interrogations" of the ODE model as described in Chkrebtii et al 2016, Schober et al 2019.  In the context of the underlying Kalman filterer/smoother, the Gaussian state-space model is

:: math::

X_n = Q(X_{n-1} - \lambda) + \lambda + R^{1/2} \epsilon_n

y_n = W X_n + \Sigma_n \eta_n,

where :math:`\epsilon_n` and :math:`\eta_n` are iid standard normals the size of :math:`X_t` and :math:`W X_t`, respectively, and where :math:`(y_n, \Sigma_n)` are generated sequentially using the ODE function :math:`F(X_t, t, \theta)` as explained in the references above.  Thus, much of the notation here is identical to that in the `kalmantv` module.
  
"""

# import numpy as np
import jax
import jax.numpy as jnp
from rodeo.jax.kalmantv import *
from rodeo.jax.kalmantv import _state_sim
# from jax import jit, lax, random
# from functools import partial
# from jax.config import config
# from kalmantv.jax.kalmantv import *
# from kalmantv.jax.kalmantv import _state_sim
# config.update("jax_enable_x64", True)


def interrogate_rodeo(key, fun, t, theta,
                      wgt_meas, mu_state_pred, var_state_pred):
    r"""
    Interrogate method of rodeo.

    Args:
        key: JAX PRNG key.
        fun: ODE function.
        t: Time point.
        theta: ODE parameter.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.

    Returns:
        (tuple):
        - **x_meas** (ndarray(n_state)): Interrogation variable.
        - **var_meas** (ndarray(n_meas, n_meas)): Interrogation variance.

    """
    var_meas = jnp.atleast_2d(
        jnp.linalg.multi_dot([wgt_meas, var_state_pred, wgt_meas.T])
    )
    x_state = mu_state_pred
    x_meas = fun(x_state, t, theta)
    return x_meas, var_meas

def interrogate_chkrebtii(key, fun, t, theta,
                          wgt_meas, mu_state_pred, var_state_pred):
    r"""
    Interrogate method of Chkrebtii et al (2016); DOI: 10.1214/16-BA1017.

    Args:
        key: JAX PRNG key.
        fun: ODE function.
        t: Time point.
        theta: ODE parameter.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.

    Returns:
        (tuple):
        - **x_meas** (ndarray(n_state)): Interrogation variable.
        - **var_meas** (ndarray(n_meas, n_meas)): Interrogation variance.

    """
    #key, subkey = jax.random.split(key)
    n_state = len(mu_state_pred)
    z_state = jax.random.normal(key, (n_state, ))
    var_meas = jnp.atleast_2d(
        jnp.linalg.multi_dot([wgt_meas, var_state_pred, wgt_meas.T])
    )
    x_state = _state_sim(mu_state_pred, var_state_pred, z_state)
    x_meas = fun(x_state, t, theta)
    return x_meas, var_meas

def interrogate_schober(key, fun, t, theta,
                        wgt_meas, mu_state_pred, var_state_pred):
    r"""
    Interrogate method of Schober et al (2019); DOI: https://doi.org/10.1007/s11222-017-9798-7.

    Args:
        key: JAX PRNG key.
        fun: ODE function.
        t: Time point.
        theta: ODE parameter.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.

    Returns:
        (tuple):
        - **x_meas** (ndarray(n_state)): Interrogation variable.
        - **var_meas** (ndarray(n_meas, n_meas)): Interrogation variance.

    """
    n_meas = wgt_meas.shape[0]
    var_meas = jnp.zeros((n_meas, n_meas))
    x_state = mu_state_pred
    x_meas = fun(x_state, t, theta)
    return x_meas, var_meas


def _solve_filter(key, fun, x0, theta,
                  tmin, tmax, n_eval,
                  wgt_meas, wgt_state, mu_state, var_state,
                  interrogate=interrogate_rodeo):
    r"""
    Forward pass of the ODE solver.

    FIXME: Document all arguments and outputs...
    """
    # Dimensions of state and measure variables
    n_meas = wgt_meas.shape[0]
    n_state = len(mu_state)

    # arguments for kalman_filter and kalman_smooth
    mu_meas = jnp.zeros(n_meas)
    mu_state_init = x0
    var_state_init = jnp.zeros((n_state, n_state))

    # lax.scan setup
    # scan function
    def scan_fun(carry, t):
        mu_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"])
        # kalman predict
        mu_state_pred, var_state_pred = predict(
            mu_state_past=mu_state_filt,
            var_state_past=var_state_filt,
            mu_state=mu_state,
            wgt_state=wgt_state,
            var_state=var_state
        )
        # model interrogation
        x_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            t=tmin + (tmax-tmin)*(t+1)/n_eval,
            theta=theta,
            wgt_meas=wgt_meas,
            mu_state_pred=mu_state_pred,
            var_state_pred=var_state_pred
        )
        # kalman update
        mu_state_next, var_state_next = update(
            mu_state_pred=mu_state_pred,
            var_state_pred=var_state_pred,
            x_meas=x_meas,
            mu_meas=mu_meas,
            wgt_meas=wgt_meas,
            var_meas=var_meas
        )
        # output
        carry = {
            "state_filt": (mu_state_next, var_state_next),
            "key": key
        }
        stack = {
            "state_filt": (mu_state_next, var_state_next),
            "state_pred": (mu_state_pred, var_state_pred)
        }
        return carry, stack
    # scan initial value
    scan_init = {
        "state_filt": (mu_state_init, var_state_init),
        "key": key
    }
    # scan itself
    _, scan_out = jax.lax.scan(scan_fun, scan_init, jnp.arange(n_eval))
    # append initial values to front
    scan_out["state_filt"] = (
        jnp.concatenate([mu_state_init[None], scan_out["state_filt"][0]]),
        jnp.concatenate([var_state_init[None], scan_out["state_filt"][1]])
    )
    scan_out["state_pred"] = (
        jnp.concatenate([mu_state_init[None], scan_out["state_pred"][0]]),
        jnp.concatenate([var_state_init[None], scan_out["state_pred"][1]])
    )
    return scan_out


def solve_sim(key, fun, x0, theta,
              tmin, tmax, n_eval,
              wgt_meas, wgt_state, mu_state, var_state,
              interrogate=interrogate_rodeo):
    r"""
    Random draw from the stochastic ODE solver.

    Args:
        key: PRNG key.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        x0 (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        **x_state_smooth** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [0...N] for :math:`t = 0,1/N,\ldots,1`.

    """
    n_state = len(mu_state)
    key, subkey = jax.random.split(key)
    z_state = jax.random.normal(subkey, (n_eval, n_state))

    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, theta=theta, x0=x0,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=wgt_meas, wgt_state=wgt_state,
        mu_state=mu_state, var_state=var_state,
        interrogate=interrogate
    )
    mu_state_pred, var_state_pred = filt_out["state_pred"]
    mu_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    # lax.scan setup
    def scan_fun(x_state_next, smooth_kwargs):
        x_state_curr = smooth_sim(
            x_state_next=x_state_next,
            wgt_state=wgt_state,
            **smooth_kwargs
            # mu_state_filt=mu_state_filt[t],
            # var_state_filt=var_state_filt[t],
            # mu_state_pred=var_state_pred[t+1],
            # var_state_pred=var_state_pred[t+1],
            # wgt_state=wgt_state,
            # z_state=z_state[t-1]
        )
        return x_state_curr, x_state_curr
    # initialize
    scan_init = _state_sim(mu_state_filt[n_eval],
                           var_state_filt[n_eval],
                           z_state[n_eval-1])
    # scan arguments
    scan_kwargs = {
        'mu_state_filt': mu_state_filt[1:n_eval],
        'var_state_filt': var_state_filt[1:n_eval],
        'mu_state_pred': mu_state_pred[2:n_eval+1],
        'var_state_pred': var_state_pred[2:n_eval+1],
        'z_state': z_state[:n_eval-1]
    }
    # Note: initial value x0 is assumed to be known, so we don't
    # sample it.  In fact, doing so would probably fail due to cholesky
    # of a zero variance matrix...
    # kwargs_init = {
    #     'mu_state_filt': mu_state_filt[n_eval],
    #     'var_state_filt': var_state_filt[n_eval],
    #     'mu_state_pred': mu_state_pred[n_eval+1],
    #     'var_state_pred': var_state_pred[n_eval+1],
    #     'z_state': z_state[n_eval-1]}
    # scan_fun(scan_init, kwargs_init)
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [x0[None], scan_out, scan_init[None]]
    )
    return x_state_smooth


def solve_mv(key, fun, x0, theta,
             tmin, tmax, n_eval,
             wgt_meas, wgt_state, mu_state, var_state,
             interrogate=interrogate_rodeo):
    r"""
    Mean and variance of the stochastic ODE solver.

    Args:
        key: PRNG key.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        x0 (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        (tuple):
        - **mu_state_smooth** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`X_t` at times
          :math:`t = 0,1/N,\ldots,1`.
        - **var_state_smooth** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.

    """
    n_state = len(mu_state)
    # forward pass
    # key, subkey = jax.random.split(key)
    filt_out = _solve_filter(
        key=key,
        fun=fun, theta=theta, x0=x0,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=wgt_meas, wgt_state=wgt_state,
        mu_state=mu_state, var_state=var_state,
        interrogate=interrogate
    )
    mu_state_pred, var_state_pred = filt_out["state_pred"]
    mu_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mu_state_curr, var_state_curr = smooth_mv(
            mu_state_next=state_next["mu"],
            var_state_next=state_next["var"],
            wgt_state=wgt_state,
            **smooth_kwargs
            # mu_state_filt=mu_state_filt[t],
            # var_state_filt=var_state_filt[t],
            # mu_state_pred=var_state_pred[t+1],
            # var_state_pred=var_state_pred[t+1],
            # wgt_state=wgt_state
        )
        state_curr = {
            "mu": mu_state_curr,
            "var": var_state_curr
        }
        return state_curr, state_curr
    # initialize
    scan_init = {
        "mu": mu_state_filt[n_eval],
        "var": var_state_filt[n_eval]
    }
    # scan arguments
    scan_kwargs = {
        'mu_state_filt': mu_state_filt[1:n_eval],
        'var_state_filt': var_state_filt[1:n_eval],
        'mu_state_pred': mu_state_pred[2:n_eval+1],
        'var_state_pred': var_state_pred[2:n_eval+1]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    mu_state_smooth = jnp.concatenate(
        [x0[None], scan_out["mu"], scan_init["mu"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_state, n_state))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mu_state_smooth, var_state_smooth


def solve(key, fun, x0, theta,
          tmin, tmax, n_eval,
          wgt_meas, wgt_state, mu_state, var_state,
          interrogate=interrogate_rodeo):
    r"""
    Both random draw and mean/variance of the stochastic ODE solver. 

    Args:
        key: PRNG key.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        x0 (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        **x_state_smooth** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [0...N] for
          :math:`t = 0,1/N,\ldots,1`.

    """
    n_state = len(mu_state)
    key, subkey = jax.random.split(key)
    z_state = jax.random.normal(subkey, (n_eval, n_state))

    # forward pass
    #key, subkey = jax.random.split(key)
    filt_out = _solve_filter(
        key=key,
        fun=fun, theta=theta, x0=x0,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=wgt_meas, wgt_state=wgt_state,
        mu_state=mu_state, var_state=var_state,
        interrogate=interrogate
    )
    mu_state_pred, var_state_pred = filt_out["state_pred"]
    mu_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        x_state_curr, mu_state_curr, var_state_curr = smooth(
            x_state_next=state_next["x"],
            mu_state_next=state_next["mu"],
            var_state_next=state_next["var"],
            wgt_state=wgt_state,
            **smooth_kwargs
        )
        state_curr = {
            "x": x_state_curr,
            "mu": mu_state_curr,
            "var": var_state_curr
        }
        return state_curr, state_curr
    # initialize
    scan_init = {
        "x": _state_sim(mu_state_filt[n_eval],
                        var_state_filt[n_eval],
                        z_state[n_eval-1]),
        "mu": mu_state_filt[n_eval],
        "var": var_state_filt[n_eval]
    }
    # scan arguments
    # Slice these arrays so they are aligned.
    # More precisely, for time step t, want filt[t], pred[t+1], z_state[t-1]
    scan_kwargs = {
        'mu_state_filt': mu_state_filt[1:n_eval],
        'var_state_filt': var_state_filt[1:n_eval],
        'mu_state_pred': mu_state_pred[2:n_eval+1],
        'var_state_pred': var_state_pred[2:n_eval+1],
        'z_state': z_state[:n_eval-1]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [x0[None], scan_out["x"], scan_init["x"][None]]
    )
    mu_state_smooth = jnp.concatenate(
        [x0[None], scan_out["mu"], scan_init["mu"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_state, n_state))[None], scan_out["var"],
         scan_init["var"][None]]
    )

    return x_state_smooth, mu_state_smooth, var_state_smooth
