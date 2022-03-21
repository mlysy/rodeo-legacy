r"""
Stochastic block solver for ODE initial value problems.

In the notation defined in the `ode_solve`, recall that the Gaussian state space model underlying the ODE-IVP is

:: math::

X_n = Q(X_{n-1} - \lambda) + \lambda + R^{1/2} \epsilon_n

y_n = W X_n + \Sigma_n \eta_n.

This module optimizes the calculations when :math:`Q`, :math:`R`, and :math:`W`, are block diagonal matrices of conformable and "stackable" sizes.  That is, recall that the dimension of these matrices are `n_state x n_state`, `n_state x n_state`, and `n_meas x n_state`, respectively.  Then suppose that :math:`Q` and :math:`R` consist of `n_block` blocks of size `n_bstate x n_bstate`, where `n_bstate = n_state/n_block`, and :math:`W` consists of `n_block` blocks of size `n_bmeas x n_bstate`, where `n_bmeas = n_meas/n_block`.  Then :math:`Q`, :math:`R`, :math:`W` can be stored as 3D arrays of size `n_block x n_bstate x n_bstate` and `n_block x n_bmeas x n_bstate`.  It is under this paradigm that the `ode_block_solve` module operates.

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
    Rodeo interrogation method.

    Args:
        key: JAX PRNG key.
        wgt_meas (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.

    Returns:
        (tuple):
        - **x_meas** (ndarray(n_block, n_bmeas)): Interrogation variable.
        - **var_meas** (ndarray(n_block, n_bmeas, n_bmeas)): Interrogation variance.

    """
    n_block = mu_state_pred.shape[0]
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        wgt_meas, var_state_pred
    )
    # var_meas = jnp.linalg.multi_dot([wgt_meas, var_state_pred, wgt_meas.T])
    x_state = jnp.ravel(mu_state_pred)
    x_meas = jnp.reshape(fun(x_state, t, theta), newshape=(n_block, -1))
    return x_meas, var_meas


def _solve_filter(key, fun, x0, theta,
                  tmin, tmax, n_eval,
                  wgt_meas, wgt_state, mu_state, var_state,
                  interrogate=interrogate_rodeo):
    r"""
    Forward pass of the ODE solver.

    FIXME: Document all arguments and outputs...
    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = wgt_meas.shape
    #n_state = len(mu_state)

    # arguments for kalman_filter and kalman_smooth
    mu_meas = jnp.zeros((n_block, n_bmeas))
    mu_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # lax.scan setup
    # scan function
    def scan_fun(carry, t):
        mu_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"], num=1)
        # kalman predict
        mu_state_pred, var_state_pred = jax.vmap(lambda b:
            predict(
                mu_state_past=mu_state_filt[b],
                var_state_past=var_state_filt[b],
                mu_state=mu_state[b],
                wgt_state=wgt_state[b],
                var_state=var_state[b]
            )
        )(jnp.arange(n_block))
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
        mu_state_next, var_state_next = jax.vmap(lambda b:
            update(
                mu_state_pred=mu_state_pred[b],
                var_state_pred=var_state_pred[b],
                x_meas=x_meas[b],
                mu_meas=mu_meas[b],
                wgt_meas=wgt_meas[b],
                var_meas=var_meas[b]
            )
        )(jnp.arange(n_block))
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
    n_block, n_bstate = mu_state.shape
    #key, subkey = jax.random.split(key)
    z_state = jax.random.normal(key, (n_eval, n_block, n_bstate))

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
        mu_state_filt = smooth_kwargs['mu_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mu_state_pred = smooth_kwargs['mu_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        z_state = smooth_kwargs['z_state']
        x_state_curr = jax.vmap(lambda b:
            smooth_sim(
                x_state_next=x_state_next[b],
                wgt_state=wgt_state[b],
                # **smooth_kwargs
                mu_state_filt=mu_state_filt[b],
                var_state_filt=var_state_filt[b],
                mu_state_pred=mu_state_pred[b],
                var_state_pred=var_state_pred[b],
                z_state=z_state[b]
                )
        )(jnp.arange(n_block))
        return x_state_curr, x_state_curr
    # initialize
    scan_init = jax.vmap(lambda b: 
                         _state_sim(mu_state_filt[n_eval, b],
                         var_state_filt[n_eval, b],
                         z_state[n_eval-1, b]))(jnp.arange(n_block))
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
    x_state_smooth = jnp.reshape(x_state_smooth, newshape=(-1, n_block*n_bstate))
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
    key, subkey = jax.random.split(key)
    filt_out = _solve_filter(
        key=subkey,
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
    key, subkey = jax.random.split(key)
    filt_out = _solve_filter(
        key=subkey,
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
        x_state_curr, mu_state_curr, var_state_curr = smooth_mv(
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
