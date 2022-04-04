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
import numpy as np
from rodeo.jax.kalmantv import *
from rodeo.jax.kalmantv import _state_sim
from rodeo.jax.utils import *
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
        fun: ODE function.
        t: Time point.
        theta: ODE parameter.
        wgt_meas (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.

    Returns:
        (tuple):
        - **x_meas** (ndarray(n_block, n_bmeas)): Interrogation variable.
        - **var_meas** (ndarray(n_block, n_bmeas, n_bmeas)): Interrogation variance.

    """
    n_block, n_bmeas, _ = wgt_meas.shape
    var_meas = np.zeros((n_block, n_bmeas, n_bmeas))
    for i in range(n_block):
        var_meas[i] = np.linalg.multi_dot([wgt_meas[i], var_state_pred[i], wgt_meas[i].T])

    # var_meas = jnp.linalg.multi_dot([wgt_meas, var_state_pred, wgt_meas.T])
    # x_state = jnp.ravel(mu_state_pred)
    # x_meas = jnp.reshape(fun(x_state, t, theta), newshape=(n_block, -1))
    x_meas = fun(mu_state_pred, t, theta)
    var_meas = jnp.array(var_meas)
    return x_meas, var_meas


def interrogate_chkrebtii(key, fun, t, theta,
                          wgt_meas, mu_state_pred, var_state_pred):
    r"""
    Interrogate method of Chkrebtii et al (2016); DOI: 10.1214/16-BA1017.

    Same arguments and returns as :func:`~ode_block_solve.interrogate_rodeo`.

    """
    n_block, n_bmeas, n_bstate = wgt_meas.shape
    z_state = jax.random.normal(key, (n_block, n_bstate))
    var_meas = np.zeros((n_block, n_bmeas, n_bmeas))
    x_state = np.zeros((n_block, n_bstate))
    for i in range(n_block):
        var_meas[i] = np.linalg.multi_dot([wgt_meas[i], var_state_pred[i], wgt_meas[i].T])
        x_state[i] = _state_sim(mu_state_pred[i], var_state_pred[i], z_state[i])
    # x_state = jnp.ravel(x_state)
    # x_meas = jnp.reshape(fun(x_state, t, theta), newshape=(n_block, -1))
    x_meas = fun(mu_state_pred, t, theta)
    var_meas = jnp.array(var_meas)
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
    mu_state_filt = np.zeros((n_eval+1, n_block, n_bstate))
    mu_state_pred = np.zeros((n_eval+1, n_block, n_bstate))
    var_state_filt = np.zeros((n_eval+1, n_block, n_bstate, n_bstate))
    var_state_pred = np.zeros((n_eval+1, n_block, n_bstate, n_bstate))

    # initialize
    mu_state_filt[0] = x0
    mu_state_pred[0] = x0

    for t in range(n_eval):
        key, subkey = jax.random.split(key)
        for b in range(n_block):
            mu_state_pred[t+1, b], var_state_pred[t+1, b] = \
                predict(
                    mu_state_past=mu_state_filt[t, b],
                    var_state_past=var_state_filt[t, b],
                    mu_state=mu_state[b],
                    wgt_state=wgt_state[b],
                    var_state=var_state[b]
                )
        # model interrogation
        x_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            t=tmin + (tmax-tmin)*(t+1)/n_eval,
            theta=theta,
            wgt_meas=wgt_meas,
            mu_state_pred=mu_state_pred[t+1],
            var_state_pred=var_state_pred[t+1]
        )
        for b in range(n_block):
            # kalman update
            mu_state_filt[t+1, b], var_state_filt[t+1, b] = \
                update(
                    mu_state_pred=mu_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b],
                    x_meas=x_meas[b],
                    mu_meas=mu_meas[b],
                    wgt_meas=wgt_meas[b],
                    var_meas=var_meas[b]
                )
    return mu_state_pred, var_state_pred, mu_state_filt, var_state_filt


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
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`x_t` at time :math:`t = a`.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mu_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        (ndarray(n_steps, n_state)): Sample solution for :math:`X_t` at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mu_state.shape
    key, subkey = jax.random.split(key)
    z_state = jax.random.normal(subkey, (n_eval, n_block, n_bstate))
    x_state_smooth = np.zeros((n_eval+1, n_block, n_bstate))
    x_state_smooth[0] = x0

    # forward pass
    mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = \
        _solve_filter(
            key=key,
            fun=fun, theta=theta, x0=x0,
            tmin=tmin, tmax=tmax, n_eval=n_eval,
            wgt_meas=wgt_meas, wgt_state=wgt_state,
            mu_state=mu_state, var_state=var_state,
            interrogate=interrogate
        )

    for b in range(n_block):
        x_state_smooth[n_eval, b] = \
            _state_sim(
                mu_state_filt[n_eval, b],
                var_state_filt[n_eval, b],
                z_state[n_eval-1, b])

    for t in range(n_eval-1, 0, -1):
        for b in range(n_block):
            x_state_smooth[t, b] = \
                smooth_sim(
                    x_state_next=x_state_smooth[t+1, b],
                    wgt_state=wgt_state[b],
                    mu_state_filt=mu_state_filt[t, b],
                    var_state_filt=var_state_filt[t, b],
                    mu_state_pred=mu_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b],
                    z_state=z_state[t-1, b]
                )
    
    # x_state_smooth = jnp.reshape(x_state_smooth, newshape=(-1, n_block*n_bstate))
    return jnp.array(x_state_smooth)

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
        x0 (float): Initial value of the state variable :math:`x_t` at time :math:`t = a`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        (tuple):
        - **mu_state_smooth** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mu_state.shape
    mu_state_smooth = np.zeros((n_eval+1, n_block, n_bstate))
    mu_state_smooth[0] = x0
    var_state_smooth = np.zeros((n_eval+1, n_block, n_bstate, n_bstate))

    # forward pass
    mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = \
        _solve_filter(
            key=key,
            fun=fun, theta=theta, x0=x0,
            tmin=tmin, tmax=tmax, n_eval=n_eval,
            wgt_meas=wgt_meas, wgt_state=wgt_state,
            mu_state=mu_state, var_state=var_state,
            interrogate=interrogate
        )
    
    mu_state_smooth[-1] = mu_state_filt[-1]
    var_state_smooth[-1] = var_state_filt[-1]
    # backward pass
    for t in range(n_eval-1, 0, -1):
        for b in range(n_block):
            mu_state_smooth[t, b], var_state_smooth[t, b] = \
                smooth_mv(
                    mu_state_next=mu_state_smooth[t+1, b],
                    var_state_next=var_state_smooth[t+1, b],
                    wgt_state=wgt_state[b],
                    mu_state_filt=mu_state_filt[t, b],
                    var_state_filt=var_state_filt[t, b],
                    mu_state_pred=mu_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b],
            )
    # mu_state_smooth = jnp.reshape(mu_state_smooth, newshape=(-1, n_block*n_bstate))
    # var_state_smooth = block_diag(jnp.array(var_state_smooth))
    return jnp.array(mu_state_smooth), jnp.array(var_state_smooth)


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
        x0 (float): Initial value of the state variable :math:`x_t` at time :math:`t = a`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        (tuple):
        - **x_state_smooth** (ndarray(n_steps, n_state)): Sample solution for :math:`X_t` at times :math:`t \in [a, b]`.
        - **mu_state_smooth** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mu_state.shape
    key, subkey = jax.random.split(key)
    z_state = jax.random.normal(subkey, (n_eval, n_block, n_bstate))
    x_state_smooth = np.zeros((n_eval+1, n_block, n_bstate))
    x_state_smooth[0] = x0
    mu_state_smooth = np.zeros((n_eval+1, n_block, n_bstate))
    mu_state_smooth[0] = x0
    var_state_smooth = np.zeros((n_eval+1, n_block, n_bstate, n_bstate))

    # forward pass
    mu_state_pred, var_state_pred, mu_state_filt, var_state_filt = \
        _solve_filter(
            key=key,
            fun=fun, theta=theta, x0=x0,
            tmin=tmin, tmax=tmax, n_eval=n_eval,
            wgt_meas=wgt_meas, wgt_state=wgt_state,
            mu_state=mu_state, var_state=var_state,
            interrogate=interrogate
        )

    mu_state_smooth[-1] = mu_state_filt[-1]
    var_state_smooth[-1] = var_state_filt[-1]

    for b in range(n_block):
        x_state_smooth[n_eval, b] = \
            _state_sim(
                mu_state_filt[n_eval, b],
                var_state_filt[n_eval, b],
                z_state[n_eval-1, b])

    # backward pass
    for t in range(n_eval-1, 0, -1):
        for b in range(n_block):
            x_state_smooth[t, b], mu_state_smooth[t, b], var_state_smooth[t, b] = \
                smooth(
                    x_state_next=x_state_smooth[t+1, b],
                    mu_state_next=mu_state_smooth[t+1, b],
                    var_state_next=var_state_smooth[t+1, b],
                    wgt_state=wgt_state[b],
                    mu_state_filt=mu_state_filt[t, b],
                    var_state_filt=var_state_filt[t, b],
                    mu_state_pred=mu_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b],
                    z_state=z_state[t-1, b]
                )
    # x_state_smooth = jnp.reshape(x_state_smooth, newshape=(-1, n_block*n_bstate))
    # mu_state_smooth = jnp.reshape(mu_state_smooth, newshape=(-1, n_block*n_bstate))
    # var_state_smooth = block_diag(jnp.array(var_state_smooth))
    return jnp.array(x_state_smooth), jnp.array(mu_state_smooth), jnp.array(var_state_smooth)
