"""
.. module:: kalman_ode

Probabilistic ODE solver based on the Kalman filter and smoother.

"""
cimport cython
import numpy as np
cimport numpy as np

from KalmanTVODE cimport KalmanTVODE

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef kalman_ode(fun,
                 double[::1] x0_state,
                 double tmin,
                 double tmax,
                 int n_eval,
                 double[::1, :] wgt_state,
                 double[::1] mu_state, 
                 double[::1, :] var_state,
                 double[::1, :] wgt_meas, 
                 double[::1, :] z_state_sim,
                 double[::1, :] x_meass,
                 bint smooth_mv=True,
                 bint smooth_sim=False,
                 bint offline=False):
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
        z_state_sim (ndarray(n_dim_state, 2*n_steps)): Random N(0,1) matrix for forecasting and smoothing.
        smooth_mv (bool): Flag for returning the smoothed mean and variance.
        smooth_sim (bool): Flag for returning the smoothed simulated state.
        offline (bool): Flag for offline forecasted `x_meas`. 

    Returns:
        (tuple):
        - **x_state_smooths** (ndarray(n_timesteps, n_dim_state)): Sample solution at time t given observations from times [0...N] for
          :math:`t = 0,1/N,\ldots,1`.
        - **mu_state_smooths** (ndarray(n_timesteps, n_dim_state)): Posterior mean of the solution process :math:`y_n` at times
          :math:`t = 0,1/N,\ldots,1`.
        - **var_state_smooths** (ndarray(n_timesteps, n_dim_state, n_dim_state)): Posterior variance of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.

    """
    # Dimensions of state and measure variables
    cdef int n_dim_meas = wgt_meas.shape[0]
    cdef int n_dim_state = mu_state.shape[0]
    cdef int n_steps = n_eval + 1
    # argumgents for kalman_filter and kalman_smooth
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooths = np.zeros((n_dim_state, n_steps),
                                                                 dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooths = np.zeros((n_dim_state, n_dim_state, n_steps),
                                                                  dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooths = np.zeros((n_dim_state, n_steps),
                                                                dtype=DTYPE, order='F')
    
    cdef np.ndarray[DTYPE_t, ndim=1] x_state = np.zeros(n_dim_state, dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] x_meas = np.zeros((n_dim_meas, n_steps), dtype=DTYPE, order='F')

    cdef int t
    # forward pass
    if not offline:
        ktvode = new KalmanTVODE(n_dim_meas, n_dim_state, n_steps, & x0_state[0],
                                & x_state[0], & mu_state[0], & wgt_state[0, 0],
                                & var_state[0, 0], & x_meas[0, 0], & wgt_meas[0, 0], 
                                & z_state_sim[0, 0], & x_state_smooths[0, 0],
                                & mu_state_smooths[0, 0], & var_state_smooths[0, 0, 0])
    else:
        ktvode = new KalmanTVODE(n_dim_meas, n_dim_state, n_steps, & x0_state[0],
                                & x_state[0], & mu_state[0], & wgt_state[0, 0],
                                & var_state[0, 0], & x_meass[0, 0], & wgt_meas[0, 0], 
                                & z_state_sim[0, 0], & x_state_smooths[0, 0],
                                & mu_state_smooths[0, 0], & var_state_smooths[0, 0, 0])
    for t in range(n_eval):
        if not offline:
            # kalman filter:
            ktvode.predict(t)
            ktvode.forecast(t)
            x_meas[:, t+1] = fun(x_state, tmin + (tmax-tmin)*(t+1)/n_eval)
            ktvode.update(t)
        else:
            ktvode.filter(t)
    # backward pass
    ktvode.smooth_update(smooth_mv, smooth_sim)
    if smooth_mv and smooth_sim:
        return x_state_smooths, mu_state_smooths, var_state_smooths
    elif smooth_mv:
        return mu_state_smooths, var_state_smooths
    elif smooth_sim:
        return x_state_smooths
