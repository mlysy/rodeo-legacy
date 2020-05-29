cimport cython
import numpy as np
cimport numpy as np

from probDE.utils import rand_mat
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
                 theta,
                 bint smooth_mv=True,
                 bint smooth_sim=False,
                 bint offline=False):
    """
    Probabilistic ODE solver based on the Kalman filter and smoother. Returns an approximate solution to the higher order ODE

    .. math:: w' x_t = F(x_t, t, \\theta)

    on the time interval :math:`t \in [a, b]` with initial condition :math:`x_0 = x_0`. The corresponding variable names are

    The specific model we are using to approximate the solution :math:`x_n` is

    .. math::

        X_n = c + T X_{n-1} + R_n^{1/2} \epsilon_n

        y_n = d + W X_n + H_n^{1/2} \eta_n

    where :math:`\epsilon_n` and :math:`\eta_n` are independent :math:`N(0,1)` distributions and
    :math:`X_n = (x_n, y_n)` at time n and :math:`y_n` denotes the observation at time n.

    Args:
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        x0_state (ndarray(n_state)): Initial value of the state variable :math:`x_t` at time :math:`t = 0`; :math:`x_0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated,
            such that discretization timestep is :math:`dt = (b-a)/N`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.
        z_state_sim (ndarray(n_state, 2*n_steps)): Random N(0,1) matrix for forecasting and smoothing.
        x_meass (ndarray(n_state, n_steps)): Optional offline observations.
        theta (ndarray(n_theta)): Parameter in the ODE function.
        smooth_mv (bool): Flag for returning the smoothed mean and variance.
        smooth_sim (bool): Flag for returning the smoothed simulated state.
        offline (bool): Flag for offline forecasted `x_meas`. 

    Returns:
        (tuple):
        - **x_state_smooths** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [a...b].
        - **mu_state_smooths** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`y_n` at times
          [a...b].
        - **var_state_smooths** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at
          times [a...b].

    """
    # Dimensions of state and measure variables
    cdef int n_meas = wgt_meas.shape[0]
    cdef int n_state = mu_state.shape[0]
    cdef int n_steps = n_eval + 1
    # argumgents for kalman_filter and kalman_smooth
    cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooths = np.zeros((n_state, n_steps),
                                                                 dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooths = np.zeros((n_state, n_state, n_steps),
                                                                  dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooths = np.zeros((n_state, n_steps),
                                                                dtype=DTYPE, order='F')
    
    cdef np.ndarray[DTYPE_t, ndim=1] x_state = np.zeros(n_state, dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, ndim=2] x_meas = np.zeros((n_meas, n_steps), dtype=DTYPE, order='F')

    cdef int t
    # forward pass
    if not offline:
        ktvode = new KalmanTVODE(n_meas, n_state, n_steps, & x0_state[0],
                                 & x_state[0], & mu_state[0], & wgt_state[0, 0],
                                 & var_state[0, 0], & x_meas[0, 0], & wgt_meas[0, 0], 
                                 & z_state_sim[0, 0], & x_state_smooths[0, 0],
                                 & mu_state_smooths[0, 0], & var_state_smooths[0, 0, 0])
    else:
        ktvode = new KalmanTVODE(n_meas, n_state, n_steps, & x0_state[0],
                                 & x_state[0], & mu_state[0], & wgt_state[0, 0],
                                 & var_state[0, 0], & x_meass[0, 0], & wgt_meas[0, 0], 
                                 & z_state_sim[0, 0], & x_state_smooths[0, 0],
                                 & mu_state_smooths[0, 0], & var_state_smooths[0, 0, 0])
    for t in range(n_eval):
        if not offline:
            # kalman filter:
            ktvode.predict(t)
            ktvode.forecast(t)
            #ktvode.forecast_sch(t)
            x_meas[:, t+1] = fun(x_state, tmin + (tmax-tmin)*(t+1)/n_eval, theta)
            ktvode.update(t)
        else:
            ktvode.filter(t)
    # backward pass
    ktvode.smooth_update(smooth_mv, smooth_sim)
    del ktvode
    if smooth_mv and smooth_sim:
        return x_state_smooths, mu_state_smooths, var_state_smooths
    elif smooth_mv:
        return mu_state_smooths, var_state_smooths
    elif smooth_sim:
        return x_state_smooths

cdef class KalmanODE:
    """
    Create an object with the method of a probabilistic ODE solver, :func:`kalman_ode`, based on the Kalman filter and smoother. 

    Args:
        n_state (int): Size of the state.
        n_meas (int): Size of the measure.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated,
            such that discretization timestep is :math:`dt = b/N`.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        wgt_meas (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.
        z_states (ndarray(n_state)): Random matrix for simulating from :math:`N(0, 1)`.

    """
    cdef int n_state, n_meas, n_eval
    cdef double tmin, tmax
    cdef object fun
    cdef object __wgt_state
    cdef object __mu_state
    cdef object __var_state
    cdef object __wgt_meas
    cdef object __z_states
        
    def __cinit__(self, int n_state, int n_meas, double tmin, double tmax, int n_eval, object fun, **init):
        self.n_state = n_state
        self.n_meas = n_meas
        self.tmin = tmin
        self.tmax = tmax
        self.n_eval = n_eval
        self.fun = fun
        self.__wgt_state = None
        self.__mu_state = None
        self.__var_state = None
        self.__wgt_meas = None
        self.__z_states = None
        for key in init.keys():
            self.__setattr__(key, init[key])
    
    @property
    def mu_state(self):
        return self.__mu_state

    @mu_state.setter
    def mu_state(self, mu_state):
        self.__mu_state = mu_state.copy('F')
    
    @mu_state.deleter
    def mu_state(self):
        self.__mu_state = None

    @property
    def var_state(self):
        return self.__var_state

    @var_state.setter
    def var_state(self, var_state):
        self.__var_state = var_state.copy('F')
    
    @var_state.deleter
    def var_state(self):
        self.__var_state = None

    @property
    def wgt_state(self):
        return self.__wgt_state

    @wgt_state.setter
    def wgt_state(self, wgt_state):
        self.__wgt_state = wgt_state.copy('F')
    
    @wgt_state.deleter
    def wgt_state(self):
        self.__wgt_state = None

    @property
    def wgt_meas(self):
        return self.__wgt_meas

    @wgt_meas.setter
    def wgt_meas(self, wgt_meas):
        self.__wgt_meas = wgt_meas.copy('F')
    
    @wgt_meas.deleter
    def wgt_meas(self):
        self.__wgt_meas = None

    @property
    def z_states(self):
        return self.__z_states

    @z_states.setter
    def z_states(self, z_states):
        self.__z_states = z_states.copy('F')
    
    @z_states.deleter
    def z_states(self):
        self.__z_states = None
    
    cpdef solve(self, double[::1] x0_state, theta=None, bint mv=False, bint sim=True):
        """
        Returns an approximate solution to the higher order ODE

        .. math:: w' x_t = F(x_t, t, \\theta)

        on the time interval :math:`t \in [a, b]` with initial condition :math:`x_0 = x_0`.
        
        Args:
            x0_state (ndarray(n_state)): Initial value of the state variable :math:`x_t` at time :math:`t = 0`; :math:`x_0`.
            theta (ndarray(n_theta)): Parameter in the ODE function.
            smooth_mv (bool): Flag for returning the smoothed mean and variance.
            smooth_sim (bool): Flag for returning the smoothed simulated state.
        
        Returns:
            (tuple):
            - **kalman_sim** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [a...b].
            - **kalman_mu** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`y_n` at times
              [a...b].
            - **kalman_var** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at
              times [a...b].

        """
        if (self.__wgt_state is None or self.__mu_state is None or 
           self.__var_state is None or self.__wgt_meas is None):
            raise ValueError("wgt_state, mu_state, var_state, and/or wgt_meas is not set.")
        
        if self.__z_states is None:
            self.__z_states = rand_mat(2*(self.n_eval+1), self.n_state)
        if mv and sim:
            kalman_sim, kalman_mu, kalman_var = \
                kalman_ode(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                          self.__wgt_state, self.__mu_state, self.__var_state,
                          self.__wgt_meas, self.__z_states, None, theta, mv, sim)
            kalman_sim = np.ascontiguousarray(kalman_sim.T)
            kalman_mu = np.ascontiguousarray(kalman_mu.T)
            kalman_var = np.ascontiguousarray(kalman_var.T)
            return kalman_sim, kalman_mu, kalman_var
        elif mv:
            kalman_mu, kalman_var = \
                kalman_ode(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                          self.__wgt_state, self.__mu_state, self.__var_state,
                          self.__wgt_meas, self.__z_states, None, theta, mv, sim)
            kalman_mu = np.ascontiguousarray(kalman_mu.T)
            kalman_var = np.ascontiguousarray(kalman_var.T)
            return kalman_mu, kalman_var
        elif sim:
            kalman_sim = \
                kalman_ode(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                          self.__wgt_state, self.__mu_state, self.__var_state,
                          self.__wgt_meas, self.__z_states, None, theta, mv, sim)
            kalman_sim = np.ascontiguousarray(kalman_sim.T)
            return kalman_sim
        