# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
cimport cython
import numpy as np
cimport numpy as np

from probDE.utils import rand_mat
from KalmanTVODE cimport KalmanTVODE

DTYPE = np.double
ctypedef np.double_t DTYPE_t
    
cpdef _copynm(dest, source, name):
    if not source.data.f_contiguous:
        raise TypeError('{} is not f contiguous.'.format(name))
    if not source.shape==dest.shape:
        raise TypeError('{} has incorrect shape.'.format(name))
    dest[:] = source
    return source

cdef class KalmanODE:
    """
    Create an object with the method of a probabilistic ODE solver, :func:`kalman_ode`, based on the Kalman filter and smoother. 

    Args:
        n_state (int): Size of the state.
        n_meas (int): Size of the measure.
        tmin (double): First time point of the time interval to be evaluated; :math:`a`.
        tmax (double): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated,
            such that discretization timestep is :math:`dt = b/N`.
        fun (function): Higher order ODE function :math:`W x_n = F(x_n, n)` taking arguments :math:`x` and :math:`n`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        z_state (ndarray(n_state)): Random matrix for simulating from :math:`N(0, 1)`.

    """
    cdef int n_state, n_meas, n_eval, n_steps
    cdef double tmin, tmax
    cdef object fun
    cdef double[::1, :] _wgt_state
    cdef double[::1] _mu_state
    cdef double[::1, :] _var_state
    cdef double[::1, :] _z_state
        
    def __cinit__(self, int n_state, int n_meas, double tmin, double tmax, int n_eval, object fun,
                  double[::1] mu_state, double[::1, :] wgt_state, double[::1, :] var_state, 
                  double[::1, :] z_state=None):
        self.n_state = n_state
        self.n_meas = n_meas
        self.tmin = tmin
        self.tmax = tmax
        self.n_eval = n_eval
        self.n_steps = n_eval + 1
        self.fun = fun
        self._mu_state = np.empty(self.n_state, order='F')
        self._wgt_state = np.empty((self.n_state, self.n_state), order='F')
        self._var_state = np.empty((self.n_state, self.n_state), order='F')
        self._z_state = np.empty((self.n_state, 2*self.n_steps), order='F')

        # iniitalize kalman variables
        self._mu_state[:] = mu_state
        self._wgt_state[:] = wgt_state
        self._var_state[:] = var_state

        if z_state is not None:
            self._z_state = z_state
    
    @property
    def mu_state(self):
        return self._mu_state

    @mu_state.setter
    def mu_state(self, mu_state):
        _copynm(self._mu_state, mu_state, 'mu_state')
    
    @mu_state.deleter
    def mu_state(self):
        self._mu_state = None

    @property
    def var_state(self):
        return self._var_state

    @var_state.setter
    def var_state(self, var_state):
        _copynm(self._var_state, var_state, 'var_state')
    
    @var_state.deleter
    def var_state(self):
        self._var_state = None

    @property
    def wgt_state(self):
        return self._wgt_state

    @wgt_state.setter
    def wgt_state(self, wgt_state):
        _copynm(self._wgt_state, wgt_state, 'wgt_state')
    
    @wgt_state.deleter
    def wgt_state(self):
        self._wgt_state = None

    @property
    def z_state(self):
        return self._z_state

    @z_state.setter
    def z_state(self, z_state):
        _copynm(self._z_state, z_state, 'z_state')
    
    @z_state.deleter
    def z_state(self):
        self._z_state = None

    cpdef solve(self, double[::1] x0_state, double[::1, :] wgt_meas, object theta=None, bint sim_sol=True):
        r"""
        Returns an approximate solution to the higher order ODE

        .. math:: W x_n = F(x_n, t, \theta)

        on the time interval :math:`t \in [a, b]` with initial condition :math:`x_0 = x_0`.
        
        Args:
            x0_state (ndarray(n_state)): Initial value of the state variable :math:`x_n` at time :math:`t = 0`; :math:`x_0`.
            wgt_meas (ndarray(n_var, n_state)): Transition matrix defining the measure prior; :math:`W`.
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
        if self._z_state is None:
            self._z_state = rand_mat(2*(self.n_eval+1), self.n_state)
        
        if not np.asarray(x0_state).data.f_contiguous:
            raise TypeError('{} is not f contiguous.'.format('x0_state'))
        if len(x0_state)!=self.n_state:
            raise TypeError('{} has incorrect shape.'.format('x0_state'))
        
        if not np.asarray(wgt_meas).data.f_contiguous:
            raise TypeError('{} is not f contiguous.'.format('wgt_meas'))
        if np.asarray(wgt_meas).shape!=(self.n_meas, self.n_state):
            raise TypeError('{} has incorrect shape.'.format('wgt_meas'))

        # argumgents for kalman_filter and kalman_smooth
        cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooth = np.empty((self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooth = np.empty((self.n_state, self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooth = np.empty((self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        
        cdef np.ndarray[DTYPE_t, ndim=1] x_state = np.empty(self.n_state, dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=1] x_meas = np.empty(self.n_meas, dtype=DTYPE, order='F')
        
        cdef int t
        # forward pass
        ktvode = new KalmanTVODE(self.n_meas, self.n_state, self.n_steps, & x0_state[0],
                                & x_state[0], & self._mu_state[0], & self._wgt_state[0, 0],
                                & self._var_state[0, 0], & x_meas[0], & wgt_meas[0, 0], 
                                & self._z_state[0, 0], & x_state_smooth[0, 0],
                                & mu_state_smooth[0, 0], & var_state_smooth[0, 0, 0])

        for t in range(self.n_eval):
            # kalman filter:
            ktvode.predict(t)
            ktvode.forecast(t)
            #ktvode.forecast_sch(t)
            self.fun(x_state, self.tmin + (self.tmax-self.tmin)*(t+1)/self.n_eval, theta, x_meas)
            ktvode.update(t)
        
        # backward pass
        ktvode.smooth_update(sim_sol)
        del ktvode
        if sim_sol:
            return x_state_smooth.T, None
        else:
            return mu_state_smooth.T, var_state_smooth.T
        