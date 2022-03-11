# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
cimport cython
import numpy as np
cimport numpy as np

from rodeo.utils import rand_mat
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
    r"""
    Probabilistic ODE solver based on the Kalman filter and smoother. Returns an approximate solution to the higher order ODE

    .. math:: W x_n = F(x_n, n, \theta)

    on the time interval :math:`n \in [0, T]` with initial condition :math:`x_0 = x_0`. 
    The specific model we are using to approximate the solution :math:`x_n` is

    .. math::

        x_n = Q(x_{n-1} -\lambda) + \lambda + R_n^{1/2} \epsilon_n

        y_n = W x_n + \Sigma_n^{1/2} \eta_n

    where :math:`\epsilon_n` and :math:`\eta_n` are independent :math:`N(0,1)` distributions and
    :math:`y_n` denotes the model interrogation (observation) at time n.

    Args:
        W (ndarray(n_var, n_state)): Transition matrix defining the measure prior; :math:`W`.
        tmin (int): First time point of the time interval to be evaluated.
        tmax (int): Last time point of the time interval to be evaluated; :math:`T`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated,
            such that discretization timestep is :math:`dt = T/N`.
        ode_fun (function): Higher order ODE function :math:`W x_n = F(x_n, n)` taking arguments :math:`x` and :math:`n`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; denoted by :math:`\lambda`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R`.
        z_state (ndarray(n_state, 2*n_eval)): Random N(0,1) matrix for forecasting and smoothing.

    """
    cdef int n_state, n_meas, n_eval, n_steps
    cdef double tmin, tmax
    cdef object ode_fun
    cdef double[::1, :] _wgt_meas
    cdef double[::1, :] _wgt_state
    cdef double[::1] _mu_state
    cdef double[::1, :] _var_state
    cdef double[::1, :] _z_state
        
    def __cinit__(self, double[::1, :] W, double tmin, double tmax, int n_eval, object ode_fun,
                  double[::1] mu_state, double[::1, :] wgt_state, double[::1, :] var_state, 
                  double[::1, :] z_state=None):
        self.n_state = W.shape[1]
        self.n_meas = W.shape[0]
        self.tmin = tmin
        self.tmax = tmax
        self.n_eval = n_eval
        self.n_steps = n_eval + 1
        self.ode_fun = ode_fun
        self._wgt_meas = np.empty((self.n_meas, self.n_state), order='F')
        self._mu_state = np.empty(self.n_state, order='F')
        self._wgt_state = np.empty((self.n_state, self.n_state), order='F')
        self._var_state = np.empty((self.n_state, self.n_state), order='F')
        self._z_state = np.zeros((self.n_state, 2*self.n_eval), order='F')

        # iniitalize kalman variables
        self._wgt_meas[:] = W
        self._mu_state[:] = mu_state
        self._wgt_state[:] = wgt_state
        self._var_state[:] = var_state

        if z_state is not None:
            self.z_state[:] = z_state

    @property
    def wgt_meas(self):
        return self._wgt_meas
    
    @wgt_meas.setter
    def wgt_meas(self, wgt_meas):
        _copynm(self._wgt_meas, wgt_meas, 'wgt_meas')
    
    @wgt_meas.deleter
    def wgt_meas(self):
        self._wgt_meas = None
    
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
        self._z_state = np.zeros((self.n_state, 2*self.n_eval), order='F')

    cdef KalmanTVODE * _solve_filter(self, double[::1, :] x_state_smooth, double[::1, :] mu_state_smooth,
                                     double[::1, :, :] var_state_smooth, double[::1] x0, object theta=None,
                                     method="rodeo"):
        r"""
        Forward pass filter step in the KalmanODE solver. 
        
        """
        if not np.any(self._z_state):
            self.z_state[:] = rand_mat(2*self.n_eval, self.n_state)
        
        if not np.asarray(x0).data.f_contiguous:
            raise TypeError('{} is not f contiguous.'.format('x0'))
        if len(x0)!=self.n_state:
            raise TypeError('{} has incorrect shape.'.format('x0'))
        

        # argumgents for kalman_filter and kalman_smooth
        cdef np.ndarray[DTYPE_t, ndim=1] x_state = np.empty(self.n_state, dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=1] x_meas = np.empty(self.n_meas, dtype=DTYPE, order='F')

        cdef int t
        ktvode = new KalmanTVODE(self.n_meas, self.n_state, self.n_steps, & x0[0],
                                & x_state[0], & self._mu_state[0], & self._wgt_state[0, 0],
                                & self._var_state[0, 0], & x_meas[0], & self._wgt_meas[0, 0], 
                                & self._z_state[0, 0], & x_state_smooth[0, 0],
                                & mu_state_smooth[0, 0], & var_state_smooth[0, 0, 0])

        # forward pass
        for t in range(self.n_eval):
            # kalman filter:
            ktvode.predict(t)
            if method=="chkrebtii":
                ktvode.interrogate_chkrebtii(t)
            elif method=="schober":
                ktvode.interrogate_schober(t)
            else:
                ktvode.interrogate_rodeo(t)
            self.ode_fun(x_state, self.tmin + (self.tmax-self.tmin)*(t+1)/self.n_eval, theta, x_meas)
            ktvode.update(t)
        
        return ktvode

    cpdef solve(self, double[::1] x0, double[::1, :] W=None, object theta=None, method="rodeo"):
        r"""
        Returns a sample solution, a posterior mean and variance of the solution process to ODE problem.

        Args:
            x0 (ndarray(n_state)): Initial value of the state variable :math:`x_n` at time :math:`t = 0`; :math:`x_0`.
            W (ndarray(n_var, n_state)): Transition matrix defining the measure prior; :math:`W`.
            theta (ndarray(n_theta)): Parameter in the ODE function.
            method (string): Interrogation method.
        
        Returns:
            (tuple):
            - **x_state_smooth** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [0...T].
            - **mu_state_smooth** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`y_n` at times
              [0...T].
            - **var_state_smooth** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at
              times [0...T].

        """
        cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooth = np.empty((self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooth = np.empty((self.n_state, self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooth = np.empty((self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        if W is not None:
            self._wgt_meas[:] = W

        ktvode = self._solve_filter(x_state_smooth, mu_state_smooth, var_state_smooth, x0, theta, method)
        
        # backward pass
        ktvode.smooth_update(True, True)
        del ktvode
        
        return x_state_smooth.T, mu_state_smooth.T, var_state_smooth.T
    
    cpdef solve_sim(self, double[::1] x0, double[::1, :] W=None, object theta=None, method="rodeo"):
        r"""
        Only returns a sample solution from :func:`~KalmanODE.KalmanODE.solve`.
        """

        cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooth = np.empty((self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooth = np.empty((self.n_state, self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooth = np.empty((self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')

        if W is not None:
            self._wgt_meas[:] = W

        ktvode = self._solve_filter(x_state_smooth, mu_state_smooth, var_state_smooth, x0, theta, method)
        
        # backward pass
        ktvode.smooth_update(False, True)
        del ktvode
        
        return x_state_smooth.T
    
    cpdef solve_mv(self, double[::1] x0, double[::1, :] W=None, object theta=None, method="rodeo"):
        r"""
        Only returns the mean and variance from :func:`~KalmanODE.KalmanODE.solve`.
        """

        cdef np.ndarray[DTYPE_t, ndim=2] mu_state_smooth = np.empty((self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=3] var_state_smooth = np.empty((self.n_state, self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        cdef np.ndarray[DTYPE_t, ndim=2] x_state_smooth = np.empty((self.n_state, self.n_steps),
                                                                    dtype=DTYPE, order='F')
        if W is not None:
            self._wgt_meas[:] = W

        ktvode = self._solve_filter(x_state_smooth, mu_state_smooth, var_state_smooth, x0, theta, method)
        
        # backward pass
        ktvode.smooth_update(True, False)
        del ktvode
        
        return mu_state_smooth.T, var_state_smooth.T
