# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
cimport cython
import numpy as np
cimport numpy as np

from kalmantv.cython.kalmantv cimport KalmanTV, state_sim
from kalmantv.cython.blas cimport mat_triple_mult, vec_copy, mat_copy
from probDE.utils import rand_mat

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cpdef forecast(double[::1] x_state,
               double[::1, :] var_meas,
               double[::1, :] twgt_meas,
               double[::1, :] llt_state,
               const double[::1, :] wgt_meas,
               const double[::1] mu_state_pred,
               const double[::1, :] var_state_pred,
               const double[::1] z_state):
    r"""
    Forecast the observed state from the current state.
    
    Args:
        x_state (ndarray(n_state)): Simulated state.
        var_meas (ndarray(n_meas, n_meas)): Variance of simulated measure.
        twgt_meas (ndarray(n_meas, n_state)): Temporary matrix to store intermediate operation.
        llt_state (ndarray(n_state, n_state)): Temporary matrix to store cholesky factorization.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.
    
    Returns:
        (tuple):
        - **x_state** (ndarray(n_state)): Simulated state.
        - **var_meas** (ndarray(n_meas, n_meas)): Variance of simulated measure.
        - **twgt_meas** (ndarray(n_meas, n_state)): Temporary matrix to store intermediate operation.
        - **llt_state** (ndarray(n_state, n_state)): Temporary matrix to store cholesky factorization.
    
    """
    cdef char* wgt_trans = 'N'
    cdef char* wgt_trans2 = 'T'
    cdef char* var_trans = 'N'
    cdef int var_alpha = 1, var_beta = 0
    mat_triple_mult(var_meas, twgt_meas, wgt_trans, var_trans, wgt_trans2,
                    var_alpha, var_beta, wgt_meas, var_state_pred, wgt_meas)
    state_sim(x_state, llt_state, mu_state_pred, var_state_pred, z_state)
    return

cpdef forecast_sch(double[::1] x_state,
                   double[::1, :] var_meas,
                   double[::1, :] twgt_meas,
                   const double[::1, :] wgt_meas,
                   const double[::1] mu_state_pred,
                   const double[::1, :] var_state_pred):
    r"""
    Forecast the observed state from the current state via the Schobert method.
    
    Args:
        x_state (ndarray(n_state)): Simulated state.
        var_meas (ndarray(n_meas, n_meas)): Variance of simulated measure.
        twgt_meas (ndarray(n_meas, n_state)): Temporary matrix to store intermediate operation.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from 
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given 
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
    
    Returns:
        (tuple):
        - **x_state** (ndarray(n_state)): Simulated state.
        - **var_meas** (ndarray(n_meas, n_meas)): Variance of simulated measure.
        - **twgt_meas** (ndarray(n_meas, n_state)): Temporary matrix to store intermediate operation.
    
    """
    cdef char* wgt_trans = 'N'
    cdef char* wgt_trans2 = 'T'
    cdef char* var_trans = 'N'
    cdef int var_alpha = 1, var_beta = 0
    mat_triple_mult(var_meas, twgt_meas, wgt_trans, var_trans, wgt_trans2,
                    var_alpha, var_beta, wgt_meas, var_state_pred, wgt_meas)
    vec_copy(x_state, mu_state_pred)
    return

cpdef _copynm(dest, source, name):
    if not source.data.f_contiguous:
        raise TypeError('{} is not f contiguous.'.format(name))
    if not source.shape==dest.shape:
        raise TypeError('{} has incorrect shape.'.format(name))
    dest[:] = source
    return source

cdef class KalmanODE:
    r"""
    Create a Kalman Time-Varying object. The methods of the object can predict, update, sample and 
    smooth the mean and variance of the Kalman Filter. This method is useful if one wants to track 
    an object with streaming observations.

    Args:
        n_meas (int): Size of the measure.
        n_state (int): Size of the state.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_eval (int): Number of discretization points (:math:`N`) of the time interval that is evaluated,
            such that discretization timestep is :math:`dt = b/N`.
        mu_state (ndarray(n_state)): Transition_offsets defining the solution prior; denoted by :math:`c`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`T`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R`.
        mu_meas (ndarray(n_meas)): Transition_offsets defining the measure prior; denoted by :math:`d`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`H`.
        z_state (ndarray(n_state, 2*n_steps)): Random N(0,1) matrix for forecasting and smoothing.

    """
    cdef int n_state, n_meas, n_eval, n_steps
    cdef double tmin, tmax
    cdef object fun
    cdef double[::1, :] _wgt_state
    cdef double[::1] _mu_state
    cdef double[::1, :] _var_state
    cdef double[::1, :] _z_state
    cdef KalmanTV ktv

    # malloc variables
    cdef double[::1, :] mu_state_pred
    cdef double[::1, :, :] var_state_pred 
    cdef double[::1, :] mu_state_filt
    cdef double[::1, :, :] var_state_filt 
    cdef double[::1, :] mu_state_smooth
    cdef double[::1, :, :] var_state_smooth
    cdef double[::1, :] x_state_smooth
    cdef double[::1] x_state 
    cdef double[::1] x_meas 
    cdef double[::1] mu_meas
    cdef double[::1, :] var_meas

    # temp variables
    cdef double[::1, :] llt_state 
    cdef double[::1, :] twgt_meas 
        
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

        # malloc variables
        self.mu_state_pred = np.empty((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.var_state_pred = np.empty((self.n_state, self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.mu_state_filt = np.empty((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.var_state_filt = np.empty((self.n_state, self.n_state, self.n_steps), dtype=DTYPE, order='F')
        #self.mu_state_smooth = np.empty((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        #self.var_state_smooth = np.empty((self.n_state, self.n_state, self.n_steps), dtype=DTYPE, order='F')
        #self.x_state_smooth = np.empty((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.x_state = np.empty(self.n_state, dtype=DTYPE, order='F')
        self.x_meas = np.empty(self.n_meas, dtype=DTYPE, order='F')
        self.mu_meas = np.zeros(self.n_meas, dtype=DTYPE, order='F')
        self.var_meas = np.empty((self.n_meas, self.n_meas), dtype=DTYPE, order='F')
        self.llt_state = np.empty((self.n_state, self.n_state), dtype=DTYPE, order='F')
        self.twgt_meas = np.empty((self.n_meas, self.n_state), dtype=DTYPE, order='F')

        # initialize class
        self.ktv = KalmanTV(self.n_meas, self.n_state)
    
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

    cpdef _solve_filter(self, double[::1] x0_state, double[::1, :] wgt_meas, object theta=None):
        r"""
        Forward pass filter step in the KalmanODE solver. 

        Args:
            x0_state (ndarray(n_state)): Initial value of the state variable :math:`x_n` at time :math:`t = 0`; :math:`x_0`.
            wgt_meas (ndarray(n_var, n_state)): Transition matrix defining the measure prior; :math:`W`.
            theta (ndarray(n_theta)): Parameter in the ODE function.
        
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

        vec_copy(self.mu_state_filt[:, 0], x0_state)
        vec_copy(self.mu_state_pred[:, 0], self.mu_state_filt[:, 0])
        self.var_state_filt[:, :, 0] = 0

        #forward pass
        for t in range(self.n_eval):
            # kalman filter:
            self.ktv.predict(self.mu_state_pred[:, t+1],
                             self.var_state_pred[:, :, t+1],
                             self.mu_state_filt[:, t],
                             self.var_state_filt[:, :, t],
                             self._mu_state,
                             self._wgt_state,
                             self._var_state)
            forecast(self.x_state,
                    self.var_meas,
                    self.twgt_meas,
                    self.llt_state,
                    wgt_meas,
                    self.mu_state_pred[:, t+1],
                    self.var_state_pred[:, :, t+1],
                    self._z_state[:, t])
            #forecast_sch(x_state,
            #             var_meas,
            #             twgt_meas,
            #             wgt_meas,
            #             mu_state_pred[:, t+1],
            #             var_state_pred[:, :, t+1])
            self.fun(self.x_state, self.tmin + (self.tmax-self.tmin)*(t+1)/self.n_eval, theta, self.x_meas)
            self.ktv.update(self.mu_state_filt[:, t+1],
                            self.var_state_filt[:, :, t+1],
                            self.mu_state_pred[:, t+1],
                            self.var_state_pred[:, :, t+1],
                            self.x_meas,
                            self.mu_meas,
                            wgt_meas,
                            self.var_meas)
        return

    cpdef solve(self, double[::1] x0_state, double[::1, :] wgt_meas, object theta=None):
        r"""
        Probabilistic ODE solver based on the Kalman filter and smoother. Returns an approximate solution to the higher order ODE

        .. math:: W x_n = F(x_n, n, \theta)

        on the time interval :math:`n \in [a, b]` with initial condition :math:`x_0 = x_0`. The corresponding variable names are

        The specific model we are using to approximate the solution :math:`x_n` is

        .. math::

            X_n = c + T X_{n-1} + R_n^{1/2} \epsilon_n

            y_n = d + W X_n + H_n^{1/2} \eta_n

        where :math:`\epsilon_n` and :math:`\eta_n` are independent :math:`N(0,1)` distributions and
        :math:`X_n = (x_n, y_n)` at time n and :math:`y_n` denotes the observation at time n.

        Args:
            x0_state (ndarray(n_state)): Initial value of the state variable :math:`x_n` at time :math:`t = 0`; :math:`x_0`.
            wgt_meas (ndarray(n_var, n_state)): Transition matrix defining the measure prior; :math:`W`.
            theta (ndarray(n_theta)): Parameter in the ODE function.
            sim_sol (bool): Flag for returning the smoothed simulated state.
        
        Returns:
            (tuple):
            - **kalman_sim** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [a...b].
            - **kalman_mu** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`y_n` at times
              [a...b].
            - **kalman_var** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at
              times [a...b].

        """
        cdef double[::1, :] mu_state_smooth
        cdef double[::1, :, :] var_state_smooth
        cdef double[::1, :] x_state_smooth
        
        mu_state_smooth = np.empty((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        var_state_smooth = np.empty((self.n_state, self.n_state, self.n_steps), dtype=DTYPE, order='F')
        x_state_smooth = np.empty((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        
        self._solve_filter(x0_state, wgt_meas, theta)

        vec_copy(mu_state_smooth[:, 0], self.mu_state_filt[:, 0])
        vec_copy(x_state_smooth[:, 0], x0_state)
        vec_copy(mu_state_smooth[:, self.n_eval], self.mu_state_filt[:, self.n_eval])
        mat_copy(var_state_smooth[:, :, self.n_eval], self.var_state_filt[:, :, self.n_eval])
        state_sim(x_state_smooth[:, self.n_eval],
                  self.llt_state,
                  mu_state_smooth[:, self.n_eval],
                  var_state_smooth[:, :, self.n_eval],
                  self._z_state[:, self.n_eval])
        # backward pass
        for t in range(self.n_eval-1, 0, -1):
            self.ktv.smooth(x_state_smooth[:, t],
                            mu_state_smooth[:, t],
                            var_state_smooth[:, :, t],
                            x_state_smooth[:, t+1],
                            mu_state_smooth[:, t+1],
                            var_state_smooth[:, :, t+1],
                            self.mu_state_filt[:, t],
                            self.var_state_filt[:, :, t],
                            self.mu_state_pred[:, t+1],
                            self.var_state_pred[:, :, t+1],
                            self._wgt_state,
                            self._z_state[:, t+self.n_steps])
            
        return np.asarray(x_state_smooth.T), np.asarray(mu_state_smooth.T), np.asarray(var_state_smooth.T)

    cpdef solve_sim(self, double[::1] x0_state, double[::1, :] wgt_meas, object theta=None):
        r"""
        Only returns simulate solutions from :func:`~KalmanODE.KalmanODE.solve`.
        """

        cdef double[::1, :] x_state_smooth
        x_state_smooth = np.empty((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        
        self._solve_filter(x0_state, wgt_meas, theta)

        vec_copy(x_state_smooth[:, 0], x0_state)
        state_sim(x_state_smooth[:, self.n_eval],
                  self.llt_state,
                  self.mu_state_filt[:, self.n_eval],
                  self.var_state_filt[:, :, self.n_eval],
                  self._z_state[:, self.n_eval])
        # backward pass
        for t in range(self.n_eval-1, 0, -1):
            self.ktv.smooth_sim(x_state_smooth[:, t],
                                x_state_smooth[:, t+1],
                                self.mu_state_filt[:, t],
                                self.var_state_filt[:, :, t],
                                self.mu_state_pred[:, t+1],
                                self.var_state_pred[:, :, t+1],
                                self._wgt_state,
                                self._z_state[:, t+self.n_steps])
        
        return np.asarray(x_state_smooth.T)
    
    cpdef solve_mv(self, double[::1] x0_state, double[::1, :] wgt_meas, object theta=None):
        r"""
        Only returns mean and variance from :func:`~KalmanODE.KalmanODE.solve`.
        """

        cdef double[::1, :] mu_state_smooth
        cdef double[::1, :, :] var_state_smooth
        mu_state_smooth = np.empty((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        var_state_smooth = np.empty((self.n_state, self.n_state, self.n_steps), dtype=DTYPE, order='F')
        
        self._solve_filter(x0_state, wgt_meas, theta)

        vec_copy(mu_state_smooth[:, 0], self.mu_state_filt[:, 0])
        vec_copy(mu_state_smooth[:, self.n_eval], self.mu_state_filt[:, self.n_eval])
        mat_copy(var_state_smooth[:, :, self.n_eval], self.var_state_filt[:, :, self.n_eval])
        # backward pass
        for t in range(self.n_eval-1, 0, -1):
            self.ktv.smooth_mv(mu_state_smooth[:, t],
                               var_state_smooth[:, :, t],
                               mu_state_smooth[:, t+1],
                               var_state_smooth[:, :, t+1],
                               self.mu_state_filt[:, t],
                               self.var_state_filt[:, :, t],
                               self.mu_state_pred[:, t+1],
                               self.var_state_pred[:, :, t+1],
                               self._wgt_state)
        
        return np.asarray(mu_state_smooth.T), np.asarray(var_state_smooth.T)
