# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False
cimport cython
import numpy as np
cimport numpy as np

from kalmantv.kalmantv cimport KalmanTV, state_sim
from kalmantv.blas cimport mat_triple_mult, vec_copy, mat_copy
from probDE.utils import rand_mat

DTYPE = np.double

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
        z_states (ndarray(n_state, 2*n_steps)): Random N(0,1) matrix for forecasting and smoothing.

    """
    cdef int n_state, n_meas, n_eval, n_steps
    cdef double tmin, tmax
    cdef object fun
    cdef double[::1, :] __wgt_state
    cdef double[::1] __mu_state
    cdef double[::1, :] __var_state
    cdef double[::1, :] __z_states

    # malloc variables
    cdef double[::1, :] mu_state_preds
    cdef double[::1, :, :] var_state_preds 
    cdef double[::1, :] mu_state_filts 
    cdef double[::1, :, :] var_state_filts 
    cdef double[::1, :] mu_state_smooths 
    cdef double[::1, :, :] var_state_smooths 
    cdef double[::1, :] x_state_smooths 
    cdef double[::1] x_state 
    cdef double[::1] x_meas 
    cdef double[::1] mu_meas
    cdef double[::1, :] var_meas

    # temp variables
    cdef double[::1, :] llt_state 
    cdef double[::1, :] twgt_meas 
        
    def __cinit__(self, int n_state, int n_meas, double tmin, double tmax, int n_eval, object fun, **init):
        self.n_state = n_state
        self.n_meas = n_meas
        self.tmin = tmin
        self.tmax = tmax
        self.n_eval = n_eval
        self.n_steps = n_eval + 1
        self.fun = fun
        self.__wgt_state = None
        self.__mu_state = None
        self.__var_state = None
        self.__z_states = None
        for key in init.keys():
            self.__setattr__(key, init[key])
        
        # malloc variables
        self.mu_state_preds = np.zeros((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.var_state_preds = np.zeros((self.n_state, self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.mu_state_filts = np.zeros((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.var_state_filts = np.zeros((self.n_state, self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.mu_state_smooths = np.zeros((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.var_state_smooths = np.zeros((self.n_state, self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.x_state_smooths = np.zeros((self.n_state, self.n_steps), dtype=DTYPE, order='F')
        self.x_state = np.empty(self.n_state, dtype=DTYPE, order='F')
        self.x_meas = np.empty(self.n_meas, dtype=DTYPE, order='F')
        self.mu_meas = np.zeros(self.n_meas, dtype=DTYPE, order='F')
        self.var_meas = np.zeros((self.n_meas, self.n_meas), dtype=DTYPE, order='F')
        self.llt_state = np.empty((self.n_state, self.n_state), dtype=DTYPE, order='F')
        self.twgt_meas = np.empty((self.n_meas, self.n_state), dtype=DTYPE, order='F')
    
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
    def z_states(self):
        return self.__z_states

    @z_states.setter
    def z_states(self, z_states):
        self.__z_states = z_states.copy('F')
    
    @z_states.deleter
    def z_states(self):
        self.__z_states = None
    
    cpdef kalman_ode(self, 
                     double[::1] x0_state,
                     double[::1, :] wgt_meas,
                     double[::1, :] x_meass,
                     object theta,
                     bint smooth_mv,
                     bint smooth_sim,
                     bint offline):
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
            x0_state (ndarray(n_state)): Initial value of the state variable :math:`x_n` at time :math:`n = 0`; :math:`x_0`.
            wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
            x_meass (ndarray(n_state, n_steps)): Optional offline observations.
            theta (ndarray(n_theta)): Parameter in the ODE function.
            smooth_mv (bool): Flag for returning the smoothed mean and variance.
            smooth_sim (bool): Flag for returning the smoothed simulated state.
            offline (bool): Flag for offline forecasted `x_meas`. 

        """
        # temp variables
        cdef int t
        
        # initialize variables
        self.mu_meas = np.zeros(self.n_meas, dtype=DTYPE, order='F')
        self.var_meas = np.zeros((self.n_meas, self.n_meas), dtype=DTYPE, order='F')
        vec_copy(self.mu_state_filts[:, 0], x0_state)
        vec_copy(self.mu_state_preds[:, 0], self.mu_state_filts[:, 0])
        vec_copy(self.mu_state_smooths[:, 0], self.mu_state_filts[:, 0])
        vec_copy(self.x_state_smooths[:, 0], x0_state)
        
        # initialize class
        ktv = KalmanTV(self.n_meas, self.n_state)
        
        #forward pass
        for t in range(self.n_eval):
            if not offline:
                # kalman filter:
                ktv.predict(self.mu_state_preds[:, t+1],
                            self.var_state_preds[:, :, t+1],
                            self.mu_state_filts[:, t],
                            self.var_state_filts[:, :, t],
                            self.__mu_state,
                            self.__wgt_state,
                            self.__var_state)
                forecast(self.x_state,
                        self.var_meas,
                        self.twgt_meas,
                        self.llt_state,
                        wgt_meas,
                        self.mu_state_preds[:, t+1],
                        self.var_state_preds[:, :, t+1],
                        self.__z_states[:, t])
                #forecast_sch(x_state,
                #             var_meas,
                #             twgt_meas,
                #             wgt_meas,
                #             mu_state_preds[:, t+1],
                #             var_state_preds[:, :, t+1])
                self.fun(self.x_state, self.tmin + (self.tmax-self.tmin)*(t+1)/self.n_eval, theta, self.x_meas)
                ktv.update(self.mu_state_filts[:, t+1],
                        self.var_state_filts[:, :, t+1],
                        self.mu_state_preds[:, t+1],
                        self.var_state_preds[:, :, t+1],
                        self.x_meas,
                        self.mu_meas,
                        wgt_meas,
                        self.var_meas)
            else:
                ktv.filter(self.mu_state_preds[:, t+1],
                           self.var_state_preds[:, :, t+1],
                           self.mu_state_filts[:, t+1],
                           self.var_state_filts[:, :, t+1],
                           self.mu_state_filts[:, t],
                           self.var_state_filts[:, :, t],
                           self.__mu_state,
                           self.__wgt_state,
                           self.__var_state,
                           x_meass[:, t+1],
                           self.mu_meas,
                           wgt_meas,
                           self.var_meas)

        vec_copy(self.mu_state_smooths[:, self.n_eval], self.mu_state_filts[:, self.n_eval])
        mat_copy(self.var_state_smooths[:, :, self.n_eval], self.var_state_filts[:, :, self.n_eval])
        state_sim(self.x_state_smooths[:, self.n_eval],
                self.llt_state,
                self.mu_state_smooths[:, self.n_eval],
                self.var_state_smooths[:, :, self.n_eval],
                self.__z_states[:, self.n_eval])
        # backward pass
        for t in reversed(range(1, self.n_eval)):
            if smooth_mv and smooth_sim:
                ktv.smooth(self.x_state_smooths[:, t],
                           self.mu_state_smooths[:, t],
                           self.var_state_smooths[:, :, t],
                           self.x_state_smooths[:, t+1],
                           self.mu_state_smooths[:, t+1],
                           self.var_state_smooths[:, :, t+1],
                           self.mu_state_filts[:, t],
                           self.var_state_filts[:, :, t],
                           self.mu_state_preds[:, t+1],
                           self.var_state_preds[:, :, t+1],
                           self.__wgt_state,
                           self.__z_states[:, t+self.n_steps])
            elif smooth_mv:
                ktv.smooth_mv(self.mu_state_smooths[:, t],
                              self.var_state_smooths[:, :, t],
                              self.mu_state_smooths[:, t+1],
                              self.var_state_smooths[:, :, t+1],
                              self.mu_state_filts[:, t],
                              self.var_state_filts[:, :, t],
                              self.mu_state_preds[:, t+1],
                              self.var_state_preds[:, :, t+1],
                              self.__wgt_state)
            elif smooth_sim:
                ktv.smooth_sim(self.x_state_smooths[:, t],
                               self.x_state_smooths[:, t+1],
                               self.mu_state_filts[:, t],
                               self.var_state_filts[:, :, t],
                               self.mu_state_preds[:, t+1],
                               self.var_state_preds[:, :, t+1],
                               self.__wgt_state,
                               self.__z_states[:, t+self.n_steps])
        del ktv


    cpdef solve(self, double[::1] x0_state, double[::1, :] wgt_meas, theta=None, bint mv=False, bint sim=True):
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
        if (self.__wgt_state is None or self.__mu_state is None or 
           self.__var_state is None):
            raise ValueError("wgt_state, mu_state, and/or var_state is not set.")
        
        if self.__z_states is None:
            self.__z_states = rand_mat(2*(self.n_eval+1), self.n_state)
        
        self.kalman_ode(x0_state, wgt_meas, None, theta, mv, sim, False)
        if mv and sim:
            kalman_sim = np.ascontiguousarray(self.x_state_smooths.T)
            kalman_mu = np.ascontiguousarray(self.mu_state_smooths.T)
            kalman_var = np.ascontiguousarray(self.var_state_smooths.T)
            return kalman_sim, kalman_mu, kalman_var
        elif mv:
            kalman_mu = np.ascontiguousarray(self.mu_state_smooths.T)
            kalman_var = np.ascontiguousarray(self.var_state_smooths.T)
            return kalman_mu, kalman_var
        elif sim:
            kalman_sim = np.ascontiguousarray(self.x_state_smooths.T)
            return kalman_sim
            