import numpy as np
cimport numpy as np

from probDE.utils.utils import zero_pad
from probDE.utils.utils import root_gen
from probDE.Kalman.kalman_initial_draw import kalman_initial_draw
from probDE.Kalman import kalman_ode_higher
from probDE.Kalman.higher_mvncond import higher_mvncond
from probDE.Kalman.multi_mvncond import multi_mvncond
from probDE.utils import rand_mat
from probDE.cython.KalmanTest.kalman_ode import kalman_ode

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef class KalmanODE:
    cdef int n_state, n_meas, n_eval
    cdef double tmin, tmax
    cdef object fun
    cdef object __wgt_state
    cdef object __mu_state
    cdef object __var_state
    cdef object __wgt_meas
    cdef object __z_states
        
    def __cinit__(self, int n_state, int n_meas, double tmin, double tmax, int n_eval, object fun):
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
        
    @classmethod
    def initialize(cls, kinit, n_state, n_meas, tmin, tmax, n_eval, fun):
        kode = cls(n_state, n_meas, tmin, tmax, n_eval, fun)
        wgt_meas, wgt_state, var_state, _ = kinit
        kode.wgt_meas = wgt_meas
        kode.wgt_state = wgt_state
        kode.var_state = var_state
        kode.mu_state = np.zeros(n_state)
        kode.z_states = rand_mat(2*(n_eval+1), n_state)
        return kode
    
    cpdef solve(self, double[::1] x0_state, theta=None, bint mv=False, bint sim=True):
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
        