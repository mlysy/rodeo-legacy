import numpy as np
cimport numpy as np
from probDE.cython.KalmanTest.kalman_ode import kalman_ode

cdef class KalmanODE:
    cdef int n_state, n_meas, n_eval
    cdef double tmin, tmax
    cdef object fun
    cdef public:
        object wgt_state
        object mu_state
        object var_state
        object wgt_meas
        object z_states
        
    def __cinit__(self, n_state, n_meas, tmin, tmax, n_eval, fun):
        self.n_state = n_state
        self.n_meas = n_meas
        self.tmin = tmin
        self.tmax = tmax
        self.n_eval = n_eval
        self.fun = fun.odefun
        self.wgt_state = None
        self.mu_state = None
        self.var_state = None
        self.wgt_meas = None
        self.z_states = None
    
    cpdef rand_mat(self, int n, int p, bint pd=True):
        cdef np.ndarray[np.double_t, ndim=2] V = np.zeros((p, n), order='F')
        V[:] = np.random.randn(p, n)
        if (p == n) and pd:
            V[:] = np.matmul(V, V.T)
        return V
        
    cpdef solve(self, double[::1] x0_state, theta=None, bint mv=False, bint sim=True):
        if (self.wgt_state is None or self.mu_state is None or 
           self.var_state is None or self.wgt_meas is None):
            raise ValueError("wgt_state, mu_state, var_state, wgt_meas is not set.")
        
        if self.z_states is None:
            self.z_states = self.rand_mat(2*self.n_eval, self.n_state)
        
        if mv and sim:
            kalman_sim, kalman_mu, kalman_var = \
                kalman_ode(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                          self.wgt_state, self.mu_state, self.var_state,
                          self.wgt_meas, self.z_states, None, mv, sim)
            kalman_sim = np.ascontiguousarray(kalman_sim.T)
            kalman_mu = np.ascontiguousarray(kalman_mu.T)
            kalman_var = np.ascontiguousarray(kalman_var.T)
            return kalman_sim, kalman_mu, kalman_var
        elif mv:
            kalman_mu, kalman_var = \
                kalman_ode(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                          self.wgt_state, self.mu_state, self.var_state,
                          self.wgt_meas, self.z_states, None, mv, sim)
            kalman_mu = np.ascontiguousarray(kalman_mu.T)
            kalman_var = np.ascontiguousarray(kalman_var.T)
            return kalman_mu, kalman_var
        elif sim:
            kalman_sim = \
                kalman_ode(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                          self.wgt_state, self.mu_state, self.var_state,
                          self.wgt_meas, self.z_states, None, mv, sim)
            kalman_sim = np.ascontiguousarray(kalman_sim.T)
            return kalman_sim