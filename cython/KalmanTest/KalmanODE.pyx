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
        self.fun = fun
        self.wgt_state = None
        self.mu_state = None
        self.var_state = None
        self.wgt_meas = None
        self.z_states = None
    
    cpdef np.ndarray[DTYPE_t, ndim=1] initialize(self, double[::1] w, double tau, double sigma, np.ndarray[DTYPE_t, ndim=1] x0):
        cdef np.ndarray[DTYPE_t, ndim=1] delta_t = np.array([(self.tmax - self.tmin)/self.n_eval])
        self.wgt_meas = zero_pad(w, self.n_state)
        cdef np.ndarray[DTYPE_t, ndim=1] roots = root_gen(tau, self.n_state)
        cdef np.ndarray[DTYPE_t, ndim=1] x0_state = kalman_initial_draw(roots, sigma, x0, self.n_state)
        self.wgt_state, self.var_state = higher_mvncond(delta_t, roots, sigma)
        self.mu_state = np.zeros(self.n_state)
        return x0_state
    
    cpdef np.ndarray[DTYPE_t, ndim=1] multi_initialize(self, double[:, ::1] w_mat, double tau, list sigmalst, 
                                                       double scale, np.ndarray[DTYPE_t, ndim=2] x0):
        cdef int n_var, i, w_len
        cdef list rootlst
        n_var = len(x0)
        cdef np.ndarray[DTYPE_t, ndim=1] delta_t = np.array([(self.tmax - self.tmin)/self.n_eval])
        cdef np.ndarray[DTYPE_t, ndim=1] roots = root_gen(tau, self.n_state)
        rootlst = [roots*scale]*n_var
        cdef np.ndarray[DTYPE_t, ndim=2] W_mat = np.zeros((n_var, n_var*self.n_state), order='F')
        cdef np.ndarray[DTYPE_t, ndim=1] x0_state = np.zeros(n_var*self.n_state)
        for i in range(n_var):
            w_len = len(w_mat[i])
            W_mat[i, self.n_state*i : self.n_state*i + w_len] = w_mat[i]
            x0_state[self.n_state*i : self.n_state*(i+1)] = kalman_initial_draw(rootlst[i], sigmalst[i], x0[i], self.n_state)
        self.wgt_meas = W_mat
        self.wgt_state, self.var_state = multi_mvncond(delta_t, rootlst, sigmalst)
        self.mu_state = np.zeros(self.n_state*n_var)
        return x0_state

    cpdef solve(self, double[::1] x0_state, theta=None, bint mv=False, bint sim=True):
        if (self.wgt_state is None or self.mu_state is None or 
           self.var_state is None or self.wgt_meas is None):
            raise ValueError("wgt_state, mu_state, var_state, wgt_meas is not set.")
        
        if self.z_states is None:
            self.z_states = rand_mat(2*(self.n_eval+1), self.n_state)
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
