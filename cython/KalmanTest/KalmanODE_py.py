import numpy as np

from probDE.utils.utils import zero_pad
from probDE.utils.utils import root_gen
from probDE.Kalman.kalman_initial_draw import kalman_initial_draw
from probDE.Kalman import kalman_ode_higher
from probDE.Kalman.higher_mvncond import higher_mvncond
from probDE.Kalman.multi_mvncond import multi_mvncond
from probDE.utils.utils import rand_mat

class KalmanODE_py:
    def __init__(self, n_state, n_meas, tmin, tmax, n_eval, fun):
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
    
    # def initialize(self, kinit):
    #     wgt_meas, wgt_state, var_state, _ = kinit
    #     mu_state = np.zeros(self.n_state)
    #     z_states = rand_mat(2*(self.n_eval+1), self.n_state)
    #     self.wgt_state = wgt_state
    #     self.mu_state = mu_state
    #     self.var_state = var_state
    #     self.wgt_meas = wgt_meas
    #     self.z_states = z_states
    #     return
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

    def solve(self, x0_state, theta=None, mv=False, sim=True):
        if (self.wgt_state is None or self.mu_state is None or 
            self.var_state is None or self.wgt_meas is None):
            raise ValueError("wgt_state, mu_state, var_state, wgt_meas is not set.")
            
        if self.z_states is None:
            self.z_states = rand_mat(2*(self.n_eval+1), self.n_state)
            
        return kalman_ode_higher(self.fun, x0_state, self.tmin, self.tmax, self.n_eval, 
                                 self.wgt_state, self.mu_state, self.var_state, self.wgt_meas,
                                 self.z_states, theta, mv, sim)
                                 