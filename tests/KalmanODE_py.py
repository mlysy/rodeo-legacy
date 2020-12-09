import numpy as np

from probDE.utils.utils import rand_mat
from depreciated.kalman.kalman_ode_higher import kalman_ode_higher


class KalmanODE_py:
    def __init__(self, W, tmin, tmax, n_eval, fun, **init):
        self.n_state = W.shape[1]
        self.n_meas = W.shape[0]
        self.tmin = tmin
        self.tmax = tmax
        self.n_eval = n_eval
        self.fun = fun
        self.wgt_meas = W
        self.wgt_state = None
        self.mu_state = None
        self.var_state = None
        self.z_state = None
        for key in init.keys():
            self.__setattr__(key, init[key])

    def solve_sim(self, x0_state, wgt_meas=None, theta=None, mv=False, sim=True):
        if (self.wgt_state is None or self.mu_state is None or
                self.var_state is None):
            raise ValueError("wgt_state, mu_state, var_state is not set.")

        if self.z_state is None:
            self.z_state = rand_mat(2*(self.n_eval+1), self.n_state)

        return kalman_ode_higher(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                                 self.wgt_state, self.mu_state, self.var_state, wgt_meas,
                                 self.z_state, theta, mv, sim)

    def solve_mv(self, x0_state, wgt_meas=None, theta=None, mv=True, sim=False):
        if (self.wgt_state is None or self.mu_state is None or
                self.var_state is None):
            raise ValueError("wgt_state, mu_state, var_state is not set.")

        if self.z_state is None:
            self.z_state = rand_mat(2*(self.n_eval+1), self.n_state)

        return kalman_ode_higher(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                                 self.wgt_state, self.mu_state, self.var_state, wgt_meas,
                                 self.z_state, theta, mv, sim)

    def solve(self, x0_state, wgt_meas=None, theta=None, mv=True, sim=True):
        if (self.wgt_state is None or self.mu_state is None or
                self.var_state is None):
            raise ValueError("wgt_state, mu_state, var_state is not set.")

        if self.z_state is None:
            self.z_state = rand_mat(2*(self.n_eval+1), self.n_state)

        return kalman_ode_higher(self.fun, x0_state, self.tmin, self.tmax, self.n_eval,
                                 self.wgt_state, self.mu_state, self.var_state, wgt_meas,
                                 self.z_state, theta, mv, sim)
