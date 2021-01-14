import unittest
import numpy as np

from rodeo.ibm import ibm_init
from rodeo.cython.KalmanODE import KalmanODE as KalmanODE_cy
from rodeo.eigen.KalmanODE import KalmanODE as KalmanODE_ei
from rodeo.numba.KalmanODE import KalmanODE as KalmanODE_nb
from KalmanODE_py import KalmanODE_py
from rodeo.utils.utils import rand_mat, indep_init, zero_pad
from utils import *

class KalmanODETest(unittest.TestCase):
    def test_chkrebtii(self):
        # LHS vector of ODE
        w_mat = np.array([[0.0, 0.0, 1.0]])

        # These parameters define the order of the ODE and the CAR(p) process
        n_deriv = [2]
        n_deriv_prior = [4]

        # it is assumed that the solution is sought on the interval [tmin, tmax].
        n_eval = 100
        tmin = 0
        tmax = 10

        # IBM process scale factor
        sigma = [.5]

        # Initial value, x0, for the IVP
        x0 = np.array([-1., 0., 1.])
        x0_state = zero_pad(x0, n_deriv, n_deriv_prior)
        W = zero_pad(w_mat, n_deriv, n_deriv_prior)

        # Get parameters needed to run the solver
        dt = (tmax-tmin)/n_eval
        # All necessary parameters are in kinit, namely, T, c, R, W
        kinit = ibm_init(dt, n_deriv_prior, sigma)
        z_state = rand_mat(n_eval, sum(n_deriv_prior))

        # Get Cython solution
        kode_cy = KalmanODE_cy(W, tmin, tmax, n_eval, chkrebtii_kalman, **kinit)
        kode_cy.z_state = z_state
        # Run the solver to get an approximation
        ksim_cy = kode_cy.solve_sim(x0_state)

        # Get Eigen solution
        kode_ei = KalmanODE_ei(W, tmin, tmax, n_eval, chkrebtii_kalman, **kinit)
        kode_ei.z_state = z_state
        ksim_ei = kode_ei.solve_sim(x0_state)

        # Get Numba solution
        kode_nb = KalmanODE_nb(W, tmin, tmax, n_eval, chkrebtii_kalman_nb, **kinit)
        kode_nb.z_state = z_state
        ksim_nb = kode_nb.solve_sim(x0_state, W, None)

        # Get Python solution
        kalmanode_py = KalmanODE_py(W, tmin, tmax, n_eval, chkrebtii_kalman, **kinit) # Initialize the class
        kalmanode_py.z_state = z_state
        ksim_py = kalmanode_py.solve_sim(x0_state, W)

        self.assertLessEqual(rel_err(ksim_cy[:, 0], ksim_py[:, 0]), 0.001)
        self.assertLessEqual(rel_err(ksim_cy[1:, 1], ksim_py[1:, 1]), 0.001)
        self.assertLessEqual(rel_err(ksim_cy[:, 0], ksim_ei[:, 0]), 0.001)
        self.assertLessEqual(rel_err(ksim_cy[1:, 1], ksim_ei[1:, 1]), 0.001)
        self.assertLessEqual(rel_err(ksim_cy[:, 0], ksim_nb[:, 0]), 0.001)
        self.assertLessEqual(rel_err(ksim_cy[1:, 1], ksim_nb[1:, 1]), 0.001)

if __name__ == '__main__':
    unittest.main()

