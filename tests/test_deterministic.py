import unittest
import numpy as np
from scipy import integrate

from rodeo.ibm import ibm_init
from rodeo.cython.KalmanODE import KalmanODE
from rodeo.utils.utils import rand_mat, indep_init, zero_pad
from utils import *

class KalmanTVODETest(unittest.TestCase):
    def test_chkrebtii(self):
        # LHS vector of ODE
        w_mat = np.array([[0.0, 0.0, 1.0]])

        # These parameters define the order of the ODE and the CAR(p) process
        n_deriv = [2]
        n_deriv_prior = [4]

        # it is assumed that the solution is sought on the interval [tmin, tmax].
        n_eval = 300
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

        # Initialize the Kalman class
        kalmanode = KalmanODE(W, tmin, tmax, n_eval, chkrebtii_kalman, **kinit)
        # Run the solver to get an approximation
        kalman_sim = kalmanode.solve_sim(x0_state)

        # Get deterministic solution from odeint
        tseq = np.linspace(tmin, tmax, n_eval+1)
        detode = integrate.odeint(chkrebtii_odeint, [-1, 0], tseq)
        self.assertLessEqual(rel_err(kalman_sim[:, 0], detode[:, 0]), 10.0)
        self.assertLessEqual(rel_err(kalman_sim[1:, 1], detode[1:, 1]), 10.0)

if __name__ == '__main__':
    unittest.main()
