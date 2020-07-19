import unittest
import numpy as np
from math import sin
from scipy import integrate

from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils.utils import rand_mat, indep_init

def sum_err(X1, X2):
    """Sum of relative error between two numpy arrays."""
    return np.sum(np.abs((X1.ravel() - X2.ravel())/X1.ravel()))

def chkrebtii_kalman(x_t, t, theta=None):
    """Chkrebtii function in kalman format."""
    return np.array([sin(2*t) - x_t[0]])

def chkrebtii_odeint(x_t, t):
    """Chkrebtii function in odeint format."""
    return [x_t[1], sin(2*t) - x_t[0]]

class KalmanTVODETest(unittest.TestCase):
    def test_chkrebtii(self):
        # LHS vector of ODE
        w_vec = np.array([0.0, 0.0, 1.0])

        # These parameters define the order of the ODE and the CAR(p) process
        n_meas = 1
        n_state = 4

        # it is assumed that the solution is sought on the interval [tmin, tmax].
        n_eval = 300
        tmin = 0
        tmax = 10

        # The rest of the parameters can be tuned according to ODE
        # For this problem, we will use
        tau = 50
        sigma = .001

        # Initial value, x0, for the IVP
        x0 = np.array([-1., 0., 1.])

        # Get parameters needed to run the solver
        dt = (tmax-tmin)/n_eval
        # All necessary parameters are in kinit, namely, T, c, R, W
        kinit, x0_state = indep_init([car_init(n_state, tau, sigma, dt, w_vec, x0)], n_state)

        # Initialize the Kalman class
        kalmanode = KalmanODE(n_state, n_meas, tmin, tmax, n_eval, chkrebtii_kalman, **kinit)
        # Run the solver to get an approximation
        kalman_sim = kalmanode.solve(x0_state, mv=False, sim=True)

        # Get deterministic solution from odeint
        tseq = np.linspace(tmin, tmax, n_eval+1)
        detode = integrate.odeint(chkrebtii_odeint, [-1, 0], tseq)

        self.assertLessEqual(sum_err(kalman_sim[:, 0], detode[:, 0]), 10.0)
        self.assertLessEqual(sum_err(kalman_sim[1:, 1], detode[1:, 1]), 10.0)

if __name__ == '__main__':
    unittest.main()
