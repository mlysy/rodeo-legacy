import unittest
import numpy as np
from math import sin
from scipy import integrate

from probDE.ibm import ibm_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils.utils import rand_mat, indep_init, zero_pad

def sum_err(X1, X2):
    """Sum of relative error between two numpy arrays."""
    return np.sum(np.abs((X1.ravel() - X2.ravel())/X1.ravel()))

def chkrebtii_kalman(x_t, t, theta=None, x_out=None):
    """Chkrebtii function in kalman format."""
    x_out[0] = sin(2*t) - x_t[0]
    return

def chkrebtii_odeint(x_t, t):
    """Chkrebtii function in odeint format."""
    return [x_t[1], sin(2*t) - x_t[0]]

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
        print(sum_err(kalman_sim[1:, 1], detode[1:, 1]))
        self.assertLessEqual(sum_err(kalman_sim[:, 0], detode[:, 0]), 10.0)
        self.assertLessEqual(sum_err(kalman_sim[1:, 1], detode[1:, 1]), 10.0)

if __name__ == '__main__':
    unittest.main()
