import unittest
import numpy as np
from math import sin
from scipy import integrate

from probDE.ibm import ibm_init
from probDE.cython.KalmanODE import KalmanODE
from kalmanode_numba import KalmanODE as KnumODE
from probDE.utils.utils import rand_mat, indep_init, zero_pad

def sum_err(X1, X2):
    """Sum of relative error between two numpy arrays."""
    return np.sum(np.abs((X1.ravel() - X2.ravel())/X1.ravel()))

def chkrebtii_kalman(x_t, t, theta=None, x_out=None):
    """Chkrebtii function in kalman format."""
    x_out[0] = sin(2*t) - x_t[0]
    return

class KalmanNumbaTest(unittest.TestCase):
    def test_numba(self):
       # LHS vector of ODE
        w_mat = np.array([[0.0, 0.0, 1.0]])

        # These parameters define the order of the ODE and the CAR(p) process
        n_obs = 1
        n_deriv = [3]
        n_deriv_prior = [4]
        p = sum(n_deriv_prior)

        # it is assumed that the solution is sought on the interval [tmin, tmax].
        n_eval = 100
        tmin = 0
        tmax = 10

        # The rest of the parameters can be tuned according to ODE
        # For this problem, we will use
        sigma = [.5]

        # Initial value, x0, for the IVP
        x0 = np.array([-1., 0., 1.])
        x0_state = zero_pad(x0, n_deriv, n_deriv_prior)
        W = zero_pad(w_mat, n_deriv, n_deriv_prior)

        # Get parameters needed to run the solver
        dt = (tmax-tmin)/n_eval
        # All necessary parameters are in kinit, namely, T, c, R, W
        kinit = ibm_init(dt, n_deriv_prior, sigma)
        kinit = indep_init(kinit, n_deriv_prior)
        z_state = rand_mat(2*(n_eval+1), p)

        # Initialize the Kalman class
        kalmanode = KalmanODE(p, n_obs, tmin, tmax, n_eval, chkrebtii_kalman, **kinit)
        kalmanode.z_states = z_state
        # Run the solver to get an approximation
        k_sim = kalmanode.solve(x0_state, W, mv=False, sim=True)

        # Get numba solution
        knumode = KnumODE(p, n_obs, tmin, tmax, n_eval, chkrebtii_kalman, **kinit)
        knumode.z_state = z_state
        kk_sim = knumode.solve(x0_state, W, None, sim_sol=True)

        self.assertLessEqual(sum_err(k_sim[:, 0],  kk_sim[:, 0]), 0.0)
        self.assertLessEqual(sum_err(k_sim[1:, 1], kk_sim[1:, 1]), 0.0)

if __name__ == '__main__':
    unittest.main()
