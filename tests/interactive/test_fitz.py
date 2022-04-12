import unittest
from scipy.integrate import odeint
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random

from rodeo.jax.ode_solve import *
from rodeo.jax.ode_block_solve import _solve_filter
import utils

class TestFitzOdeint(unittest.TestCase):
    """
    Check whether rodeo and odeint gives approximately the same results.
    
    """
    setUp = utils.fitz_setup

    def test_fitz(self):
        det = odeint(self.fitz_odeint, self.x0, self.tseq, args=(self.theta,))
        sim = solve_sim(key=self.key, fun=self.fitz_jax,
                         x0=self.x0_block, theta=self.theta,
                         tmin=self.tmin, tmax=self.tmax, n_eval=self.n_eval,
                         wgt_meas=self.W_block, **self.ode_init)
        m = solve_mv(key=self.key, fun=self.fitz_jax,
                     x0=self.x0_block, theta=self.theta,
                     tmin=self.tmin, tmax=self.tmax, n_eval=self.n_eval,
                     wgt_meas=self.W_block, **self.ode_init)[0]
        self.assertLessEqual(utils.rel_err(sim[:, :, 0], det), 5.0)
        self.assertLessEqual(utils.rel_err(m[:, :, 0], det), 5.0)

if __name__ == '__main__':
    unittest.main()
