import unittest
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from rodeo.jax.ode_block_solve import *
from rodeo.jax.ode_block_solve import _solve_filter
import ode_block_solve_for as bfor
import utils
from jax.config import config
config.update("jax_enable_x64", True)

class TestrodeoFor(unittest.TestCase):
    """
    Test if lax scan version of rodeo gives the same results as for-loop version.

    """
    setUp = utils.fitz_setup 

    def test_interrogate_rodeo(self):
        x_meas1, var_meas1 = interrogate_rodeo(
            key=self.key,
            fun=self.fitz_jax,
            t=self.t,
            theta=self.theta,
            wgt_meas=self.W_block,
            mu_state_pred=self.x0_block,
            var_state_pred=self.var_block
        )
        # for
        x_meas2, var_meas2 = bfor.interrogate_rodeo(
            key=self.key,
            fun=self.fitz_jax,
            t=self.t,
            theta=self.theta,
            wgt_meas=self.W_block,
            mu_state_pred=self.x0_block,
            var_state_pred=self.var_block
        )
        
        self.assertAlmostEqual(utils.rel_err(x_meas1, x_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)
    
    def test_interrogate_chkrebtii(self):
        x_meas1, var_meas1 = interrogate_chkrebtii(
            key=self.key,
            fun=self.fitz_jax,
            t=self.t,
            theta=self.theta,
            wgt_meas=self.W_block,
            mu_state_pred=self.x0_block,
            var_state_pred=self.var_block
        )
        # for
        x_meas2, var_meas2 = bfor.interrogate_chkrebtii(
            key=self.key,
            fun=self.fitz_jax,
            t=self.t,
            theta=self.theta,
            wgt_meas=self.W_block,
            mu_state_pred=self.x0_block,
            var_state_pred=self.var_block
        )
        
        self.assertAlmostEqual(utils.rel_err(x_meas1, x_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)

    def test_solve_sim(self):
        sim1 = solve_sim(key=self.key, fun=self.fitz_jax,
                         x0=self.x0_block, theta=self.theta,
                         tmin=self.tmin, tmax=self.tmax, n_eval=self.n_eval,
                         wgt_meas=self.W_block, **self.ode_init)
        # for
        sim2 = bfor.solve_sim(key=self.key, fun=self.fitz_jax,
                              x0=self.x0_block, theta=self.theta,
                              tmin=self.tmin, tmax=self.tmax, n_eval=self.n_eval,
                              wgt_meas=self.W_block, **self.ode_init)
        self.assertAlmostEqual(utils.rel_err(sim1, sim2), 0.0)
    
    def test_solve_mv(self):
        mu1, var1 = solve_mv(key=self.key, fun=self.fitz_jax,
                             x0=self.x0_block, theta=self.theta,
                             tmin=self.tmin, tmax=self.tmax, n_eval=self.n_eval,
                             wgt_meas=self.W_block, **self.ode_init)
        # for
        mu2, var2 = bfor.solve_mv(key=self.key, fun=self.fitz_jax,
                                  x0=self.x0_block, theta=self.theta,
                                  tmin=self.tmin, tmax=self.tmax, n_eval=self.n_eval,
                                  wgt_meas=self.W_block, **self.ode_init)
        self.assertAlmostEqual(utils.rel_err(mu1, mu2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var1, var2), 0.0)
    
    def test_solve(self):
        sim1, mu1, var1 = solve(key=self.key, fun=self.fitz_jax,
                                x0=self.x0_block, theta=self.theta,
                                tmin=self.tmin, tmax=self.tmax, n_eval=self.n_eval,
                                wgt_meas=self.W_block, **self.ode_init)
        # for
        sim2, mu2, var2 = bfor.solve(key=self.key, fun=self.fitz_jax,
                                     x0=self.x0_block, theta=self.theta,
                                     tmin=self.tmin, tmax=self.tmax, n_eval=self.n_eval,
                                     wgt_meas=self.W_block, **self.ode_init)
        self.assertAlmostEqual(utils.rel_err(mu1, mu2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var1, var2), 0.0)
        self.assertAlmostEqual(utils.rel_err(sim1, sim2), 0.0)

if __name__ == '__main__':
    unittest.main()
