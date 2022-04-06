from re import sub
import unittest
import jax
import jax.numpy as jnp
import jax.random as random
import rodeo.jax.kalmantv as ktv
import utils
from rodeo.jax.utils import mvncond
from jax.config import config
config.update("jax_enable_x64", True)
# --- kalmantv.predict ---------------------------------------------------------

class TestKalmanTVGM(unittest.TestCase):
    """
    Test if KalmanTV gives the same results as Gaussian Markov process.

    """
    setUp = utils.kalman_setup

    def test_predict(self):
        # theta_{0|0}
        mu_state_past, var_state_past = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mu_state_pred1, var_state_pred1 = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        mu_state_pred2, var_state_pred2 = ktv.predict(
            mu_state_past=mu_state_past,
            var_state_past=var_state_past,
            mu_state=self.mu_state[1],
            wgt_state=self.wgt_state[1],
            var_state=self.var_state[1]
        )

        self.assertAlmostEqual(utils.rel_err(mu_state_pred1, mu_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)

    def test_update(self):
        # theta_{1|0}
        mu_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{1|1}
        mu_state_filt1, var_state_filt1 = utils.kalman_theta(
            m=1, y=self.x_meas, mu=self.mu_gm, Sigma=self.var_gm
        )
        mu_state_filt2, var_state_filt2 = ktv.update(
            mu_state_pred=mu_state_pred,
            var_state_pred=var_state_pred,
            x_meas=self.x_meas[1],
            mu_meas=self.mu_meas[1],
            wgt_meas=self.wgt_meas[1],
            var_meas=self.var_meas[1]
        )

        self.assertAlmostEqual(utils.rel_err(mu_state_filt1, mu_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)

    def test_filter(self):
        # theta_{0|0}
        mu_state_past, var_state_past = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mu_state_pred1, var_state_pred1 = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{1|1}
        mu_state_filt1, var_state_filt1 = utils.kalman_theta(
            m=1, y=self.x_meas, mu=self.mu_gm, Sigma=self.var_gm
        )
        mu_state_pred2, var_state_pred2, \
            mu_state_filt2, var_state_filt2 = ktv.filter(
                mu_state_past=mu_state_past,
                var_state_past=var_state_past,
                mu_state=self.mu_state[1],
                wgt_state=self.wgt_state[1],
                var_state=self.var_state[1],
                x_meas=self.x_meas[1],
                mu_meas=self.mu_meas[1],
                wgt_meas=self.wgt_meas[1],
                var_meas=self.var_meas[1]
            )

        self.assertAlmostEqual(utils.rel_err(mu_state_pred1, mu_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mu_state_filt1, mu_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)

    def test_smooth_mv(self):
        # theta_{1|1}
        mu_state_next, var_state_next = utils.kalman_theta(
            m=1, y=self.x_meas, mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{0|0}
        mu_state_filt, var_state_filt = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mu_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{0|1}
        mu_state_smooth1, var_state_smooth1 = utils.kalman_theta(
            m=0, y=self.x_meas, mu=self.mu_gm, Sigma=self.var_gm
        )

        mu_state_smooth2, var_state_smooth2 = ktv.smooth_mv(
            mu_state_next=mu_state_next,
            var_state_next=var_state_next,
            mu_state_filt=mu_state_filt,
            var_state_filt=var_state_filt,
            mu_state_pred=mu_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=self.wgt_state[0]
        )

        self.assertAlmostEqual(utils.rel_err(mu_state_smooth1, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_smooth1, var_state_smooth2), 0.0)

    def test_smooth_sim(self):
        # theta_{0|0}
        mu_state_filt, var_state_filt = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mu_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{0:1|1}
        mu_state_smooth, var_state_smooth = utils.kalman_theta(
            m=[0, 1], y=self.x_meas, mu=self.mu_gm, Sigma=self.var_gm
        )
        A, b, V = mvncond(
            mu=mu_state_smooth.ravel(),
            Sigma=var_state_smooth.reshape(2*self.n_state, 2*self.n_state),
            icond=jnp.array([False]*self.n_state + [True]*self.n_state)
        )
        x_state_smooth1 = random.multivariate_normal(self.key, A.dot(self.x_state_next)+b, V)
        #x_state_smooth1 = ktv._state_sim(
        #    mu_state=A.dot(self.x_state_next)+b,
        #    var_state=V,
        #    z_state=self.z_state
        #)

        x_state_smooth2 = ktv.smooth_sim(
            key =self.key,
            x_state_next=self.x_state_next,
            mu_state_filt=mu_state_filt,
            var_state_filt=var_state_filt,
            mu_state_pred=mu_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=self.wgt_state[0]
        )
        self.assertAlmostEqual(utils.rel_err(x_state_smooth1, x_state_smooth2), 0.0)

    def test_smooth(self):
        # theta_{1|1}
        mu_state_next, var_state_next = utils.kalman_theta(
            m=1, y=self.x_meas, mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{0|0}
        mu_state_filt, var_state_filt = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mu_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mu_gm, Sigma=self.var_gm
        )
        # theta_{0:1|1}
        mu_state_smooth1, var_state_smooth1 = utils.kalman_theta(
            m=[0, 1], y=self.x_meas, mu=self.mu_gm, Sigma=self.var_gm
        )
        A, b, V = mvncond(
            mu=mu_state_smooth1.ravel(),
            Sigma=var_state_smooth1.reshape(2*self.n_state, 2*self.n_state),
            icond=jnp.array([False]*self.n_state + [True]*self.n_state)
        )
        #x_state_smooth1 = ktv._state_sim(
        #    mu_state=A.dot(self.x_state_next)+b,
        #    var_state=V,
        #    z_state=self.z_state
        #)
        x_state_smooth1 = jax.random.multivariate_normal(self.key, A.dot(self.x_state_next)+b, V)
        x_state_smooth2, mu_state_smooth2, var_state_smooth2 = ktv.smooth(
            key=self.key,
            x_state_next=self.x_state_next,
            mu_state_next=mu_state_next,
            var_state_next=var_state_next,
            mu_state_filt=mu_state_filt,
            var_state_filt=var_state_filt,
            mu_state_pred=mu_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=self.wgt_state[0],
        )
        self.assertAlmostEqual(utils.rel_err(x_state_smooth1, x_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mu_state_smooth1[0], mu_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_smooth1[0, :, 0, :].squeeze(), var_state_smooth2), 0.0)

if __name__ == '__main__':
    unittest.main()
