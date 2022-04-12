import unittest
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import rodeo.jax.kalmantv as ktv
import utils
from jax.config import config
config.update("jax_enable_x64", True)

class TestKalmanTVJit(unittest.TestCase):
    """
    Check whether jit and unjitted gives the same result.
    """
    setUp = utils.kalman_setup
    
    def test_predict(self):
        self.key, *subkeys = random.split(self.key, 3)
        mu_state_past = random.normal(subkeys[0], (self.n_state,))
        var_state_past = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_past = var_state_past.dot(var_state_past.T)
        # without jit
        mu_state_pred1, var_state_pred1 = \
             ktv.predict(mu_state_past, var_state_past,
                         self.mu_state[0], self.wgt_state[0], self.var_state[0])
        # with jit
        predict_jit = jax.jit(ktv.predict)
        mu_state_pred2, var_state_pred2 = \
            predict_jit(mu_state_past, var_state_past,
                        self.mu_state[0], self.wgt_state[0], self.var_state[0])
        # objective function for gradient
        def obj_fun(mu_state_past, var_state_past, 
                    mu_state, wgt_state, var_state): 
            return jnp.mean(
                ktv.predict(mu_state_past, var_state_past,
                            mu_state, wgt_state, var_state)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mu_state_past, var_state_past,
            self.mu_state[0], self.wgt_state[0], self.var_state[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mu_state_past, var_state_past,
            self.mu_state[0], self.wgt_state[0], self.var_state[0])
        self.assertAlmostEqual(utils.rel_err(mu_state_pred1, mu_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_update(self):
        self.key, *subkeys = random.split(self.key, 3)
        mu_state_pred = random.normal(subkeys[0], (self.n_state,))
        var_state_pred = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_pred = var_state_pred.dot(var_state_pred.T)
        # without jit
        mu_state_filt1, var_state_filt1 = \
             ktv.update(mu_state_pred, var_state_pred,
                        self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # with jit
        update_jit = jax.jit(ktv.update)
        mu_state_filt2, var_state_filt2 = \
            update_jit(mu_state_pred, var_state_pred,
                       self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # objective function for gradient
        def obj_fun(mu_state_pred, var_state_pred,
                    x_meas, mu_meas, wgt_meas, var_meas):
            return jnp.mean(
                ktv.update(mu_state_pred, var_state_pred,
                           x_meas, mu_meas, wgt_meas, var_meas)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mu_state_pred, var_state_pred,
            self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mu_state_pred, var_state_pred,
            self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        self.assertAlmostEqual(utils.rel_err(mu_state_filt1, mu_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_filter(self):
        self.key, *subkeys = random.split(self.key, 3)
        mu_state_past = random.normal(subkeys[0], (self.n_state,))
        var_state_past = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_past = var_state_past.dot(var_state_past.T)
        # without jit
        mu_state_pred1, var_state_pred1, mu_state_filt1, var_state_filt1 = \
            ktv.filter(mu_state_past, var_state_past,
                       self.mu_state[0], self.wgt_state[0], self.var_state[0],
                       self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # with jit
        filter_jit = jax.jit(ktv.filter)
        mu_state_pred2, var_state_pred2, mu_state_filt2, var_state_filt2 = \
            filter_jit(mu_state_past, var_state_past,
                       self.mu_state[0], self.wgt_state[0], self.var_state[0],
                       self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # objective function for gradient
        def obj_fun(mu_state_past, var_state_past,
                    mu_state, wgt_state, var_state,
                    x_meas, mu_meas, wgt_meas, var_meas):
            return jnp.mean(
                ktv.filter(mu_state_past, var_state_past,
                           mu_state, wgt_state, var_state,
                           x_meas, mu_meas, wgt_meas, var_meas)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mu_state_past, var_state_past,
            self.mu_state[0], self.wgt_state[0], self.var_state[0],
            self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mu_state_past, var_state_past,
            self.mu_state[0], self.wgt_state[0], self.var_state[0],
            self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        self.assertAlmostEqual(utils.rel_err(mu_state_pred1, mu_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mu_state_filt1, mu_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)
    

    def test_smooth_mv(self):
        self.key, *subkeys = random.split(self.key, 7)
        mu_state_next = random.normal(subkeys[0], (self.n_state,))
        var_state_next = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_next = var_state_next.dot(var_state_next.T)
        mu_state_filt = random.normal(subkeys[2], (self.n_state,))
        var_state_filt = random.normal(subkeys[3], (self.n_state, self.n_state))
        var_state_filt = var_state_filt.dot(var_state_filt.T)
        mu_state_pred = random.normal(subkeys[4], (self.n_state,))
        var_state_pred = random.normal(subkeys[5], (self.n_state, self.n_state))
        var_state_pred = var_state_pred.dot(var_state_pred.T)
        # without jit
        mu_state_smooth1, var_state_smooth1 = \
            ktv.smooth_mv(mu_state_next, var_state_next,
                          mu_state_filt, var_state_filt,
                          mu_state_pred, var_state_pred,
                          self.wgt_state[0])
        # with jit
        mv_jit = jax.jit(ktv.smooth_mv)
        mu_state_smooth2, var_state_smooth2 = \
            mv_jit(mu_state_next, var_state_next,
                   mu_state_filt, var_state_filt,
                   mu_state_pred, var_state_pred,
                   self.wgt_state[0])
        # objective function for gradient
        def obj_fun(mu_state_next, var_state_next,
                    mu_state_filt, var_state_filt,
                    mu_state_pred, var_state_pred,
                    wgt_state):
            return jnp.mean(
                ktv.smooth_mv(mu_state_next, var_state_next,
                              mu_state_filt, var_state_filt,
                              mu_state_pred, var_state_pred,
                              wgt_state)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mu_state_next, var_state_next,
            mu_state_filt, var_state_filt,
            mu_state_pred, var_state_pred,
            self.wgt_state[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mu_state_next, var_state_next,
            mu_state_filt, var_state_filt,
            mu_state_pred, var_state_pred,
            self.wgt_state[0])
        self.assertAlmostEqual(utils.rel_err(mu_state_smooth1, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_smooth1, var_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_smooth_sim(self):
        self.key, *subkeys = random.split(self.key, 4)
        mu_state_past = random.normal(subkeys[0], (self.n_state,))
        var_state_past = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_past = var_state_past.dot(var_state_past.T)
        x_state_next = random.normal(subkeys[2], (self.n_state,))
        # without jit
        mu_state_pred1, var_state_pred1, mu_state_filt1, var_state_filt1 = \
            ktv.filter(mu_state_past, var_state_past,
                       self.mu_state[0], self.wgt_state[0], self.var_state[0],
                       self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        mu_state_sim1, var_state_sim1 = \
            ktv.smooth_sim(x_state_next,
                           mu_state_filt1, var_state_filt1,
                           mu_state_pred1, var_state_pred1,
                           self.wgt_state[0])
        # with jit
        filter_jit = jax.jit(ktv.filter)
        mu_state_pred2, var_state_pred2, mu_state_filt2, var_state_filt2 = \
            filter_jit(mu_state_past, var_state_past,
                       self.mu_state[0], self.wgt_state[0], self.var_state[0],
                       self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        sim_jit = jax.jit(ktv.smooth_sim)
        mu_state_sim2, var_state_sim2 = \
            sim_jit(x_state_next,
                    mu_state_filt2, var_state_filt2,
                    mu_state_pred2, var_state_pred2,
                    self.wgt_state[0])
        # objective function for gradient
        def obj_fun(x_state_next,
                    mu_state_filt, var_state_filt,
                    mu_state_pred, var_state_pred,
                    wgt_state):
            return jnp.mean(
                ktv.smooth_sim(x_state_next,
                               mu_state_filt, var_state_filt,
                               mu_state_pred, var_state_pred,
                               wgt_state)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun, argnums=1)(
            x_state_next,
            mu_state_filt1, var_state_filt1,
            mu_state_pred1, var_state_pred1,
            self.wgt_state[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun, argnums=1))(
            x_state_next,
            mu_state_filt2, var_state_filt2,
            mu_state_pred2, var_state_pred2,
            self.wgt_state[0])
        self.assertAlmostEqual(utils.rel_err(mu_state_sim1, mu_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_sim1, var_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_smooth(self):
        self.key, *subkeys = random.split(self.key, 6)
        mu_state_past = random.normal(subkeys[0], (self.n_state,))
        var_state_past = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_past = var_state_past.dot(var_state_past.T)
        x_state_next = random.normal(subkeys[2], (self.n_state,))
        mu_state_next = random.normal(subkeys[3], (self.n_state,))
        var_state_next = random.normal(subkeys[4], (self.n_state, self.n_state))
        var_state_next = var_state_next.dot(var_state_next.T)
        # without jit
        mu_state_pred1, var_state_pred1, mu_state_filt1, var_state_filt1 = \
            ktv.filter(mu_state_past, var_state_past,
                       self.mu_state[0], self.wgt_state[0], self.var_state[0],
                       self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        mu_state_sim1, var_state_sim1, mu_state_smooth1, var_state_smooth1= \
            ktv.smooth(x_state_next, 
                       mu_state_next, var_state_next,
                       mu_state_filt1, var_state_filt1,
                       mu_state_pred1, var_state_pred1,
                       self.wgt_state[0])
        # with jit
        filter_jit = jax.jit(ktv.filter)
        mu_state_pred2, var_state_pred2, mu_state_filt2, var_state_filt2 = \
            filter_jit(mu_state_past, var_state_past,
                       self.mu_state[0], self.wgt_state[0], self.var_state[0],
                       self.x_meas[0], self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        smooth_jit = jax.jit(ktv.smooth)
        mu_state_sim2, var_state_sim2, mu_state_smooth2, var_state_smooth2 = \
            smooth_jit(x_state_next,
                       mu_state_next, var_state_next,
                       mu_state_filt2, var_state_filt2,
                       mu_state_pred2, var_state_pred2,
                       self.wgt_state[0])
        # objective function for gradient
        def obj_fun(x_state_next,
                    mu_state_next, var_state_next,
                    mu_state_filt, var_state_filt,
                    mu_state_pred, var_state_pred,
                    wgt_state):
            return jnp.mean(
                ktv.smooth(x_state_next,
                           mu_state_next, var_state_next,
                           mu_state_filt, var_state_filt,
                           mu_state_pred, var_state_pred,
                           wgt_state)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            x_state_next,
            mu_state_next, var_state_next,
            mu_state_filt1, var_state_filt1,
            mu_state_pred1, var_state_pred1,
            self.wgt_state[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            x_state_next,
            mu_state_next, var_state_next,
            mu_state_filt2, var_state_filt2,
            mu_state_pred2, var_state_pred2,
            self.wgt_state[0])
        self.assertAlmostEqual(utils.rel_err(mu_state_smooth1, mu_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_smooth1, var_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mu_state_sim1, mu_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_sim1, var_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_forecast(self):
        self.key, *subkeys = random.split(self.key, 3)
        mu_state_pred = random.normal(subkeys[0], (self.n_state,))
        var_state_pred = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_pred = var_state_pred.dot(var_state_pred.T)
        # without jit
        mu_fore1, var_fore1 = \
            ktv.forecast(mu_state_pred, var_state_pred,
                         self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # with jit
        fore_jit = jax.jit(ktv.forecast)
        mu_fore2, var_fore2 = \
            fore_jit(mu_state_pred, var_state_pred,
                     self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # objective function for gradient
        def obj_fun(mu_state_pred, var_state_pred,
                    mu_meas, wgt_meas, var_meas):
            return jnp.mean(
                ktv.forecast(mu_state_pred, var_state_pred,
                             mu_meas, wgt_meas, var_meas)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mu_state_pred, var_state_pred,
            self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mu_state_pred, var_state_pred,
            self.mu_meas[0], self.wgt_meas[0], self.var_meas[0])
        self.assertAlmostEqual(utils.rel_err(mu_fore1, mu_fore2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_fore1, var_fore2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

if __name__ == '__main__':
    unittest.main()
