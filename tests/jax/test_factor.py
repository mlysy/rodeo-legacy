"""
Unit tests for the factorizations of pdfs used in the bridge proposal.
In particular, suppose we are given

```
        tildeX_1 | tildeX_0 ~ N(Q_1 tildeX_0, R_1)
    tildeX_2 | tildeX_1, tildeX_0 ~ N(Q_2 tildeX_1, R_2)
Y | tildeX_2, tildeX_1, tildeX_0 ~ N(W tildeX_2 + b, \Omega)
```

We are interested in factoring the pdf p(tildeX_1, tildeX_2, Y | tildeX_0)= 
p(tildeX_1 | tildeX_0) p(tildeX_2 | tildeX_1) p(Y | tildeX_2). 
and P(W, Y) = P(Y) P(W|Y).
"""
import unittest
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from utils import var_sim, rel_err
from jax.config import config
config.update("jax_enable_x64", True)

class TestFact(unittest.TestCase):

    def test_tri_fact(self):
        """
        Check if p(tildeX_1, tildeX_2, Y | tildeX_0)= 
        p(tildeX_1 | tildeX_0) p(tildeX_2 | tildeX_1) p(Y | tildeX_2).
        """
        key = random.PRNGKey(0)
        n_lat = 3  # number of dimensions of X
        n_obs = 2  # number of dimensions of Y

        # generate random values of the matrices and vectors
        key, subkey = random.split(key)
        tildeX_0 = random.normal(subkey, (n_lat,))

        key, *subkeys = random.split(key, num=4)
        Q_1 = random.normal(subkeys[0], (n_lat, n_lat))
        R_1 = var_sim(subkeys[1], n_lat)
        tildeX_1 = random.normal(subkeys[2], (n_lat,))

        key, *subkeys = random.split(key, num=4)
        Q_2 = random.normal(subkeys[0], (n_lat, n_lat))
        R_2 = var_sim(subkeys[1], n_lat)
        tildeX_2 = random.normal(subkeys[2], (n_lat,))

        key, *subkeys = random.split(key, num=5)
        W = random.normal(subkeys[0], (n_obs, n_lat))
        b = random.normal(subkeys[1], (n_obs,))
        Omega = var_sim(subkeys[2], n_obs)
        Y = random.normal(subkeys[3], (n_obs,))

        # joint distribution using single mvn
        mu_tildeX1 = jnp.matmul(Q_1, tildeX_0)
        mu_tildeX2 = jnp.matmul(Q_2, mu_tildeX1)
        mu_Y = jnp.matmul(W, mu_tildeX2) + b
        Sigma_12 = jnp.matmul(R_1, Q_2.T)
        Sigma_13 = jnp.matmul(Sigma_12, W.T)
        Sigma_22 = jnp.matmul(Q_2, Sigma_12) + R_2
        Sigma_23 = jnp.matmul(Sigma_22, W.T)
        Sigma_33 = jnp.matmul(W, Sigma_23) + Omega
        mu = jnp.block([mu_tildeX1, mu_tildeX2, mu_Y])
        Sigma = jnp.block([
            [R_1, Sigma_12, Sigma_13],
            [Sigma_12.T, Sigma_22, Sigma_23],
            [Sigma_13.T, Sigma_23.T, Sigma_33]
        ])
        
        # joint distribution using factorization
        lpdf1 = jsp.stats.multivariate_normal.logpdf(tildeX_1, jnp.matmul(Q_1, tildeX_0), R_1)
        lpdf1 = lpdf1 + jsp.stats.multivariate_normal.logpdf(tildeX_2, jnp.matmul(Q_2, tildeX_1), R_2)
        lpdf1 = lpdf1 + \
            jsp.stats.multivariate_normal.logpdf(Y, jnp.matmul(W, tildeX_2) + b, Omega)

        # joint distribution using single mvn
        lpdf2 = jsp.stats.multivariate_normal.logpdf(jnp.block([tildeX_1, tildeX_2, Y]), mu, Sigma)

        self.assertAlmostEqual(rel_err(lpdf1, lpdf2), 0.0)
        
if __name__ == '__main__':
    unittest.main()
