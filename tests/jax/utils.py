import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.

    Adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    x1 = X1.ravel() * 1.0
    x2 = X2.ravel() * 1.0
    return jnp.max(jnp.abs((x1 - x2)/(0.1 + x1)))

def var_sim(key, size):
    """
    Generate a variance matrix of given size.
    """
    Z = random.normal(key, (size, size))
    return jnp.matmul(Z.T, Z)
