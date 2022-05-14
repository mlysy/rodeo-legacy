import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


@partial(jax.jit, static_argnums=(0, 5))
def euler(fun, x0, theta, tmin, tmax, n_eval):
    r"Evaluate Euler approximation given :math:`\theta`"
    step_size = (tmax - tmin)/n_eval

    # setup lax.scan:
    # scan function
    def scan_fun(x_old, t):
        x_new = x_old + fun(x_old, tmin + step_size*t, theta)*step_size
        return x_new, x_new
    (_, X_t) = lax.scan(scan_fun, x0, jnp.arange(n_eval))

    X_t = jnp.concatenate([x0[None], X_t])
    return X_t
