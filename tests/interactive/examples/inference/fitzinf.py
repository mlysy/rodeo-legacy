from scipy.integrate import odeint
from .inference import inference
import numpy as np
import jax

from rodeo.jax.ode_solve import *

mv_jit = jax.jit(solve_mv, static_argnums=(1, 6))

def fitz(X_t, t, theta):
    "Fitz ODE written for jax"
    a, b, c = theta
    p = len(X_t)//2
    V, R = X_t[0], X_t[p]
    return jnp.array([c*(V - V*V*V/3 + R),
                      -1/c*(V - a + b*R)])

class fitzinf(inference):
    r"Inference assuming a normal prior"
        
    def simulate(self, x0, theta, gamma, tseq):
        r"Get the observations assuming a normal distribution."
        X_t = odeint(fitz, x0, tseq, args=(theta,))
        e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t
    
    def kalman_solve(self, step_size, obs_size, x0, theta):
        r"Using Kalman solver to compute solutions"
        self.key, subkey = jax.random.split(self.key)
        data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = mv_jit(subkey, self.fun, x0, theta, self.tmin, self.tmax, self.n_eval, self.W, **self.kinit)[0]
        X_t = X_t[:, :, 0]
        X_t = self.thinning(ode_tseq, data_tseq, X_t)
        return X_t
    
    def euler_solve(self, x0, step_size, obs_size, theta):
        r"Using Euler method to compute solutions"
        data_tseq = jnp.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = jnp.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.euler(fitz, x0, ode_tseq, data_tseq, step_size, theta)
        return X_t
    