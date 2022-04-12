from scipy.integrate import odeint
from .inference import inference
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from rodeo.jax.ode_solve import *

mv_jit = jax.jit(solve_mv, static_argnums=(1, 6))

def seirah(X_t, t, theta):
    "SEIRAH ODE function"
    p = len(X_t)//6
    S, E, I, R, A, H = X_t[::p]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q= theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    return jnp.array([x1, x2, x3, x4, x5, x6])

class seirahinf(inference):
    r"Inference using the France Covid data from Prague et al."
    
    def loglike_pois(self, x, mean):
        r"Calculate the loglikelihood of the poisson distribution."
        return jnp.sum(jsp.stats.poisson.logpmf(x, mean))
    
    def covid_obs(self, X_t, theta):
        r"Compute the observations as detailed in the paper"
        I_in = theta[1]*X_t[:,1]/theta[3]
        H_in = X_t[:,2]/theta[5]
        X_in = jnp.array([I_in, H_in]).T
        return X_in
    
    def simulate(self, x0, theta, tseq):
        r"""Get the observations for the SEIRAH Covid example.
        None of the compartments are directly observed, however 
        the daily infections and hospitalizations are observed. 
        They can be computed as
        
        .. math::

            I^{(in)}(t) = rE(t)/D_e
            H^{(in)}(t) = I(t)/D_q

        """
        X_t = odeint(seirah, x0, tseq, args=(theta,))
        X_in = self.covid_obs(X_t, theta)
        Y_in = np.random.default_rng(111).poisson(X_in)
        return Y_in, X_in
    
    def kalman_solve(self, step_size, obs_size, x0, theta):
        r"Using Kalman solver to compute solutions"
        self.key, subkey = jax.random.split(self.key)
        data_tseq = np.linspace(self.tmin+1, self.tmax, int((self.tmax-self.tmin)/obs_size))
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = mv_jit(subkey, self.fun, x0, theta, self.tmin, self.tmax, self.n_eval, self.W, **self.kinit)[0]
        X_t = X_t[:, :, 0]
        X_t = self.thinning(ode_tseq, data_tseq, X_t)
        X_in = self.covid_obs(X_t, theta)
        return X_in
    
    def euler_solve(self, x0, step_size, obs_size, theta):
        r"Using Euler method to compute solutions"
        data_tseq = np.linspace(self.tmin+1, self.tmax, int((self.tmax-self.tmin)/obs_size))
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.euler(seirah, x0, ode_tseq, data_tseq, step_size, theta)
        X_in = self.covid_obs(X_t, theta)
        return X_in
