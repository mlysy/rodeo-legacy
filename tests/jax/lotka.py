from scipy.integrate import odeint
from inference_jax import inference
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

class lotka(inference):
    r"""
    Perform parameter inference for the Lotka-Volterra model using the base inference class. 

    Args:
        x0 (ndarray(n_state)): Initial value of the state variable :math:`x_t` at 
            time :math:`t = 0`.
        phi_true (ndarray(n_theta)): Log of true :math:`\theta`.
        sigma (float): Noise parameter to simulate the observations.
        tseq (ndarray(n_eval)): Discretization points to simulate the data.
        log (bool): Flag for log scale.
    """

    def loglike(self, x, mean, var):
        r"Calculate the loglikelihood of the lognormal distribution."
        return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=var))
    
    def simulate(self, x0, phi_true, sigma, tseq, log=False):
        r"Get the observations assuming a normal distribution."
        theta_true = np.exp(phi_true)
        X_t = odeint(self.fun, x0, tseq, args=(theta_true,))[1:,]
        e_t = np.random.default_rng().normal(loc=0.0, scale=1, size=X_t.shape)
        if log is False:
            Y_t = X_t + sigma*e_t
        else:
            Y_t = np.exp(X_t) + sigma*e_t
            X_t = np.exp(X_t)
        return Y_t, X_t
