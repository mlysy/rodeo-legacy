from scipy.integrate import odeint
from inference_jax import inference
import jax.numpy as jnp
import jax.scipy as jsp

class proka(inference):
    r"""
    Perform parameter inference for the Prokaryotic auto-regulatory gene network
    model using the base inference class. 

    Args:
        x0 (ndarray(n_state)): Initial value of the state variable :math:`x_t` at 
            time :math:`t = 0`.
        phi_true (ndarray(n_theta)): Log of true :math:`\theta`.
        sigma (float): Noise parameter to simulate the observations.
        tseq (ndarray(n_eval)): Discretization points to simulate the data.
        log (bool): Flag for log scale.
    """

    def loglike(self, x, mean, sd):
        r"Calculate the loglikelihood of the normal distribution."
        return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

    def logprior(self, x, mean, sd):
        return self.loglike(x, mean, sd)
