"""
.. module:: cov_fun
    :synopsis: Covariance and cross-covariance functions for the solution process x_t and its derivative v_t = dx_t/dt.
"""
import numpy as np
from BayesODE.cov_rect import cov_vv_re, cov_xv_re, cov_xx_re
from BayesODE.cov_square_exp import cov_vv_se, cov_xv_se, cov_xx_se
from BayesODE.cov_exp import cov_vv_ex, cov_xv_ex, cov_xx_ex

def cov_prior(tseq, model, gamma, alpha):
    """Calculate covariance and cross-covariance matrices between :math:x_t and :math:v_t.

    :param tseq: Vector of *N* timepoints at which to calculate the covariances.
    :type tseq: float
    :param model: Name of the covariance model to use.  Possible choices are:

        - *exp2*: Squared-exponential.
        - *rect*: Rectangular.
        - *exp*: Exponential.

    :type model: str
    :param gamma: Covariance decorrelation time.
    :type gamma: float
    :param alpha: Covariance scale parameter.
    :type alpha: float
    :returns: Three *N x N* matrices:

        - **Sigma_vv** = cov(v(tseq), v(tseq))
        - **Sigma_xx** = cov(x(tseq), x(tseq))
        - **Sigma_xv** = cov(x(tseq), v(tseq))

    :rtype: numpy array
    """

    N = len(tseq)
    Sigma_vv = np.zeros((N, N))
    Sigma_xx = np.zeros((N, N))
    Sigma_xv = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if model == 'exp2':
                Sigma_vv[i,j] = cov_vv_se(tseq[i], tseq[j], gamma)/alpha
                Sigma_xx[i,j] = cov_xx_se(tseq[i], tseq[j], gamma)/alpha
                Sigma_xv[i,j] = cov_xv_se(tseq[i], tseq[j], gamma)/alpha
            elif model == 'rect':
                Sigma_vv[i,j] = cov_vv_re(tseq[i], tseq[j], gamma)/alpha
                Sigma_xx[i,j] = cov_xx_re(tseq[i], tseq[j], gamma)/alpha
                Sigma_xv[i,j] = cov_xv_re(tseq[i], tseq[j], gamma)/alpha
            elif model == 'exp':
                Sigma_vv[i,j] = cov_vv_ex(tseq[i], tseq[j], gamma)/alpha
                Sigma_xx[i,j] = cov_xx_ex(tseq[i], tseq[j], gamma)/alpha
                Sigma_xv[i,j] = cov_xv_ex(tseq[i], tseq[j], gamma)/alpha
            else:
                raise ValueError('Invalid model.')

    return Sigma_vv, Sigma_xx, Sigma_xv
