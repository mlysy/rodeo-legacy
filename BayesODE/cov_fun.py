"""
.. module:: cov_fun
    :synopsis: Covariance and cross-covariance functions for the solution process `x_t` and its derivative `v_t = dx_t/dt`.
"""

def cov_prior(tseq, model, gamma, alpha):
    """Calculate covariance and cross-covariance matrices between `x_t` and `v_t`.

    :param tseq: Vector of `N` timepoints at which to calculate the covariances.
    :type tseq: float
    :param model: Name of covariance model to use.  Possible choices are:
    - `exp2`: Squared-exponential.
    - `rect`: Rectangular.
    - `exp`: Exponential.
    :type model: str
    :param gamma: Covariance decorrelation time.
    :type gamma: float
    :param alpha: Covariance scale parameter.
    :type alpha: float
    :returns: Three `N x N` matrices:
    - `Sigma_vv = cov(v(tseq), v(tseq))`
    - `Sigma_xx = cov(x(tseq), x(tseq))`
    - `Sigma_xv = cov(x(tseq), v(tseq))`
    :rtype: ???
    """

    N = len(tseq)
    Sigma_vv = np.zeros((N, N))
    Sigma_xx = np.zeros((N, N))
    Sigma_xv = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if model == 'exp2':
                Sigma_vv[i,j] = cov_vv_se(t[i], t[j], gamma)/alpha
                Sigma_xx[i,j] = cov_xx_se(t[i], t[j], gamma)/alpha
                Sigma_xv[i,j] = cov_xv_se(t[i], t[j], gamma)/alpha
            elif model == 'rect':
                Sigma_vv[i,j] = cov_vv_re(t[i], t[j], gamma)/alpha
                Sigma_xx[i,j] = cov_xx_re(t[i], t[j], gamma)/alpha
                Sigma_xv[i,j] = cov_xv_re(t[i], t[j], gamma)/alpha
            elif model == 'exp':
                Sigma_vv[i,j] = cov_vv_ex(t[i], t[j], gamma)/alpha
                Sigma_xx[i,j] = cov_xx_ex(t[i], t[j], gamma)/alpha
                Sigma_xv[i,j] = cov_xv_ex(t[i], t[j], gamma)/alpha
            else:
                raise ValueError('Invalid model.')

    return Sigma_vv, Sigma_xx, Sigma_xv
