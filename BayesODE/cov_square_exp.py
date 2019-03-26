"""
.. module:: cov_square_exp
    :synopsis: Covariance and cross-covariance functions for the solution process `x_t` and its derivative `v_t = dx_t/dt` under the squared-exponential correlation model, `cov(v_t, v_s) = exp(-|t-s|^2/gamma^2)`.
"""

@jit
def cov_vv_se(t, s, gamma):
    """Calculate covariance function of `v_t`.

    Evaluates `cov(v_t, v_s) = exp(-|t-s|^2/gamma^2)`.

    :param t: First time point.
    :type t: float
    :param s: Second time point.
    :type s: float
    :param gamma: Decorrelation time, such that `cov(v_t, v_{t+gamma}` = 1/e`.
    :type gamma: float
    :returns: The value of the covariance.
    :rtype: float
    """
    # no point in dragging around sqrt(pi), etc.
    # instead, the output should be:
    # z = (t-s)/gamma
    # return exp(-z*z) # better to use z*z instead of z**2, google it :)
    # I've left your original code for now...
    return exp(-((t - s)**2) / (4 * gamma**2)) * sqrt(pi) * gamma


@jit
def cov_xv_se(t, s, gamma):
    """Calculate cross-covariance between `x_t` and `v_t`.

    .. note::
    - Replace integer powers by multiplication!
    - Better to break up calculation into multiple lines, and save calculations whenever possible.  For example, you compute `gamma**3` four times in this function...better to save it once as `g3 = gamma**3`.
    """
    ans = pi * gamma**2 * (s) * erf(s / (2 * w)) \
        + 2 * sqrt(pi) * gamma**3 * exp(-(s**2) / (4 * gamma**2)) \
        - pi * gamma**2 * (t - s) * erf((t - s) / (2 * gamma)) \
        - 2 * sqrt(pi) * gamma**3 * exp(-(t - s)**2 / (4 * gamma**2)) \
        + pi * gamma**2 * s * erf(t / (2 * gamma)) \
        + 2 * sqrt(pi) * gamma**3 * exp(-(t**2) / (4 * gamma**2)) \
        - 2 * sqrt(pi) * gamma**3
    return ans
    # return integrate.dblquad(sigma_xdot, 0, t, lambda s: 0, lambda s: s)


@jit
def cov_xx_se(t, s, gamma):
    """Calculate covariance function of `x_t`.
    """
    ans = pi * gamma**2 * erf((t - s) / (2 * gamma)) + pi * gamma**2 * erf(s / (2 * gamma))
    return ans
    # return integrate.quad(sigma_xdot, 0, s, args=t)
