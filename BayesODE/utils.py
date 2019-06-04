
import scipy.linalg as scl

def mvCond(mu, Sigma, icond):
    """
    For y ~ N(mu, Sigma), returns A, b, and V such that y[~icond] | y[icond] ~ N(A * y[icond] + b, V).
    """
    # if y1 = y[~icond] and y2 = y[icond], should have A = Sigma12 * Sigma22^{-1}
    A = np.dot(Sigma[np.ix_(~icond, icond)],scl.cho_solve(scl.cho_factor(Sigma[np.ix_(icond,icond)]), np.identity(sum(icond))))
    b = mu[~icond] - np.dot(A, mu[icond]) # mu1 - A * mu2
    V = Sigma[np.ix_(~icond,~icond)] - np.dot(A, Sigma[np.ix_(icond,~icond)]) # Sigma11 - A * Sigma21
    return A, b, V
