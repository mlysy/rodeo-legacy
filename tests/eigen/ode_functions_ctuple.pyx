# cythonized versions of the ODE functions used in the timing benchmarks
import cython
from libc.math cimport sin


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef chkrebtii_fun(double[::1] X, double t, tuple theta, double[::1] out):
    """
    Chkrebtii ODE function.
    """
    out[0] = sin(2*t) - X[0]
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lorenz_fun(double[::1] X, double t, (double, double, double) theta, double[::1] out):
    """
    Lorenz63 ODE function.
    """
    rho, sigma, beta = theta
    cdef int p = len(X)//3
    x, y, z = X[p*0], X[p*1], X[p*2]
    out[0] = -sigma*x + sigma*y
    out[1] = rho*x - y - x*z
    out[2] = -beta*z + x*y
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fitz_fun(double[::1] X, double t, (double, double, double) theta, double[::1] out):
    """
    FitzHugh-Nagumo ODE function.
    """
    a, b, c = theta
    cdef int p = len(X)//2
    V, R = X[0], X[p]
    out[0] = c*(V - V**3/3 + R)
    out[1] = -1/c*(V - a + b*R)
    return


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mseir_fun(double[::1] X, double t, (double, double, double, double, double, double) theta, double[::1] out):
    """
    MSEIR ODE function.
    """
    cdef int p = len(X)//5
    M, S, E, I, R = X[::p]
    cdef double N = M+S+E+I+R
    Lambda, delta, beta, mu, epsilon, gamma = theta
    out[0] = Lambda - delta*M - mu*M
    out[1] = delta*M - beta*S*I/N - mu*S
    out[2] = beta*S*I/N - (epsilon + mu)*E
    out[3] = epsilon*E - (gamma + mu)*I
    out[4] = gamma*I - mu*R
    return
