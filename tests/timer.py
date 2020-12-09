from timeit import default_timer as timer
from scipy.integrate import odeint


def timing(kalmanode, x0_state, W, theta, n_loops):
    start = timer()
    for i in range(n_loops):
        kalmanode.solve_sim(x0_state, W, theta)
    end = timer()
    return (end - start)/n_loops


def det_timing(f, x0, tseq, n_loops, theta=None):
    start = timer()
    for i in range(n_loops):
        _ = odeint(f, x0, tseq, args=(theta, ))
    end = timer()
    return (end - start)/n_loops
