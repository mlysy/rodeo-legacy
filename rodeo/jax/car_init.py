import jax
import jax.numpy as jnp

from rodeo.jax.utils import mvncond

def car_mou(roots, sigma):
    """
    Calculates parameters for the mOU CAR(p) process.
    
    Args:
        roots (ndarray(p)): Roots to the p-th order polynomial of the CAR(p) process.
        sigma (float): Parameter in mOU volatility matrix.

    Returns:
        (tuple):
        - **Sigma_tilde** (ndarray(p, p)): :math:`\widetilde{\Sigma}` in CAR process.
        - **Q** (ndarray(p, p)): :math:`Q` in CAR process.

    """
    p = len(roots)
    row = jnp.ones(p)
    
    # lax.scan setup
    # scan function
    def scan_fun(carry, t):
        stack = carry
        carry = -carry * roots
        return carry, stack
    
    _, Q = jax.lax.scan(scan_fun, row, jnp.arange(p))
    Q_inv = jnp.linalg.pinv(Q)
    Sigma = jnp.zeros(p)
    Sigma = Sigma.at[p-1].set(sigma * sigma)
    Sigma_tilde = jnp.matmul(Q_inv * Sigma, Q_inv.T)
    
    return Sigma_tilde, Q

def car_varinf(roots, sigma):
    """
    Computes the :math:`V_{\infty}` for the CAR(p) process.
    
    Args:
        roots (ndarray(p)): Roots to the p-th order polynomial of the car(p) process.
        sigma (float): Parameter in mOU volatility matrix.
    
    Returns:
        (ndarray(p, p)): :math:`V_{\infty}`.

    """
    p = len(roots)
    Sigma_tilde, Q = car_mou(roots, sigma)
    I, J = jnp.meshgrid(jnp.arange(p), jnp.arange(p),
                        indexing="ij", sparse=True)
    
    V_tilde_inf = Sigma_tilde/(roots[I] + roots[J])
    V_inf = jnp.linalg.multi_dot([Q, V_tilde_inf, Q.T])
    
    return V_inf

def car_var(dt, roots, sigma):
    """
    Computes the variance function for the CAR(p) process :math:`var(X_t)`
    
    Args:
        dt (float): Time point at which :math:`x_t` is evaluated. 
        roots (ndarray(p)): Roots to the p-th order polynomial of the car(p) process.
        sigma (float): Parameter in mOU volatility matrix.

    Returns:
        (ndarray(p, p)): Evaluates :math:`var(X_t)`.

    """
    p = len(roots)
    Sigma_tilde, Q = car_mou(roots, sigma)
    I, J = jnp.meshgrid(jnp.arange(p), jnp.arange(p),
                        indexing="ij", sparse=True)
    V_tilde = Sigma_tilde/(roots[I] + roots[J]) * (1.0 - jnp.exp(-(roots[I] + roots[J]) * dt))
    var = jnp.linalg.multi_dot([Q, V_tilde, Q.T])  # V_deltat
    return var

def root_gen(tau, p):
    """
    Creates p CAR model roots.

    Args:
        tau (int): First root parameter.
        p (int): Number of roots to generate.

    Returns:
        (ndarray(p)): Vector size of p roots generated.

    """
    return jnp.append(1/tau, jnp.linspace(1 + 1/(10*(p-1)), 1.1, p-1))*10

def car_initial_draw(key, n_order, tau, sigma, x0):
    """
    Computes the initial draw X0 for the kalman process :math:`X0 \sim N(c_0, 0_{pxp})` 
    where :math:`c_0 \sim p(N(\lambda, V_{\infty}) | x_0 = x0)`.

    Args:
        key (PRNGKey): Jax PRNGKey.
        n_order (ndarray(n_block)): Derivative order for each variate. 
        sigma (ndarray(n_block)): Parameter in mOU volatility matrix.
        x0 (ndarray(n_block, n_ode)): Initial conditions of the ode.

    Returns:
        (ndarray(n_block, p)): Simulate :math:`X0 \sim N(c_0, 0_{pxp})` conditioned on x0.

    """
    n_block = len(n_order)
    p = jnp.max(n_order)
    x0_state = [None]*n_block
    key, *subkeys = jax.random.split(key, num=n_block+1)
    for i in range(n_block):
        q = len(x0[i]) - 1
        if p == q+1:
            x0_state[i] = x0[i]
            continue
        broots = root_gen(tau[i], p)
        V_inf = car_varinf(broots, sigma[i])
        icond = jnp.array([True]*(q+1) + [False]*(p-q-1))
        A, b, V = mvncond(jnp.zeros(p), V_inf, icond)
        z0 = jax.random.multivariate_normal(subkeys[i], A.dot(x0[i]) + b, V) 
        bx0_state = jnp.concatenate([x0[i], z0])
        x0_state[i] = bx0_state
    return jnp.array(x0_state)


def car_state(dt, roots, sigma):    
    """
    Calculate the state transition matrix and variance matrix using the CAR(p) process in Kalman solver.
        
    Args:
        dt (float): The size size between simulation points.
        roots (ndarray(p)): Roots to the p-th order polynomial of the CAR(p) 
            process (roots must be negative).
        sigma (float): Parameter in mOU volatility matrix

    Returns:
        (tuple):
        - **wgtState** (ndarray(p, p)): The state transition matrix defined in Kalman solver.
        - **varState** (ndarray(p, p)): The state variance matrix defined in Kalman solver.

    """
    _, Q = car_mou(roots, sigma)
    Q_inv = jnp.linalg.pinv(Q)
    varState = car_var(dt, roots, sigma)
    wgtState = jnp.matmul(Q * jnp.exp(-roots*dt), Q_inv)
    return wgtState, varState
    
def car_init(dt, n_order, tau, sigma):
    """
    Calculates the initial parameters necessary for the Kalman solver.
    Args:
        dt (float): The step size between simulation points.
        n_order (ndarray(n_block)): Derivative order of each variate.
        tau (ndarray(n_block)): First root parameter.
        sigma (ndarray(n_block)): Parameter in mOU volatility matrix.
        
    Returns:
        (tuple):
        - **wgt_state** (ndarray(n_block, p, p)) Transition matrix defining the solution prior; :math:`Q`.
        - **mu_state** (ndarray(n_block, p)): Transition_offsets defining the solution prior; denoted by :math:`c_n`.
        - **var_state** (ndarray(n_block, p, p)) Variance matrix defining the solution prior; :math:`R`.
    
    """
    n_block = len(n_order)
    p = jnp.max(n_order)
    mu_state = jnp.zeros((n_block, p))
    
    wgt_state = [None]*n_block
    var_state = [None]*n_block    

    #wgt_state, var_state = jax.vmap(lambda b:
    #    ibm_state(dt, (n_order[b]-1), sigma[b]))(jnp.arange(n_block))
    
    for i in range(n_block):
        broots = root_gen(tau[i], p)
        wgt_state[i], var_state[i] = car_state(dt, broots, sigma[i])
    
    wgt_state = jnp.array(wgt_state)
    var_state = jnp.array(var_state)
    
#     def vmap_fun(b):
#         broots = root_gen2(tau[b], n_deriv_prior[b])
#         bwgt_state, bvar_state = car_state2(dt, broots, sigma[b])
#         return bwgt_state, bvar_state
    
#     wgt_state, var_state = jax.vmap(lambda b: vmap_fun(b))(jnp.arange(n_block))
    
    
    init = {"wgt_state":wgt_state,  "mu_state":mu_state,
            "var_state":var_state}
    
    return init
