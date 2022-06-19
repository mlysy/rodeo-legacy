
with open("fit_helper.py") as fp:
    exec(fp.read())

# ODE Function definition -------------------------------------
def ode_fun(X_t, t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X_t[:, 0]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q= theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    return jnp.array([[x1], [x2], [x3], [x4], [x5], [x6]])

def rax_fun(t, X_t, theta):
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

def ode_pad(X_t, t, theta):
    p = len(X_t)//6
    S, E, I, R, A, H = X_t[::p]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    
    out = jnp.array([[S, x1, 0],
                     [E, x2, 0],
                     [I, x3, 0], 
                     [R, x4, 0],
                     [A, x5, 0],
                     [H, x6, 0]])
    return out

def covid_obs(X_t, theta):
    r"Compute the observations as detailed in the paper"
    I_in = theta[1]*X_t[:,1]/theta[3]
    H_in = X_t[:,2]/theta[5]
    X_in = jnp.array([I_in, H_in]).T
    return X_in
def loglike(Y_t, X_t, step_size, obs_size, theta):
    data_tseq = np.linspace(tmin+1, tmax, int((tmax-tmin)/obs_size))
    ode_tseq = np.linspace(tmin, tmax, int((tmax-tmin)/step_size)+1)
    X_t = thinning(ode_tseq, data_tseq, X_t)
    X_in = covid_obs(X_t, theta)
    return jnp.sum(jsp.stats.poisson.logpmf(Y_t, X_in))

def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the lognormal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))


n_deriv = 1 # number of derivatives in IVP
n_obs = 6 # number of observations.
n_deriv_prior = 3 # number of derivatives in IBM prior

# it is assumed that the solution is sought on the interval [tmin, tmax].
tmin = 0.
tmax = 60.

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = jnp.array([.1]*n_obs)
n_order = jnp.array([n_deriv_prior]*n_obs)

# Initial value, x0, for the IVP
#x0 = np.array([63884630, 15492, 21752, 0, 618013, 13388])
x0 = np.array([63884630, None, None, 0, 618013, 13388]) 

# W matrix: dimension is n_eq x sum(n_deriv)
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W = jnp.array(W_mat)

# Observations
Y_t = np.load("saves/seirah_Y.npy")

# logprior parameters
theta_true = np.array([2.23, 0.034, 0.55, 5.1, 2.3, 1.13]) # True theta
n_theta = len(theta_true)
phi_mean = np.zeros(n_theta)
phi_sd = np.log(10)*np.ones(n_theta)

# observation and step sizes
dtlst = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
obs_size = 1
step_size = dtlst[1]
    
# parameters for rodeo
kinit = ibm_init(step_size, n_order, sigma)
n_eval = int((tmax-tmin)/step_size)

# parameters for diffrax
term = ODETerm(rax_fun)
solver = Dopri5()
tseq = jnp.linspace(tmin, tmax, int((tmax-tmin)/step_size)+1)
saveat = SaveAt(ts=tseq)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

phi_init = np.append(np.log(theta_true), np.array([15492, 21752]))
mv_jit = jax.jit(solve_mv, static_argnums=(1, 6))
kalman_nlpost(phi_init, Y_t, x0, step_size, obs_size, phi_mean, phi_sd)
print(diffrax_nlpost(phi_init, Y_t, x0, step_size, obs_size, phi_mean, phi_sd))
