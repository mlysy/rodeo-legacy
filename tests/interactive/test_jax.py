# test of new jax implementation

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, random, lax
# from rodeo.ibm import ibm_init
from rodeo.ibm.ibm_init import ibm_state as ribm_state
from rodeo.ibm.ibm_init import ibm_init as ribm_init
import rodeo.ibm as ribm
import rodeo.jax.ode_block_solve as jblock
from rodeo.utils.utils import indep_init, zero_pad
from rodeo.jax.ode_solve import *
from rodeo.jax.ode_solve import _solve_filter
import rodeo.jax.ibm_init as jibm
import rodeo.jax.KalmanODE as KalmanODE
from rodeo.jax.utils import *


def print_diff(name, x1, x2):
    ad = jnp.max(jnp.abs(x1 - x2))
    print(name + " abs diff = {}".format(ad))
    return ad



# --- test rodeo.solve ---------------------------------------------------------


if False:

    # LHS vector of ODE
    w_mat = jnp.array([[0.0, 0.0, 1.0]])

    # These parameters define the order of the ODE and the CAR(p) process
    n_deriv = [2]
    n_deriv_prior = [4]

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    n_eval = 10
    tmin = jnp.array(0.)
    tmax = jnp.array(10.)

    # IBM process scale factor
    sigma = [.5]

    # Initial value, x0, for the IVP
    x0 = jnp.array([-1., 0., 1.])
    x0_state = jnp.array(zero_pad(x0, n_deriv, n_deriv_prior))
    W = jnp.array(zero_pad(w_mat, n_deriv, n_deriv_prior))

    # Get parameters needed to run the solver
    # All necessary parameters are in kinit, namely, T, c, R, W
    kinit = ribm.ibm_init((tmax-tmin)/n_eval, n_deriv_prior, sigma)
    kinit = {key: jnp.array(value[0]) for (key, value) in kinit.items()}

    def higher_fun(x_t, t, theta):
        """2nd order ODE function from Chkrebtii et al (2016)."""
        return jnp.sin(2.0*t) - x_t[0]

    key = jax.random.PRNGKey(0)

    filt_out = _solve_filter(key=key, fun=higher_fun, theta=jnp.array(0.),
                             x0=x0_state, tmin=tmin, tmax=tmax, n_eval=n_eval,
                             wgt_meas=W, **kinit)

    filt_out2 = KalmanODE._solve_filter(
        fun=higher_fun, x0=x0_state,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=W, wgt_state=kinit["wgt_state"],
        mu_state=kinit["mu_state"], var_state=kinit["var_state"],
        key=key, theta=jnp.array(0.), method="rodeo")

    print_diff("mu_state_filt", filt_out2[2].T, filt_out["state_filt"][0])
    print_diff("mu_state_pred", filt_out2[0].T, filt_out["state_pred"][0])
    print_diff("var_state_filt", jnp.moveaxis(
        filt_out2[3], 2, 0), filt_out["state_filt"][1])
    print_diff("var_state_pred", jnp.moveaxis(
        filt_out2[1], 2, 0), filt_out["state_pred"][1])

    sim_out = solve_sim(key=key, fun=higher_fun, theta=jnp.array(0.),
                        x0=x0_state, tmin=tmin, tmax=tmax, n_eval=n_eval,
                        wgt_meas=W, **kinit)

    sim_out2 = KalmanODE.solve_sim(
        fun=higher_fun, x0=x0_state,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=W, wgt_state=kinit["wgt_state"],
        mu_state=kinit["mu_state"], var_state=kinit["var_state"],
        key=key, theta=jnp.array(0.), method='rodeo'
    )

    print_diff("x_state_smooth", sim_out, sim_out2)

    mv_out = solve_mv(key=key, fun=higher_fun, theta=jnp.array(0.),
                      x0=x0_state, tmin=tmin, tmax=tmax, n_eval=n_eval,
                      wgt_meas=W, **kinit)

    mv_out2 = KalmanODE.solve_mv(
        fun=higher_fun, x0=x0_state,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=W, wgt_state=kinit["wgt_state"],
        mu_state=kinit["mu_state"], var_state=kinit["var_state"],
        key=key, theta=jnp.array(0.), method='rodeo'
    )

    print_diff("mu_state_smooth", mv_out[0], mv_out2[0])
    print_diff("var_state_smooth", mv_out[1], mv_out2[1])

else:
    pass

# --- test ibm_init ------------------------------------------------------------

if False:
    dt = .1
    n_deriv = 4
    sigma = .5
    A, Q = ribm_state(dt, n_deriv, sigma)
    A = jnp.array(A)
    Q = jnp.array(Q)

    A2, Q2 = jibm.ibm_state(dt, n_deriv, sigma)

    # I, J = jnp.meshgrid(jnp.arange(n_deriv), jnp.arange(n_deriv),
    #                     indexing="ij", sparse=True)

    # A2 = J-I
    # A2 = jnp.maximum(dt**A2/jnp.exp(jsp.special.gammaln(A2+1)),
    #                  jnp.zeros((n_deriv, n_deriv)))

    print_diff("ibm A", A, A2)
    print_diff("ibm Q", Q, Q2)

    n_order = jnp.array([4])
    sigma = jnp.array([.5])
    jint = jibm.ibm_init(dt, n_order, sigma)
    rint = ribm_init(dt, n_order, sigma)
    print_diff("ibm wgt_state", jint['wgt_state'], jnp.array(rint['wgt_state']))
    print_diff("ibm var_state", jint['var_state'], jnp.array(rint['var_state']))
else:
    pass


# --- test rodeo.solve_block ---------------------------------------------------

if True:
    def fitz_jax(X_t, t, theta):
        "FitzHugh-Nagumo ODE."
        a, b, c = theta
        V, R = X_t[0], X_t[3]
        return jnp.stack([c*(V - V*V*V/3 + R),
                          -1/c*(V - a + b*R)])

    n_deriv = 1  # Total state
    n_obs = 2  # Total measures
    n_deriv_prior = 3

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    tmin = 0.
    tmax = 40.
    h = 0.1
    n_eval = int((tmax-tmin)/h)

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    sigma = .1

    # Initial value, x0, for the IVP
    x0 = jnp.array([-1., 1.])
    v0 = jnp.array([1, 1/3])
    X0 = jnp.concatenate([x0, v0])

    # pad the inputs
    w_mat = jnp.array([[0., 1., 0., 0.], [0., 0., 0., 1.]])
    W = jnp.array(zero_pad(w_mat, [n_deriv]*n_obs, [n_deriv_prior]*n_obs))

    # function parameter
    t = jnp.array(.25)  # time
    theta = jnp.array([0.2, 0.2, 3])  # True theta
    n_order = jnp.array([n_deriv_prior]*n_obs)
    sigma = jnp.array([sigma]*n_obs)
    ode_init = jibm.ibm_init(h, n_order, sigma)
    x0_state = jnp.array(zero_pad(X0, [n_deriv]*n_obs, [n_deriv_prior]*n_obs))
    kinit = indep_init(ode_init, [n_deriv_prior]*n_obs)
    kinit = dict((k, jnp.array(v)) for k, v in kinit.items())

    key = jax.random.PRNGKey(0)
    x_meas, var_meas = interrogate_rodeo(
        key=key,
        fun=fitz_jax,
        t=t,
        theta=theta,
        wgt_meas=W,
        mu_state_pred=x0_state,
        var_state_pred=kinit["var_state"]
    )

    n_bmeas = 1
    n_bstate = n_deriv_prior
    W_block = jnp.array([W[0, 0:n_bstate],
                         W[1, n_bstate:2*n_bstate]])
    W_block = jnp.reshape(W_block, newshape=(2,1,3))
    var_block = ode_init['var_state']
    x0_block = jnp.reshape(x0_state, (n_obs, n_bstate))
    x_meas2, var_meas2 = jblock.interrogate_rodeo(
        key=key,
        fun=fitz_jax,
        t=t,
        theta=theta,
        wgt_meas=W_block,
        mu_state_pred=x0_block,
        var_state_pred=var_block
    )

    print_diff("x_meas", jnp.reshape(x_meas, (n_obs, n_bmeas)), x_meas2)
    print_diff("var_meas",
               jnp.array([var_meas[0, 0:n_bmeas],
                         var_meas[1, n_bmeas:2*n_bmeas]]),
               var_meas2)

    x_meas, var_meas = interrogate_chkrebtii(
        key=key,
        fun=fitz_jax,
        t=t,
        theta=theta,
        wgt_meas=W,
        mu_state_pred=x0_state,
        var_state_pred=kinit["var_state"]
    )

    x_meas2, var_meas2 = jblock.interrogate_chkrebtii(
        key=key,
        fun=fitz_jax,
        t=t,
        theta=theta,
        wgt_meas=W_block,
        mu_state_pred=x0_block,
        var_state_pred=var_block
    )

    print_diff("x_meas_chk", jnp.reshape(x_meas, (n_obs, n_bmeas)), x_meas2)
    print_diff("var_meas_chk",
               jnp.array([var_meas[0, 0:n_bmeas],
                         var_meas[1, n_bmeas:2*n_bmeas]]),
               var_meas2)
    x_meas, var_meas = interrogate_schober(
        key=key,
        fun=fitz_jax,
        t=t,
        theta=theta,
        wgt_meas=W,
        mu_state_pred=x0_state,
        var_state_pred=kinit["var_state"]
    )

    x_meas2, var_meas2 = jblock.interrogate_schober(
        key=key,
        fun=fitz_jax,
        t=t,
        theta=theta,
        wgt_meas=W_block,
        mu_state_pred=x0_block,
        var_state_pred=var_block
    )
    #print(var_meas2)
    #print(var_meas)
    print_diff("x_meas_sch", jnp.reshape(x_meas, (n_obs, n_bmeas)), x_meas2)
    print_diff("var_meas_sch",
               jnp.array([var_meas[0, 0:n_bmeas],
                         var_meas[1, n_bmeas:2*n_bmeas]]),
               var_meas2)

    filt_out = _solve_filter(key=key, fun=fitz_jax, theta=theta,
                             x0=x0_state, tmin=tmin, tmax=tmax, n_eval=n_eval,
                             wgt_meas=W, **kinit)

    filt_out2 = jblock._solve_filter(key=key, theta=theta,
                                     fun=fitz_jax, x0=x0_block,
                                     tmin=tmin, tmax=tmax, n_eval=n_eval,
                                     wgt_meas=W_block, **ode_init)
    
    mu_state_filt2 = jnp.reshape(filt_out2['state_filt'][0], newshape=(-1, 6))
    mu_state_pred2 = jnp.reshape(filt_out2['state_pred'][0], newshape=(-1, 6))
    var_state_filt2 = block_diag(filt_out2["state_filt"][1])
    var_state_pred2 = block_diag(filt_out2["state_pred"][1])

    print_diff("mu_state_filt", mu_state_filt2, filt_out["state_filt"][0])
    print_diff("mu_state_pred", mu_state_pred2, filt_out["state_pred"][0])
    print_diff("var_state_filt", var_state_filt2, filt_out["state_filt"][1])
    print_diff("var_state_pred", var_state_pred2, filt_out["state_pred"][1])

    sim_out = solve_sim(key=key, fun=fitz_jax, theta=theta,
                        x0=x0_state, tmin=tmin, tmax=tmax, n_eval=n_eval,
                        wgt_meas=W, **kinit)

    sim_out2 = jblock.solve_sim(key=key, theta=theta,
                                fun=fitz_jax, x0=x0_block,
                                tmin=tmin, tmax=tmax, n_eval=n_eval,
                                wgt_meas=W_block, **ode_init)
    
    print_diff("x_state_smooth", sim_out, sim_out2)
    
    mv_out = solve_mv(key=key, fun=fitz_jax, theta=theta,
                      x0=x0_state, tmin=tmin, tmax=tmax, n_eval=n_eval,
                      wgt_meas=W, **kinit)

    mv_out2 = jblock.solve_mv(key=key, theta=theta,
                              fun=fitz_jax, x0=x0_block,
                              tmin=tmin, tmax=tmax, n_eval=n_eval,
                              wgt_meas=W_block, **ode_init)

    print_diff("mu_state_smooth", mv_out[0], mv_out2[0])
    print_diff("var_state_smooth", mv_out[1], mv_out2[1])

    solve_out = solve(key=key, fun=fitz_jax, theta=theta,
                      x0=x0_state, tmin=tmin, tmax=tmax, n_eval=n_eval,
                      wgt_meas=W, **kinit)

    solve_out2 = jblock.solve(key=key, theta=theta,
                              fun=fitz_jax, x0=x0_block,
                              tmin=tmin, tmax=tmax, n_eval=n_eval,
                              wgt_meas=W_block, **ode_init)

    print_diff("x_state_smooth2", solve_out[0], solve_out2[0])
    print_diff("mu_state_smooth2", solve_out[1], solve_out2[1])
    print_diff("var_state_smooth2", solve_out[2], solve_out2[2])


else:
    pass


