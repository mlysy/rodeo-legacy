import rodeo
import rodeo.jax.gauss_markov as gm
import rodeo.jax.kalmantv as ktv
from rodeo.jax.utils import mvncond
import numpy as np
import jax
import jax.numpy as jnp


def print_diff(name, x1, x2):
    ad = np.max(np.abs(x1 - x2))
    print(name + " abs diff = {}".format(ad))
    return ad


key = jax.random.PRNGKey(0)

n_meas = 1
n_state = 2
n_tot = 2

key, *subkeys = jax.random.split(key, 10)
mu_state = jax.random.normal(subkeys[0], (n_tot, n_state))
var_state = jax.random.normal(subkeys[1], (n_tot, n_state, n_state))
var_state = jax.vmap(lambda vs: vs.dot(vs.T))(var_state)
wgt_state = jax.random.normal(subkeys[2], (n_tot-1, n_state, n_state))
# wgt_state = jnp.zeros((n_tot-1, n_state, n_state))
mu_meas = jax.random.normal(subkeys[3], (n_tot, n_meas))
var_meas = jax.random.normal(subkeys[4], (n_tot, n_meas, n_meas))
var_meas = jax.vmap(lambda vs: vs.dot(vs.T))(var_meas)
wgt_meas = jax.random.normal(subkeys[5], (n_tot, n_meas, n_state))
# wgt_meas = jnp.zeros((n_tot, n_meas, n_state))
x_meas = jax.random.normal(subkeys[6], (n_tot, n_meas))
x_state_next = jax.random.normal(subkeys[7], (n_state,))
z_state = jax.random.normal(subkeys[8], (n_state,))

A_gm, b_gm, C_gm = gm.kalman2gm(
    wgt_state=wgt_state,
    mu_state=mu_state,
    var_state=var_state,
    wgt_meas=wgt_meas,
    mu_meas=mu_meas,
    var_meas=var_meas
)

mu_gm, var_gm = gm.gauss_markov_mv(A=A_gm, b=b_gm, C=C_gm)
print(mu_gm)

def kalman_theta(m, y, mu, Sigma):
    """
    Calculate theta{m|n} using the joint density.

    Args:
        m (ndarray(n_x)): State variable indices.
        y (ndarray(n+1,n_meas)): Measurement variable observations.
        mu (ndarray(k,n_dim)): Joint mean, where `k >= max(m, n) + 1`.
        Sigma (ndarray(k,n_dim,k,n_dim)): Joint variance.

    Returns:
        (tuple):
        **mu_cond** (ndarray(n_x, n_state)): E[x_m | y_0:n].
        **var_cond** (ndarray(n_x, n_state, n_x, n_state)): var(x_m | y_0:n).
        **Note:** In both cases if `n_x == 1` the corresponding dimension is squeezed.
    """
    # dimensions
    n_tot, n_dim = mu.shape
    n_y, n_meas = y.shape
    n_state = n_dim - n_meas
    m = np.atleast_1d(m)
    n_x = len(m)
    # conditioning indices
    icond = np.full((n_tot, n_dim), False)
    icond[:n_y, n_state:n_dim] = True
    # marginal indices on the flattened scale
    imarg = np.full((n_tot, n_dim), False)
    imarg[np.ix_(m, np.arange(n_state))] = True
    imarg = np.ravel(imarg)[~np.ravel(icond)]
    A, b, V = mvncond(mu=np.ravel(mu),
                      Sigma=np.reshape(Sigma, (n_tot*n_dim, n_tot*n_dim)),
                      icond=np.ravel(icond))
    mu_mn = (A.dot(np.ravel(y)) + b)[imarg]
    V_mn = V[np.ix_(imarg, imarg)]
    mu_mn = mu_mn.reshape((n_x, n_state))
    V_mn = V_mn.reshape((n_x, n_state, n_x, n_state))
    print(icond)
    print(imarg)
    if n_x == 1:
        mu_mn = mu_mn.squeeze(axis=(0,))
        V_mn = V_mn.squeeze(axis=(0, 2))
    return mu_mn, V_mn

# --- kalmantv.predict ---------------------------------------------------------


if False:
    # theta_{0|0}
    mu_state_past, var_state_past = kalman_theta(
        m=0, y=jnp.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{1|0}
    mu_state_pred, var_state_pred = kalman_theta(
        m=1, y=jnp.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    mu_state_pred2, var_state_pred2 = ktv.predict(
        mu_state_past=mu_state_past,
        var_state_past=var_state_past,
        mu_state=mu_state[1],
        wgt_state=wgt_state[0],
        var_state=var_state[1]
    )

    print_diff("mu_state_pred", mu_state_pred, mu_state_pred2)
    print_diff("var_state_pred", var_state_pred, var_state_pred2)
else:
    pass

# --- kalman.update ------------------------------------------------------------

if False:
    # theta_{1|0}
    mu_state_pred, var_state_pred = kalman_theta(
        m=1, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{1|1}
    mu_state_filt, var_state_filt = kalman_theta(
        m=1, y=x_meas, mu=mu_gm, Sigma=var_gm
    )
    mu_state_filt2, var_state_filt2 = ktv.update(
        mu_state_pred=mu_state_pred,
        var_state_pred=var_state_pred,
        x_meas=x_meas[1],
        mu_meas=mu_meas[1],
        wgt_meas=wgt_meas[1],
        var_meas=var_meas[1]
    )

    print_diff("mu_state_filt", mu_state_filt, mu_state_filt2)
    print_diff("var_state_filt", var_state_filt, var_state_filt2)
else:
    pass

# --- kalman.filter ------------------------------------------------------------

if False:
    # theta_{0|0}
    mu_state_past, var_state_past = kalman_theta(
        m=0, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{1|0}
    mu_state_pred, var_state_pred = kalman_theta(
        m=1, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{1|1}
    mu_state_filt, var_state_filt = kalman_theta(
        m=1, y=x_meas, mu=mu_gm, Sigma=var_gm
    )
    mu_state_pred2, var_state_pred2, \
        mu_state_filt2, var_state_filt2 = ktv.filter(
            mu_state_past=mu_state_past,
            var_state_past=var_state_past,
            mu_state=mu_state[1],
            wgt_state=wgt_state[1],
            var_state=var_state[1],
            x_meas=x_meas[1],
            mu_meas=mu_meas[1],
            wgt_meas=wgt_meas[1],
            var_meas=var_meas[1]
        )

    print_diff("mu_state_pred", mu_state_pred, mu_state_pred2)
    print_diff("var_state_pred", var_state_pred, var_state_pred2)
    print_diff("mu_state_filt", mu_state_filt, mu_state_filt2)
    print_diff("var_state_filt", var_state_filt, var_state_filt2)
else:
    pass

# --- kalman.smooth_mv ---------------------------------------------------------

if False:
    # theta_{1|1}
    mu_state_next, var_state_next = kalman_theta(
        m=1, y=x_meas, mu=mu_gm, Sigma=var_gm
    )
    # theta_{0|0}
    mu_state_filt, var_state_filt = kalman_theta(
        m=0, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{1|0}
    mu_state_pred, var_state_pred = kalman_theta(
        m=1, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{0|1}
    mu_state_smooth, var_state_smooth = kalman_theta(
        m=0, y=x_meas, mu=mu_gm, Sigma=var_gm
    )

    mu_state_smooth2, var_state_smooth2 = ktv.smooth_mv(
        mu_state_next=mu_state_next,
        var_state_next=var_state_next,
        mu_state_filt=mu_state_filt,
        var_state_filt=var_state_filt,
        mu_state_pred=mu_state_pred,
        var_state_pred=var_state_pred,
        wgt_state=wgt_state[0]
    )

    print_diff("mu_state_smooth", mu_state_smooth, mu_state_smooth2)
    print_diff("var_state_smooth", var_state_smooth, var_state_smooth2)
else:
    pass


# --- kalmantv.smooth_sim ------------------------------------------------------

if True:
    # theta_{0|0}
    mu_state_filt, var_state_filt = kalman_theta(
        m=0, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{1|0}
    mu_state_pred, var_state_pred = kalman_theta(
        m=1, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )

    print('smooth')
    # theta_{0:1|1}
    mu_state_smooth, var_state_smooth = kalman_theta(
        m=[0, 1], y=x_meas, mu=mu_gm, Sigma=var_gm
    )
    print(mu_state_smooth)
    print(var_state_smooth.shape)
    A, b, V = mvncond(
        mu=mu_state_smooth.ravel(),
        Sigma=var_state_smooth.reshape(2*n_state, 2*n_state),
        icond=np.array([False]*n_state + [True]*n_state)
    )
    x_state_smooth = ktv._state_sim(
        mu_state=A.dot(x_state_next)+b,
        var_state=V,
        z_state=z_state
    )
    mu_state_smooth2, var_state_smooth2 = ktv.smooth_sim(
        x_state_next=x_state_next,
        mu_state_filt=mu_state_filt,
        var_state_filt=var_state_filt,
        mu_state_pred=mu_state_pred,
        var_state_pred=var_state_pred,
        wgt_state=wgt_state[0]
    )
    x_state_smooth2 = ktv._state_sim(
        mu_state=mu_state_smooth2,
        var_state=var_state_smooth2,
        z_state=z_state
    )
    print_diff("x_state_smooth", x_state_smooth, x_state_smooth2)
else:
    pass


# --- kalmantv.smooth ----------------------------------------------------------

if False:
    # theta_{1|1}
    mu_state_next, var_state_next = kalman_theta(
        m=1, y=x_meas, mu=mu_gm, Sigma=var_gm
    )
    # theta_{0|0}
    mu_state_filt, var_state_filt = kalman_theta(
        m=0, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{1|0}
    mu_state_pred, var_state_pred = kalman_theta(
        m=1, y=np.atleast_2d(x_meas[0]), mu=mu_gm, Sigma=var_gm
    )
    # theta_{0:1|1}
    mu_state_smooth, var_state_smooth = kalman_theta(
        m=[0, 1], y=x_meas, mu=mu_gm, Sigma=var_gm
    )
    A, b, V = mvncond(
        mu=mu_state_smooth.ravel(),
        Sigma=var_state_smooth.reshape(2*n_state, 2*n_state),
        icond=np.array([False]*n_state + [True]*n_state)
    )
    x_state_smooth = ktv._state_sim(
        mu_state=A.dot(x_state_next)+b,
        var_state=V,
        z_state=z_state
    )
    x_state_smooth2, mu_state_smooth2, var_state_smooth2 = ktv.smooth(
        x_state_next=x_state_next,
        mu_state_next=mu_state_next,
        var_state_next=var_state_next,
        mu_state_filt=mu_state_filt,
        var_state_filt=var_state_filt,
        mu_state_pred=mu_state_pred,
        var_state_pred=var_state_pred,
        wgt_state=wgt_state[0],
        z_state=z_state
    )
    print_diff("x_state_smooth", x_state_smooth, x_state_smooth2)
    print_diff("mu_state_smooth", mu_state_smooth[0], mu_state_smooth2)
    print_diff("var_state_smooth", var_state_smooth[0, :, 0, :].squeeze(),
               var_state_smooth2)
else:
    pass
