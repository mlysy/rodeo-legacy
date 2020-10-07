import numpy as np
from numba import intc, float64, deferred_type
from numba import types
from numba.experimental import jitclass
from numba.extending import register_jitable
from numba.core.errors import TypingError
from numba import types, typeof
from kalmantv.numba.kalmantv import KalmanTV, _mvn_sim, _quad_form

kalman_type = deferred_type()
kalman_type.define(KalmanTV.class_type.instance_type)

@register_jitable
def _fempty(shape):
    """
    Create an empty ndarray with the given shape in fortran order.

    Numba cannot create fortran arrays directly, so this is done by transposing a C order array, as explained [here](https://numba.pydata.org/numba-doc/dev/user/faq.html#how-can-i-create-a-fortran-ordered-array).
    """
    return np.empty(shape[::-1]).T


@register_jitable
def _interrogate_chkrebtii(x_meas, var_meas,
                           fun, t, theta,
                           wgt_meas, mu_state_pred, var_state_pred, z_state,
                           tx_state, twgt_meas, tchol_state):
    """
    Interrogate method of Chkrebtii et al (2016).

    Args:
        x_meas (ndarray(n_meas)): Interrogation variable.
        var_meas (ndarray(n_meas, n_meas)): Interrogation variance.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        mu_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from
            times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given
            observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        z_state (ndarray(n_state)): Random vector simulated from :math:`N(0, 1)`.
        tx_state (ndarray(n_state)): Temporary state variable.
        twgt_meas (ndarray(n_meas, n_state)): Temporary matrix to store intermediate operation.
        tchol_state (ndarray(n_state, n_state)): Temporary matrix to store cholesky factorization.

    Returns:
        (tuple):
        - **x_meas** (ndarray(n_meas)): Interrogation variable.
        - **var_meas** (ndarray(n_meas, n_meas)): Interrogation variance.
        - **x_state** (ndarray(n_state)): Temporary state variable.
        - **twgt_meas** (ndarray(n_meas, n_state)): Temporary matrix to store intermediate operation.
        - **tchol_state** (ndarray(n_state, n_state)): Temporary matrix to store cholesky factorization.
    """
    var_meas[:]=0.
    _quad_form(var_meas, wgt_meas, var_state_pred, twgt_meas)
    _mvn_sim(tx_state, mu_state_pred, var_state_pred, z_state, tchol_state)
    fun(tx_state, t, theta, x_meas)
    return

@register_jitable
def _copynm(dest, source, name):
    """
    Copy an ndarray without memory allocation.

    Preserves flags and type of dest.
    """
    #if not isinstance(source, np.ndarray):
    #    raise TypingError("Value must be a Numpy array.")
    if not source.shape == dest.shape:
        raise TypingError("Value has incorrect shape.")
    dest[:] = source
    return

class _KalmanODE:
    def __init__(self, n_state, n_meas, tmin, tmax, n_eval,
                 ode_fun, mu_state, wgt_state, var_state, z_state):
        self.n_state = n_state
        self.n_meas = n_meas
        self.tmin = tmin
        self.tmax = tmax
        self.n_eval = n_eval
        self.n_steps = n_eval + 1
        # property variables
        self._mu_state = _fempty((n_state,))
        self._wgt_state = _fempty((n_state, n_state))
        self._var_state = _fempty((n_state, n_state))
        self._z_state = _fempty((n_state, 2*self.n_steps))
        #for key in init.keys():
        #    self.__setattr__(key, init[key])

        self._mu_state[:] = mu_state
        self._wgt_state[:] = wgt_state
        self._var_state[:] = var_state
        if z_state is not None:
            self._z_state[:] = z_state
        # Note: how can this be Numba'ed?
        self.ode_fun = ode_fun
        # internal memory
        self.mu_state_pred = _fempty((n_state, self.n_steps))
        self.var_state_pred = _fempty((n_state, n_state, self.n_steps))
        self.mu_state_filt = _fempty((n_state, self.n_steps))
        self.var_state_filt = _fempty((n_state, n_state, self.n_steps))
        #self.mu_state_smooth = _fempty((n_state, self.n_steps))
        #self.x_state_smooth = _fempty((n_state, self.n_steps)) 
        #self.var_state_smooth = _fempty((n_state, n_state, self.n_steps))
        # don't return the variance, so just need one for updating
        self.var_state_smooth = _fempty((n_state, n_state))
        self.x_meas = _fempty((n_meas,))
        self.mu_meas = np.zeros((n_meas,))  # doesn't get updated
        self.var_meas = _fempty((n_meas, n_meas))
        self.ktv = KalmanTV(n_meas, n_state)
        #self.time = np.linspace(self.tmin, self.tmax, self.n_steps)
        # temporaries
        self.tx_state = _fempty((n_state,))
        self.twgt_meas = _fempty((n_meas, n_state))
        self.tchol_state = _fempty((n_state, n_state))

    @property
    def mu_state(self):
        return self._mu_state

    @mu_state.setter
    def mu_state(self, value):
        _copynm(self._mu_state, value, "mu_state")
        return

    @property
    def wgt_state(self):
        return self._wgt_state

    @wgt_state.setter
    def wgt_state(self, value):
        _copynm(self._wgt_state, value, "wgt_state")
        return

    @property
    def var_state(self):
        return self._var_state

    @var_state.setter
    def var_state(self, value):
        _copynm(self._var_state, value, "var_state")
        return

    @property
    def z_state(self):
        return self._z_state

    @z_state.setter
    def z_state(self, value):
        _copynm(self._z_state, value, "z_state")
        return

    def solve(self, x0, W, theta, sim_sol=False, out=None):
        if out is None:
            out = _fempty((self.n_state, self.n_steps))
        #else:
        #    if not out.shape == (self.n_state, self.n_steps):
        #        raise ValueError("out supplied has incorrect dimensions."
        
        if sim_sol:
            x_state_smooth = out
            x_state_smooth[:, 0] = x0
        else:
            mu_state_smooth = out
            mu_state_smooth[:, 0] = x0

        # forward pass
        # initialize
        self.mu_state_filt[:, 0] = x0
        self.mu_state_pred[:, 0] = x0
        self.var_state_filt[:, :, 0] = 0
        wgt_meas = W  # just renaming for convenience.
        # loop
        for t in range(self.n_eval):
            # kalman filter
            self.ktv.predict(self.mu_state_pred[:, t+1],
                             self.var_state_pred[:, :, t+1],
                             self.mu_state_filt[:, t],
                             self.var_state_filt[:, :, t],
                             self.mu_state,
                             self.wgt_state,
                             self.var_state)
            # model interrogation
            _interrogate_chkrebtii(x_meas=self.x_meas,
                                   var_meas=self.var_meas,
                                   fun=self.ode_fun,
                                   t=self.tmin + (self.tmax-self.tmin)*(t+1)/self.n_eval,
                                   theta=theta,
                                   wgt_meas=wgt_meas,
                                   mu_state_pred=self.mu_state_pred[:, t+1],
                                   var_state_pred=self.var_state_pred[:, :, t+1],
                                   z_state=self.z_state[:, t],
                                   tx_state=self.tx_state,
                                   twgt_meas=self.twgt_meas,
                                   tchol_state=self.tchol_state)
            # rest of kalman filter
            self.ktv.update(self.mu_state_filt[:, t+1],
                            self.var_state_filt[:, :, t+1],
                            self.mu_state_pred[:, t+1],
                            self.var_state_pred[:, :, t+1],
                            self.x_meas,
                            self.mu_meas,
                            wgt_meas,
                            self.var_meas)

        # backward pass
        # initialize
        self.var_state_smooth[:] = self.var_state_filt[:, :, self.n_eval]
        if sim_sol:
            _mvn_sim(x_state_smooth[:, self.n_eval],
                     self.mu_state_filt[:, self.n_eval],
                     self.var_state_smooth,
                     self.z_state[:, self.n_eval],
                     self.tchol_state)
        else:
            mu_state_smooth[:, self.n_eval] = self.mu_state_filt[:, self.n_eval]
        # loop
        for t in range(self.n_eval-1, 0, -1):
            if sim_sol:
                self.ktv.smooth_sim(x_state_smooth[:, t],
                                    x_state_smooth[:, t+1],
                                    self.mu_state_filt[:, t],
                                    self.var_state_filt[:, :, t],
                                    self.mu_state_pred[:, t+1],
                                    self.var_state_pred[:, :, t+1],
                                    self.wgt_state,
                                    self.z_state[:, t+self.n_steps])
            else:
                self.ktv.smooth_mv(mu_state_smooth[:, t],
                                   self.var_state_smooth,
                                   mu_state_smooth[:, t+1],
                                   self.var_state_smooth,
                                   self.mu_state_filt[:, t],
                                   self.var_state_filt[:, :, t],
                                   self.mu_state_pred[:, t+1],
                                   self.var_state_pred[:, :, t+1],
                                   self.wgt_state)

        return out

def KalmanODE(n_state, n_meas, tmin, tmax, n_eval,
              ode_fun, mu_state, wgt_state, var_state, z_state):
    "Create a jitted KalmanODE class."
    spec = [
        ('n_state', intc),
        ('n_meas', intc),
        ('tmin', float64),
        ('tmax', float64),
        ('n_eval', intc),
        ('n_steps', intc),
        ('_mu_state', float64[::1]),
        ('_wgt_state', float64[::1, :]),
        ('_var_state', float64[::1, :]),
        ('_z_state', float64[::1, :]),
        ('ode_fun', typeof(ode_fun)),
        ('mu_state_pred', float64[::1, :]),
        ('var_state_pred', float64[::1, :, :]),
        ('mu_state_filt', float64[::1, :]),
        ('var_state_filt', float64[::1, :, :]),
        ('var_state_smooth', float64[::1, :]),
        ('x_meas', float64[::1]),
        ('mu_meas', float64[::1]),
        ('var_meas', float64[::1, :]),
        ('ktv', kalman_type),
        ('tx_state', float64[::1]),
        ('twgt_meas', float64[::1, :]),
        ('tchol_state', float64[::1, :])
    ]
    jcl = jitclass(spec)
    kalman_cl = jcl(_KalmanODE)
    return kalman_cl(n_state, n_meas, tmin, tmax, n_eval,
                     ode_fun, mu_state, wgt_state, var_state, z_state)
