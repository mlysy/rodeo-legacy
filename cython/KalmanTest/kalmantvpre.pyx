from KalmanTVPre cimport KalmanTVPre as CKalmanTVPre

cdef class KalmanTVPre:
    cdef CKalmanTVPre * ktvpre

    def __cinit__(self, int n_meas, int n_state, int n_steps, double[::1] x0_state):
        self.ktvpre = new CKalmanTVPre(n_meas, n_state, n_steps, & x0_state[0])

    def __dealloc__(self):
        del self.ktvpre

    def predict(self,
                const int cur_step,
                const double[::1] mu_state,
                const double[::1, :] wgt_state,
                const double[::1, :] var_state):
        self.ktvpre.predict(cur_step, & mu_state[0], & wgt_state[0, 0], & var_state[0, 0])
        return

    def update(self,
               const int cur_step,
               const double[::1] x_meas,
               const double[::1, :] wgt_meas):
        self.ktvpre.update(cur_step, & x_meas[0], & wgt_meas[0, 0])
        return
    
    def filter(self,
               const int cur_step,
               const double[::1] mu_state,
               const double[::1, :] wgt_state,
               const double[::1, :] var_state,
               const double[::1] x_meas,
               const double[::1, :] wgt_meas):
        self.ktvpre.filter(cur_step, & mu_state[0], & wgt_state[0, 0], 
                           & var_state[0, 0], & x_meas[0], & wgt_meas[0, 0],)
        return

    def smooth_mv(self,
                  double[::1, :] mu_state_smooths,
                  double[::1, :, :] var_state_smooths,
                  const int cur_step,
                  const double[::1, :] wgt_state):
        self.ktvpre.smooth_mv(& mu_state_smooths[0, 0], & var_state_smooths[0, 0, 0], 
                              cur_step, & wgt_state[0, 0])
        return

    def smooth_sim(self,
                   double[::1, :] x_state_smooths,
                   const int cur_step,
                   const double[::1, :] wgt_state,
                   const double[::1, :] z_states):
        self.ktvpre.smooth_sim(& x_state_smooths[0, 0], cur_step,
                               & wgt_state[0, 0], & z_states[0, 0])
        return

    def smooth(self,
               double[::1, :] x_state_smooths,
               double[::1, :] mu_state_smooths,
               double[::1, :, :] var_state_smooths,
               const int cur_step,
               const double[::1, :] wgt_state,
               const double[::1, :] z_states):
        self.ktvpre.smooth(& x_state_smooths[0, 0], & mu_state_smooths[0, 0],
                           & var_state_smooths[0, 0, 0], cur_step,
                           & wgt_state[0, 0], & z_states[0, 0])
        return

    def state_sim(self,
                  double[::1] x_state,
                  double[::1] mu,
                  double[::1, :] var,
                  double[::1] z_state):
        self.ktvpre.state_sim(& x_state[0], & mu[0],
                              & var[0, 0], & z_state[0])
        return

    def smooth_update(self,
                      double[::1, :] x_state_smooths,
                      double[::1, :] mu_state_smooths,
                      double[::1, :, :] var_state_smooths,
                      double[::1, :] z_states):
        self.ktvpre.smooth_update(& x_state_smooths[0, 0], & mu_state_smooths[0, 0],
                                  & var_state_smooths[0, 0, 0], & z_states[0, 0])
        return
    
    def chkrebtii_int(self,
                      double[::1] x_state,
                      const int cur_step,
                      const double[::1, :] wgt_meas,
                      const double[::1, :] z_states):
        self.ktvpre.chkrebtii_int(& x_state[0], cur_step,
                                  & wgt_meas[0, 0], & z_states[0, 0])
        return
