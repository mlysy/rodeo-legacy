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
                  const int cur_step,
                  double[::1] mu_state_smooth,
                  double[::1, :] var_state_smooth,
                  const double[::1] mu_state_next,
                  const double[::1, :] var_state_next,
                  const double[::1, :] wgt_state):
        self.ktvpre.smooth_mv(cur_step, & mu_state_smooth[0], & var_state_smooth[0, 0],
                              & mu_state_next[0], & var_state_next[0, 0], & wgt_state[0, 0])
        return

    def smooth_sim(self,
                   const int cur_step,
                   double[::1] x_state_smooth,
                   const double[::1] x_state_next,
                   const double[::1, :] wgt_state,
                   const double[::1] z_state):
        self.ktvpre.smooth_sim(cur_step, & x_state_smooth[0], & x_state_next[0],
                               & wgt_state[0, 0], & z_state[0])
        return

    def smooth(self,
               const int cur_step,
               double[::1] x_state_smooth,
               double[::1] mu_state_smooth,
               double[::1, :] var_state_smooth,
               const double[::1] x_state_next,
               const double[::1] mu_state_next,
               const double[::1, :] var_state_next,
               const double[::1, :] wgt_state,
               const double[::1] z_state):
        self.ktvpre.smooth(cur_step, & x_state_smooth[0], & mu_state_smooth[0],
                           & var_state_smooth[0, 0], & x_state_next[0],
                           & mu_state_next[0], & var_state_next[0, 0],
                           & wgt_state[0, 0], & z_state[0])
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
                      double[::1] x_state_smooth,
                      double[::1] mu_state_smooth,
                      double[::1, :] var_state_smooth,
                      double[::1] z_state):
        self.ktvpre.smooth_update(& x_state_smooth[0], & mu_state_smooth[0],
                                  & var_state_smooth[0, 0], & z_state[0])
        return
                      