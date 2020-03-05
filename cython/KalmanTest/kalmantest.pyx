from KalmanTest cimport KalmanTest as CKalmanTest

cdef class KalmanTest:
    cdef CKalmanTest * ktest

    def __cinit__(self, int n_meas, int n_state, int n_steps):
        self.ktest = new CKalmanTest(n_meas, n_state, n_steps)
      
    def __dealloc__(self):
        del self.ktest
    
    def filter_smooth(self,
                      double[::1, :] mu_state_smooths,
                      double[::1, :, :] var_state_smooths,
                      const double[::1] x0_state,
                      const double[::1, :] wgt_state,
                      const double[::1] mu_state, 
                      const double[::1, :] var_state,
                      const double[::1, :] wgt_meas,
                      const double[::1, :] x_meass,
                      const double[::1, :] z_state_sim):
        self.ktest.filter_smooth(& mu_state_smooths[0, 0], & var_state_smooths[0, 0, 0],
                                 & x0_state[0], & wgt_state[0, 0], & mu_state[0],
                                 & var_state[0, 0], & wgt_meas[0, 0], & x_meass[0, 0],
                                 & z_state_sim[0, 0])
        return
        