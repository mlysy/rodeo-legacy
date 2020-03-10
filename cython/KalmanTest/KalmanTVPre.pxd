cdef extern from "KalmanTVPre.h" namespace "KalmanTVPre":
    cdef cppclass KalmanTVPre:
        KalmanTVPre(int, int, int, double*) except +
        void predict(const int cur_step,
                     const double* mu_state,
                     const double* wgt_state,
                     const double* var_state)
        void update(const int cur_step,
                    const double* x_meas,
                    const double* wgt_meas)
        void filter(const int cur_step,
                    const double* mu_state,
                    const double* wgt_state,
                    const double* var_state,
    		            const double* x_meas,
                    const double* wgt_meas)
        void smooth_mv(double* mu_state_smooths,
                       double* var_state_smooths,
                       const int cur_step,
                       const double* wgt_state) 
        void smooth_sim(double* x_state_smooths,
                        const int cur_step,
                        const double* wgt_state,
                        const double* z_states)
        void smooth(double* x_state_smooths,
                    double* mu_state_smooths,
                    double* var_state_smooths,
                    const int cur_step,
                    const double* wgt_state,
                    const double* z_states)
        void state_sim(double* x_state,
                       const double* mu,
                       const double* var,
                       const double* z_state)
        void smooth_update(double* x_state_smooths,
                           double* mu_state_smooths,
                           double* var_state_smooths,
                           const double* z_states)
        void chkrebtii_int(double* x_state,
                           const int cur_step,
                           const double* wgt_meas,
                           const double* z_states)
