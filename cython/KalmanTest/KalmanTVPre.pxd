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
        void smooth_mv(const int cur_step,
                       double* mu_state_smooth,
                       double* var_state_smooth,
                       const double* mu_state_next,
                       const double* var_state_next,
                       const double* wgt_state) 
        void smooth_sim(const int cur_step,
                        double* x_state_smooth,
                        const double* x_state_next,
                        const double* wgt_state,
                        const double* z_state)
        void smooth(const int cur_step,
                    double* x_state_smooth,
                    double* mu_state_smooth,
                    double* var_state_smooth,
                    const double* x_state_next,
                    const double* mu_state_next,
                    const double* var_state_next,
                    const double* wgt_state,
                    const double* z_state)
        void state_sim(double* x_state,
                       const double* mu,
                       const double* var,
                       const double* z_state)
        void smooth_update(double* x_state_smooth,
                           double* mu_state_smooth,
                           double* var_state_smooth,
                           const double* z_state)
