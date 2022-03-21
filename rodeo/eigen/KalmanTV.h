/// @file KalmanTV.h

#ifndef KalmanTV_h
#define KalmanTV_h 1

// #undef NDEBUG
// #define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>
#include <iostream>

namespace kalmantv {
  using namespace Eigen;

  /// Time-Varying Kalman Filter and Smoother.
  ///
  /// Model is
  ///
  /// ~~~~
  /// x_n = c_n + Qx_{n-1} + R_n^{1/2} \epsilon_n
  /// y_n = d_n + W x_n + \Sigma_n^{1/2} \eta_n
  /// ~~~~
  ///
  /// where `x_n` is the state variable of dimension `s`, `y_n` is the measurement variable of dimension `m`, and `eps_n ~iid N(0, I_s)` and `eta_n ~iid N(0, I_m)` are noise variables independent of each other.  The `KalmanTV` library uses the following naming conventions to describe the variables above:
  ///
  /// - The state-level variables `x_n`, `c_n`, `Q_n`, `R_n` and `eps_n` are denoted by `state`.
  /// - The measurement-level variables `y_n`, `d_n`, `W_n`, `H_n` and `eta_n` are denoted by `meas`.
  /// - The output variables `x_n` and `y_n` are denoted by `x`.
  /// - The mean vectors `c_n` and `d_n` are denoted by `mu`.
  /// - The variance matrices `R_n` and `H_n` are denoted by `var`.
  /// - The weight matrices `Q_n` and `W_n` are denoted by `wgt`.
  /// - The conditional means and variances are denoted by `mu_n|m = E[x_n | y_0:m]` and `Sigma_n|m = var(x_n | y_0:m)`, and jointly as `theta_n|m = (mu_n|m, Sigma_n|m)`.
  /// - Similarly, `x_n|m` denotes a draw from `p(x_n | x_n+1, y_0:m)`.
  ///
  /// **OLD CONVENTIONS**
  ///
  /// - `x`, `mu`, `var`, `wgt`.
  /// - `_past`: `n-1|n-1` (filter)
  /// - `_pred`: `n|n-1`
  /// - `_filt`: `n|n`
  /// - `_next`: `n+1|N` (smoother)
  /// - `_smooth`: `n|N`
  /// - `mu_n|m = E[x_n | y_0:m]`
  /// - similarly for `Sigma_n|m` and `theta_n|m = (mu_n|m, Sigma_n|m)`.
  /// - `x_n|m` is a draw from `p(x_n | x_n+1, y_0:m)`.
  ///
  /// So for example we have:
  /// - `x_n = xState[n]`
  /// - `W_n = wgt_meas[n]`
  /// - `E[x_n | y_0:n] = mu_state_curr`
  /// - `var(x_n | y_0:N) = var_state_smooth`
  ///
  class KalmanTV {
  private:
    int n_meas_; ///< Number of measurement dimensions.
    int n_state_; ///< Number of state dimensions.
    VectorXd tmu_state_; ///< Temporary storage for mean vector.
    VectorXd tmu_state2_;
    VectorXd tmu_state3_;
    MatrixXd tvar_state_; ///< Temporary storage for variance matrix.
    MatrixXd tvar_state2_;
    MatrixXd tvar_state3_;
    MatrixXd tvar_state4_;
    MatrixXd tchol_state_;
    VectorXd tmu_meas_;
    MatrixXd twgt_meas_;
    MatrixXd twgt_meas2_;
    MatrixXd tvar_meas_;
    LLT<MatrixXd> tllt_meas_;
    LLT<MatrixXd> tllt_state_;
  public:
    /// Typedefs
    typedef Ref<VectorXd> RefVectorXd;
    typedef const Ref<const VectorXd> cRefVectorXd;
    typedef Ref<MatrixXd> RefMatrixXd;
    typedef const Ref<const MatrixXd> cRefMatrixXd;
    typedef Map<VectorXd> MapVectorXd;
    typedef Map<const VectorXd> cMapVectorXd;
    typedef Map<MatrixXd> MapMatrixXd;
    typedef Map<const MatrixXd> cMapMatrixXd;
    /// Default constructor.
    KalmanTV() {}
    /// Constructor with arguments.
    KalmanTV(int n_meas, int n_state);
    /// Perform internal memory allocation for KalmanTV object with given dimensions.
    void set_dims(int n_meas, int n_state);
    /// Perform one prediction step of the Kalman filter.
    ///
    /// Calculates `theta_n|n-1` from `theta_n-1|n-1`.
    void predict(RefVectorXd mu_state_pred,
                 RefMatrixXd var_state_pred,
                 cRefVectorXd& mu_state_past,
                 cRefMatrixXd& var_state_past,
                 cRefVectorXd& mu_state,
                 cRefMatrixXd& wgt_state,
                 cRefMatrixXd& var_state);
    /// Perform one update step of the Kalman filter.
    ///
    /// Calculates `theta_n|n` from `theta_n|n-1`.
    void update(RefVectorXd mu_state_filt,
                RefMatrixXd var_state_filt,
                cRefVectorXd& mu_state_pred,
                cRefMatrixXd& var_state_pred,
                cRefVectorXd& x_meas,
                cRefVectorXd& mu_meas,
                cRefMatrixXd& wgt_meas,
                cRefMatrixXd& var_meas);
    /// Perform one step of the Kalman filter.
    /// Combines `predict` and `update` steps to get `theta_n|n` from `theta_n-1|n-1`.
    void filter(RefVectorXd mu_state_pred,
    		        RefMatrixXd var_state_pred,
		            RefVectorXd mu_state_filt,
    		        RefMatrixXd var_state_filt,
    		        cRefVectorXd& mu_state_past,
    		        cRefMatrixXd& var_state_past,
                    cRefVectorXd& mu_state,
                    cRefMatrixXd& wgt_state,
                    cRefMatrixXd& var_state,
                    cRefVectorXd& x_meas,
                    cRefVectorXd& mu_meas,
                    cRefMatrixXd& wgt_meas,
                    cRefMatrixXd& var_meas);
    /// Perform one step of the Kalman mean/variance smoother.
    ///
    /// Calculates `theta_n|N` from `theta_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    void smooth_mv(RefVectorXd mu_state_smooth,
                   RefMatrixXd var_state_smooth,
                   cRefVectorXd& mu_state_next,
                   cRefMatrixXd& var_state_next,
                   cRefVectorXd& mu_state_filt,
                   cRefMatrixXd& var_state_filt,
                   cRefVectorXd& mu_state_pred,
                   cRefMatrixXd& var_state_pred,
                   cRefMatrixXd& wgt_state);
    /// Perform one step of the Kalman sampling smoother.
    ///
    /// Calculates a draw `x_n|N` from `x_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    void smooth_sim(RefVectorXd xState_smooth,
                    cRefVectorXd& xState_next,
                    cRefVectorXd& mu_state_filt,
                    cRefMatrixXd& var_state_filt,
                    cRefVectorXd& mu_state_pred,
                    cRefMatrixXd& var_state_pred,
                    cRefMatrixXd& wgt_state,
                    cRefVectorXd& z_state);
    /// Perfrom one step of both Kalman mean/variance and sampling smoothers.
    void smooth(RefVectorXd xState_smooth,
                RefVectorXd mu_state_smooth,
                RefMatrixXd var_state_smooth,
                cRefVectorXd& xState_next,
                cRefVectorXd& mu_state_next,
                cRefMatrixXd& var_state_next,
                cRefVectorXd& mu_state_filt,
                cRefMatrixXd& var_state_filt,
                cRefVectorXd& mu_state_pred,
                cRefMatrixXd& var_state_pred,
                cRefMatrixXd& wgt_state,
                cRefVectorXd& z_state);
    /// Simulate a random state given the mean and variance.
    void state_sim(RefVectorXd x_state,
                   cRefVectorXd& mu_state,
                   cRefMatrixXd& var_State,
                   cRefVectorXd& z_state);
    /// Calculates `theta^X_n|n-1` from `theta_n-1|n-1`.
    void forecast(RefVectorXd mu_fore,
                  RefMatrixXd var_fore,
                  cRefVectorXd& mu_state_pred,
                  cRefMatrixXd& var_state_pred,
                  cRefVectorXd& mu_meas,
                  cRefMatrixXd& wgt_meas,
                  cRefMatrixXd& var_meas);
    // void printX() {
    //   int n = 2;
    //   int p = 3;
    //   MatrixXd X = MatrixXd::Constant(n, p, 3.14);
    //   std::cout << "X =\n" << X << std::endl;
    //   return;
    // }
  };

  /// @param[in] n_meas Number of measurement variables.
  /// @param[in] n_state Number of state variables.
  inline KalmanTV::KalmanTV(int n_meas, int n_state) {
    // Eigen::internal::set_is_malloc_allowed(true);
    set_dims(n_meas, n_state);
    // Eigen::internal::set_is_malloc_allowed(false);
  }

  /// @param[in] n_meas Number of measurement variables.
  /// @param[in] n_state Number of state variables.
  inline void KalmanTV::set_dims(int n_meas, int n_state) {
        // problem dimensions
    n_meas_ = n_meas;
    n_state_ = n_state;
    // temporary storage
    tmu_state_ = VectorXd::Zero(n_state_);
    tmu_state2_ = VectorXd::Zero(n_state_);
    // tmu_state3_ = VectorXd::Zero(n_state_);
    tvar_state_ = MatrixXd::Identity(n_state_, n_state_);
    tvar_state2_ = MatrixXd::Identity(n_state_, n_state_);
    tvar_state3_ = MatrixXd::Identity(n_state_, n_state_);
    tvar_state4_ = MatrixXd::Identity(n_state_, n_state_);
    tchol_state_ = MatrixXd::Identity(n_state_, n_state_);
    tmu_meas_ = VectorXd::Zero(n_meas_);
    tvar_meas_ = MatrixXd::Identity(n_meas_, n_meas_);
    twgt_meas_ = MatrixXd::Zero(n_meas_, n_state_);
    twgt_meas2_ = MatrixXd::Zero(n_meas_, n_state_);
    // cholesky solvers
    tllt_meas_.compute(MatrixXd::Identity(n_meas_, n_meas_));
    tllt_state_.compute(MatrixXd::Identity(n_state_, n_state_));
    return;
  }

  /// @param[out] mu_state_pred Predicted state mean `mu_n|n-1`.
  /// @param[out] var_state_pred Predicted state variance `Sigma_n|n-1`.
  /// @param[in] mu_state_past Previous state mean `mu_n-1|n-1`.
  /// @param[in] var_state_past Previous state variance `Sigma_n-1|n-1`.
  /// @param[in] mu_state Current state mean `c_n`.
  /// @param[in] wgt_state Current state transition matrix `Q_n`.
  /// @param[in] var_state Current state variance `R_n`.
  inline void KalmanTV::predict(RefVectorXd mu_state_pred,
                                RefMatrixXd var_state_pred,
                                cRefVectorXd& mu_state_past,
                                cRefMatrixXd& var_state_past,
                                cRefVectorXd& mu_state,
                                cRefMatrixXd& wgt_state,
                                cRefMatrixXd& var_state) {
    mu_state_pred.noalias() = wgt_state * mu_state_past;
    mu_state_pred += mu_state;
    // // need to assign to temporary for matrix triple product
    tvar_state_.noalias() = wgt_state * var_state_past;
    var_state_pred.noalias() = tvar_state_ * wgt_state.adjoint();
    var_state_pred += var_state;
    return;
  }

  /// @param[out] mu_state_filt Current state mean `mu_n|n`.
  /// @param[out] var_state_filt Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_pred Predicted state mean `mu_n|n-1`.
  /// @param[in] var_state_pred Predicted state variance `Sigma_n|n-1`.
  /// @param[in] x_meas Current measure `y_n`.
  /// @param[in] mu_meas Current measure mean `d_n`.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] var_meas Current measure variance `H_n`.
  inline void KalmanTV::update(RefVectorXd mu_state_filt,
                               RefMatrixXd var_state_filt,
                               cRefVectorXd& mu_state_pred,
                               cRefMatrixXd& var_state_pred,
                               cRefVectorXd& x_meas,
                               cRefVectorXd& mu_meas,
                               cRefMatrixXd& wgt_meas,
                               cRefMatrixXd& var_meas) {
    tmu_meas_.noalias() = wgt_meas * mu_state_pred;
    tmu_meas_ += mu_meas; // n_meas
    // std::cout << "tmu_meas_ = " << tmu_meas_ << std::endl;
    twgt_meas_.noalias() = wgt_meas * var_state_pred; // n_meas x n_state
    // std::cout << "twgt_meas_ = " << twgt_meas_ << std::endl;
    tvar_meas_.noalias() = twgt_meas_ * wgt_meas.adjoint();
    tvar_meas_ += var_meas; // n_meas x n_meas
    // std::cout << "tvar_meas_ = " << tvar_meas_ << std::endl;
    tllt_meas_.compute(tvar_meas_);
    twgt_meas2_.noalias() = twgt_meas_;
    // std::cout << "twgt_meas2_ = " << twgt_meas2_ << std::endl;
    tllt_meas_.solveInPlace(twgt_meas_);
    // std::cout << "twgt_meas_ = " << twgt_meas_ << std::endl;
    tmu_meas_.noalias() = x_meas - tmu_meas_;
    // std::cout << "tmu_meas_ = " << tmu_meas_ << std::endl;
    // QUESTION: is it wasteful to call adjoint() twice?
    // does it require a temporary assignment?
    mu_state_filt.noalias() = twgt_meas_.adjoint() * tmu_meas_ ;
    mu_state_filt += mu_state_pred;
    // std::cout << "mu_state_filt = " << mu_state_filt << std::endl;
    var_state_filt.noalias() = twgt_meas_.adjoint() * twgt_meas2_;
    // std::cout << "var_state_filt = " << var_state_filt << std::endl;
    var_state_filt.noalias() = var_state_pred - var_state_filt;
    // std::cout << "var_state_filt = " << var_state_filt << std::endl;
    return;
  }
  
  /// @param[out] mu_state_pred Predicted state mean `mu_n|n-1`.
  /// @param[out] var_state_pred Predicted state variance `Sigma_n|n-1`.
  /// @param[out] mu_state_filt Current state mean `mu_n|n`.
  /// @param[out] var_state_filt Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_past Previous state mean `mu_n-1|n-1`.
  /// @param[in] var_state_past Previous state variance `Sigma_n-1|n-1`.
  /// @param[in] mu_state Current state mean `c_n`.
  /// @param[in] wgt_state Current state transition matrix `Q_n`.
  /// @param[in] var_state Current state variance `R_n`.
  /// @param[in] x_meas Current measure `y_n`.
  /// @param[in] mu_meas Current measure mean `d_n`.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] var_meas Current measure variance `H_n`.
  inline void KalmanTV::filter(RefVectorXd mu_state_pred,
			       RefMatrixXd var_state_pred,
			       RefVectorXd mu_state_filt,
			       RefMatrixXd var_state_filt,
			       cRefVectorXd& mu_state_past,
			       cRefMatrixXd& var_state_past,
			       cRefVectorXd& mu_state,
			       cRefMatrixXd& wgt_state,
			       cRefMatrixXd& var_state,
			       cRefVectorXd& x_meas,
			       cRefVectorXd& mu_meas,
			       cRefMatrixXd& wgt_meas,
			       cRefMatrixXd& var_meas) {
    predict(mu_state_pred, var_state_pred,
            mu_state_past, var_state_past,
            mu_state, wgt_state, var_state);
    update(mu_state_filt, var_state_filt,
           mu_state_pred, var_state_pred,
           x_meas, mu_meas, wgt_meas, var_meas);
    return;    
  }
  
  /// @param[out] mu_state_smooth Smoothed state mean `mu_n|N`.
  /// @param[out] var_state_smooth Smoothed state variance `Sigma_n|N`.
  /// @param[in] mu_state_next Next smoothed state mean `mu_n+1|N`.
  /// @param[in] var_state_next Next smoothed state variance `Sigma_n+1|N`.
  /// @param[in] mu_state_filt Current state mean `mu_n|n`.
  /// @param[in] var_state_filt Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_pred Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_pred Predicted state variance `Sigma_n+1|n`.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  inline void KalmanTV::smooth_mv(RefVectorXd mu_state_smooth,
                                  RefMatrixXd var_state_smooth,
                                  cRefVectorXd& mu_state_next,
                                  cRefMatrixXd& var_state_next,
                                  cRefVectorXd& mu_state_filt,
                                  cRefMatrixXd& var_state_filt,
                                  cRefVectorXd& mu_state_pred,
                                  cRefMatrixXd& var_state_pred,
                                  cRefMatrixXd& wgt_state) {
    tvar_state_.noalias() = wgt_state * var_state_filt.adjoint();  
    tllt_state_.compute(var_state_pred); 
    tllt_state_.solveInPlace(tvar_state_); // equivalent to var_state_temp_tilde
    tmu_state_.noalias() = mu_state_next - mu_state_pred;
    tvar_state2_.noalias() = var_state_next - var_state_pred;
    tvar_state3_.noalias() = tvar_state_.adjoint() * tvar_state2_; 
    var_state_smooth.noalias() = tvar_state3_ * tvar_state_;          
    mu_state_smooth.noalias() = tvar_state_.adjoint() * tmu_state_;
    mu_state_smooth += mu_state_filt;
    var_state_smooth += var_state_filt;
    return;
  }
  
  /// @param[out] xState_smooth Smoothed state `X_n`.
  /// @param[in] mu_state_next Next smoothed state mean `mu_n+1|N`.
  /// @param[in] var_state_next Next smoothed state variance `Sigma_n+1|N`.
  /// @param[in] mu_state_filt Current state mean `mu_n|n`.
  /// @param[in] var_state_filt Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_pred Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_pred Predicted state variance `Sigma_n+1|n`.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] z_state Random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTV::smooth_sim(RefVectorXd xState_smooth,
                                   cRefVectorXd& xState_next,
                                   cRefVectorXd& mu_state_filt,
                                   cRefMatrixXd& var_state_filt,
                                   cRefVectorXd& mu_state_pred,
                                   cRefMatrixXd& var_state_pred,
                                   cRefMatrixXd& wgt_state,
                                   cRefVectorXd& z_state) {
    tvar_state_.noalias() = wgt_state * var_state_filt.adjoint();
    tvar_state2_.noalias() = tvar_state_; // equivalent to var_state_temp
    tllt_state_.compute(var_state_pred); 
    tllt_state_.solveInPlace(tvar_state_); // equivalent to var_state_temp_tilde
    tmu_state_.noalias() = xState_next - mu_state_pred;
    // std::cout << "tvar_state_ = " << tvar_state_ << std::endl;
    tmu_state2_.noalias() = tvar_state_.adjoint() * tmu_state_;
    tmu_state2_ += mu_state_filt;
    // mu_state_sim.noalias() = tmu_state2_;
    // std::cout << "tmu_state_= " << tmu_state_ << std::endl;
    tvar_state3_.noalias() = tvar_state_.adjoint() * tvar_state2_;
    tvar_state3_.noalias() = var_state_filt - tvar_state3_;
    // tvar_state4_.noalias() = tvar_state3_ * tvar_state3_.adjoint(); // only for testing (requires semi-positive)
    // var_state_sim.noalias() = tvar_state4_; // testing
    // Cholesky
    state_sim(xState_smooth, tmu_state2_,
              tvar_state3_, z_state);
/*     tllt_state2_.compute(tvar_state4_); // use tvar_state3_ in the algorithm
    tchol_state_ = tllt_state2_.matrixL();
    xState_smooth.noalias() = tchol_state_ * z_state;
    xState_smooth += tmu_state2_; */
    return;
  }

  /// @param[out] xState_smooth Smoothed state `X_n`.
  /// @param[out] mu_state_smooth Smoothed state mean `mu_n|N`.
  /// @param[out] var_state_smooth Smoothed state variance `Sigma_n|N`.
  /// @param[in] mu_state_next Next smoothed state mean `mu_n+1|N`.
  /// @param[in] var_state_next Next smoothed state variance `Sigma_n+1|N`.
  /// @param[in] mu_state_filt Current state mean `mu_n|n`.
  /// @param[in] var_state_filt Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_pred Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_pred Predicted state variance `Sigma_n+1|n`.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] z_state Random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTV::smooth(RefVectorXd xState_smooth,
                               RefVectorXd mu_state_smooth,
                               RefMatrixXd var_state_smooth,
                               cRefVectorXd& xState_next,
                               cRefVectorXd& mu_state_next,
                               cRefMatrixXd& var_state_next,
                               cRefVectorXd& mu_state_filt,
                               cRefMatrixXd& var_state_filt,
                               cRefVectorXd& mu_state_pred,
                               cRefMatrixXd& var_state_pred,
                               cRefMatrixXd& wgt_state,
                               cRefVectorXd& z_state) {
    smooth_mv(mu_state_smooth, var_state_smooth,
              mu_state_next, var_state_next,
              mu_state_filt, var_state_filt,
              mu_state_pred, var_state_pred, wgt_state);
    smooth_sim(xState_smooth,xState_next, 
               mu_state_filt, var_state_filt,
               mu_state_pred, var_state_pred, 
               wgt_state, z_state);
    return;
  }
 
  /// @param[out] x_state Simulated state.
  /// @param[in] mu_state State mean.
  /// @param[in] var_state State variance.
  /// @param[in] z_state Random draws from `N(0,1)` for simulating the state.
  inline void KalmanTV::state_sim(RefVectorXd x_state,
                                  cRefVectorXd& mu_state,
                                  cRefMatrixXd& var_state,
                                  cRefVectorXd& z_state) {
    tllt_state_.compute(var_state);
    tchol_state_ = tllt_state_.matrixL();
    x_state.noalias() = tchol_state_ * z_state;
    x_state += mu_state;
    return;
  }

  /// @param[out] mu_fore Predicted measurement mean `mu^X_n|n-1`
  /// @param[out] var_fore Predicted measurement variance `Sigma^XX_n|n-1`
  /// @param[in] mu_state_pred Predicted state mean `mu_n|n-1`.
  /// @param[in] var_state_pred Predicted state variance `Sigma_n|n-1`.
  /// @param[in] mu_meas Current measure mean `d_n`.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] var_meas Current measure variance `H_n`.
  inline void KalmanTV::forecast(RefVectorXd mu_fore,
                                 RefMatrixXd var_fore,
                                 cRefVectorXd& mu_state_pred,
                                 cRefMatrixXd& var_state_pred,
                                 cRefVectorXd& mu_meas,
                                 cRefMatrixXd& wgt_meas,
                                 cRefMatrixXd& var_meas) {
    mu_fore.noalias() = wgt_meas * mu_state_pred;
    mu_fore += mu_meas;
    // // need to assign to temporary for matrix triple product
    twgt_meas_.noalias() = wgt_meas * var_state_pred;
    var_fore.noalias() = twgt_meas_ * wgt_meas.adjoint();
    var_fore += var_meas;
    return;
  }

} // end namespace KalmanTV


#endif
