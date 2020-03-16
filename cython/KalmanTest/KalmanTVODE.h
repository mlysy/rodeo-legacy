/// @file KalmanTVODE.h

#ifndef KalmanTVODE_h
#define KalmanTVODE_h 1

// #undef NDEBUG
// #define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>
#include <iostream>
#include "KalmanTV.h"

namespace KalmanTVODE {
  using namespace Eigen;

  class KalmanTVODE : public KalmanTV::KalmanTV {
  private:
    int n_meas_; ///< Number of measurement dimensions.
    int n_state_; ///< Number of state dimensions.
    int n_steps_; ///< Number of time steps to evaluate.
    VectorXd x0_state_; ///< Initial x0 value of ODE.
    VectorXd mu_meas; ///< C++ level memory allocation.
    MatrixXd var_meas;
    MatrixXd mu_state_filts;
    MatrixXd var_state_filts;
    MatrixXd mu_state_preds;
    MatrixXd var_state_preds;
    MatrixXd twgt_meas_;
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
    /// Constructor
    KalmanTVODE(int n_meas, int n_state, int n_steps, double* x0_state);
    /// Perform one prediction step of the Kalman filter.
    ///
    /// Calculates `theta_n|n-1` from `theta_n-1|n-1`.
    void predict(const int cur_step,
                 cRefVectorXd& mu_state,
                 cRefMatrixXd& wgt_state,
                 cRefMatrixXd& var_state);
    /// Raw buffer equivalent.
    void predict(const int cur_step,
                 const double* mu_state,
                 const double* wgt_state,
                 const double* var_state);
    /// Perform one update step of the Kalman filter.
    ///
    /// Calculates `theta_n|n` from `theta_n|n-1`.
    void update(const int cur_step,
                cRefVectorXd& x_meas,
                cRefMatrixXd& wgt_meas);
    /// Raw buffer equivalent.
    void update(const int cur_step,
                const double* x_meas,
                const double* wgt_meas);
    /// Perform one step of the Kalman filter.
    /// Combines `predict` and `update` steps to get `theta_n|n` from `theta_n-1|n-1`.
    void filter(const int cur_step,
                cRefVectorXd& mu_state,
                cRefMatrixXd& wgt_state,
                cRefMatrixXd& var_state,
                cRefVectorXd& x_meas,
                cRefMatrixXd& wgt_meas);
    /// Raw buffer equivalent.
    void filter(const int cur_step,
                const double* mu_state,
                const double* wgt_state,
                const double* var_state,
                const double* x_meas,
                const double* wgt_meas);
    /// Perform one step of the Kalman mean/variance smoother.
    ///
    /// Calculates `theta_n|N` from `theta_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    void smooth_mv(RefMatrixXd mu_state_smooths,
                   RefMatrixXd var_state_smooths,
                   const int cur_step,
                   cRefMatrixXd& wgt_state);
    /// Raw buffer equivalent.
    void smooth_mv(double* mu_state_smooths,
                   double* var_state_smooths,
                   const int cur_step,
                   const double* wgt_state);
    /// Perform one step of the Kalman sampling smoother.
    ///
    /// Calculates a draw `x_n|N` from `x_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`.  **Is the indexing correct?**
    void smooth_sim(RefMatrixXd x_state_smooths,
                    const int cur_step,
                    cRefMatrixXd& wgt_state,
                    cRefMatrixXd& z_states);
    /// Raw buffer equivalent.                    
    void smooth_sim(double* x_state_smooths,
                    const int cur_step,
                    const double* wgt_state,
                    const double* z_states);    
    /// Perfrom one step of both Kalman mean/variance and sampling smoothers.
    void smooth(RefMatrixXd x_state_smooths,
                RefMatrixXd mu_state_smooths,
                RefMatrixXd var_state_smooths,
                const int cur_step,
                cRefMatrixXd& wgt_state,
                cRefMatrixXd& z_states);
    /// Raw buffer equivalent.                    
    void smooth(double* x_state_smooths,
                double* mu_state_smooths,
                double* var_state_smooths,
                const int cur_step,
                const double* wgt_state,
                const double* z_states);
    /// Performes the smoothing steps.
    void smooth_update(RefMatrixXd x_state_smooths,
                       RefMatrixXd mu_state_smooths,
                       RefMatrixXd var_state_smooths,
                       cRefMatrixXd& wgt_state,
                       cRefMatrixXd& z_states,
                       const bool smooth_mv,
                       const bool smooth_sim);
    /// Raw buffer equivalent.
    void smooth_update(double* x_state_smooths,
                       double* mu_state_smooths,
                       double* var_state_smooths,
                       const double* wgt_state,
                       const double* z_states,
                       const bool smooth_mv,
                       const bool smooth_sim);
    /// Perform one step of chkrebtii interrogation.
    void chkrebtii_int(RefVectorXd x_state,
                       const int cur_step,
                       cRefMatrixXd& wgt_meas,
                       cRefMatrixXd& z_states);
    /// Raw buffer equivalent.
    void chkrebtii_int(double* x_state,
                       const int cur_step,
                       const double* wgt_meas,
                       const double* z_states);
  };

  /// @param[in] n_meas Number of measurement variables.
  /// @param[in] n_state Number of state variables.
  /// @param[in] n_steps Number of time steps to evaluate.
  /// @param[in] x0_state Initial value of ODE.
  inline KalmanTVODE::KalmanTVODE(int n_meas, int n_state, int n_steps, double* x0_state) : KalmanTV(n_meas, n_state) {
    // problem dimensions
    n_meas_ = n_meas;
    n_state_ = n_state;
    n_steps_ = n_steps; 
    /// initial x0 value
    MapVectorXd x0_state_(x0_state, n_state_);
    // memory allocation
    mu_meas = VectorXd::Zero(n_meas_);
    var_meas = MatrixXd::Zero(n_meas_, n_meas_);
    mu_state_filts = MatrixXd::Zero(n_state_, n_steps_);
    var_state_filts = MatrixXd::Zero(n_state_, n_state_*n_steps_);
    mu_state_preds = MatrixXd::Zero(n_state_, n_steps_);
    var_state_preds = MatrixXd::Zero(n_state_, n_state_*n_steps_);
    /// initialize mu_state with x0_state
    mu_state_filts.col(0) = x0_state_;
    mu_state_preds.col(0) = mu_state_filts.col(0);
    // temporary storage
    twgt_meas_ = MatrixXd::Zero(n_meas_, n_state_);
  }

  /// @param[out] mu_state_preds Predicted state mean `mu_n|n-1`.
  /// @param[out] var_state_preds Predicted state variance `Sigma_n|n-1`.
  /// @param[in] mu_state_filts Previous state mean `mu_n-1|n-1`.
  /// @param[in] var_state_filts Previous state variance `Sigma_n-1|n-1`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] mu_state Current state mean `c_n`.
  /// @param[in] wgt_state Current state transition matrix `T_n`.
  /// @param[in] var_state Current state variance `R_n`.
  inline void KalmanTVODE::predict(const int cur_step,
                                   cRefVectorXd& mu_state,
                                   cRefMatrixXd& wgt_state,
                                   cRefMatrixXd& var_state) {
    KalmanTV::predict(mu_state_preds.col(cur_step+1),
                      var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                      mu_state_filts.col(cur_step),
                      var_state_filts.block(0, n_state_*cur_step, n_state_, n_state_),
                      mu_state,
                      wgt_state,
                      var_state);
    return;
  }
  /// @note Arguments updated to be identical to those with `Eigen` types, so we don't need to re-document.
  inline void KalmanTVODE::predict(const int cur_step,
                                   const double* mu_state,
                                   const double* wgt_state,
                                   const double* var_state) {
    cMapVectorXd mu_state_(mu_state, n_state_);
    cMapMatrixXd wgt_state_(wgt_state, n_state_, n_state_);
    cMapMatrixXd var_state_(var_state, n_state_, n_state_);
    predict(cur_step, mu_state_, wgt_state_, var_state_);
    return;
  }

  /// @param[out] mu_state_filts Current state mean `mu_n|n`.
  /// @param[out] var_state_filts Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_preds Predicted state mean `mu_n|n-1`.
  /// @param[in] var_state_preds Predicted state variance `Sigma_n|n-1`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] x_meas Current measure `y_n`.
  /// @param[in] mu_meas Current measure mean `d_n`.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] var_meas Current measure variance `H_n`.
  inline void KalmanTVODE::update(const int cur_step,
                                  cRefVectorXd& x_meas,
                                  cRefMatrixXd& wgt_meas) {
    KalmanTV::update(mu_state_filts.col(cur_step+1),
                     var_state_filts.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                     mu_state_preds.col(cur_step+1),
                     var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                     x_meas,
                     mu_meas,
                     wgt_meas,
                     var_meas);
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTVODE::update(const int cur_step,
                                  const double* x_meas,
                                  const double* wgt_meas) {
    cMapVectorXd x_meas_(x_meas, n_meas_);
    cMapMatrixXd wgt_meas_(wgt_meas, n_meas_, n_state_);
    update(cur_step, x_meas_, wgt_meas_);
    return;    
  }
  
  /// @param[out] mu_state_preds Predicted state mean `mu_n|n-1`.
  /// @param[out] var_state_preds Predicted state variance `Sigma_n|n-1`.
  /// @param[out] mu_state_filts Current state mean `mu_n|n`.
  /// @param[out] var_state_filts Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_preds Previous state mean `mu_n-1|n-1`.
  /// @param[in] var_state_preds Previous state variance `Sigma_n-1|n-1`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] mu_state Current state mean `c_n`.
  /// @param[in] wgt_state Current state transition matrix `T_n`.
  /// @param[in] var_state Current state variance `R_n`.
  /// @param[in] x_meas Current measure `y_n`.
  /// @param[in] mu_meas Current measure mean `d_n`.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] var_meas Current measure variance `H_n`.
  inline void KalmanTVODE::filter(const int cur_step,
                                  cRefVectorXd& mu_state,
                                  cRefMatrixXd& wgt_state,
                                  cRefMatrixXd& var_state,
                                  cRefVectorXd& x_meas,
                                  cRefMatrixXd& wgt_meas) {
    predict(cur_step, mu_state, wgt_state, var_state);
    update(cur_step, x_meas, wgt_meas);
    return;    
  }
  /// Raw buffer equivalent.
  inline void KalmanTVODE::filter(const int cur_step,
                                  const double* mu_state,
                                  const double* wgt_state,
                                  const double* var_state,
                                  const double* x_meas,
                                  const double* wgt_meas) {
    predict(cur_step, mu_state, wgt_state, var_state);
    update(cur_step, x_meas, wgt_meas);
    return;    
  }
  
  /// @param[out] mu_state_smooths Smoothed state mean `mu_n|N`.
  /// @param[out] var_state_smooths Smoothed state variance `Sigma_n|N`.
  /// @param[in] mu_state_smooths Next smoothed state mean `mu_n+1|N`.
  /// @param[in] var_state_smooths Next smoothed state variance `Sigma_n+1|N`.
  /// @param[in] mu_state_filts Current state mean `mu_n|n`.
  /// @param[in] var_state_filts Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_preds Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_preds Predicted state variance `Sigma_n+1|n`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] wgt_state Current state transition matrix `T_n`.
  inline void KalmanTVODE::smooth_mv(RefMatrixXd mu_state_smooths,
                                     RefMatrixXd var_state_smooths,
                                     const int cur_step,
                                     cRefMatrixXd& wgt_state) {
    KalmanTV::smooth_mv(mu_state_smooths.col(cur_step),
                        var_state_smooths.block(0, n_state_*cur_step, n_state_, n_state_),
                        mu_state_smooths.col(cur_step+1),
                        var_state_smooths.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                        mu_state_filts.col(cur_step),
                        var_state_filts.block(0, n_state_*cur_step, n_state_, n_state_),
                        mu_state_preds.col(cur_step+1),
                        var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                        wgt_state);
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTVODE::smooth_mv(double* mu_state_smooths,
                                     double* var_state_smooths,
                                     const int cur_step,
                                     const double* wgt_state) {
    MapMatrixXd mu_state_smooths_(mu_state_smooths, n_state_, n_steps_);
    MapMatrixXd var_state_smooths_(var_state_smooths, n_state_, n_state_*n_steps_);
    cMapMatrixXd wgt_state_(wgt_state, n_state_, n_state_);
    smooth_mv(mu_state_smooths_, var_state_smooths_,
              cur_step, wgt_state_);
    return;
  }
  
  /// @param[out] x_state_smooths Smoothed state `X_n`.
  /// @param[in] x_state_smooths Smoothed state `X_n+1`.
  /// @param[in] mu_state_filts Current state mean `mu_n|n`.
  /// @param[in] var_state_filts Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_preds Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_preds Predicted state variance `Sigma_n+1|n`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] wgt_state Current state transition matrix `T_n`.
  /// @param[in] z_states 2*n_steps of random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTVODE::smooth_sim(RefMatrixXd x_state_smooths,
                                      const int cur_step,
                                      cRefMatrixXd& wgt_state,
                                      cRefMatrixXd& z_states) {
    KalmanTV::smooth_sim(x_state_smooths.col(cur_step),
                         x_state_smooths.col(cur_step+1),
                         mu_state_filts.col(cur_step),
                         var_state_filts.block(0, n_state_*cur_step, n_state_, n_state_),
                         mu_state_preds.col(cur_step+1),
                         var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                         wgt_state,
                         z_states.col(n_steps_ + cur_step));
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTVODE::smooth_sim(double* x_state_smooths,
                                      const int cur_step,
                                      const double* wgt_state,
                                      const double* z_states) {
    MapVectorXd x_state_smooths_(x_state_smooths, n_state_, n_steps_);
    cMapMatrixXd wgt_state_(wgt_state, n_state_, n_state_);
    cMapMatrixXd z_states_(z_states, n_state_, 2*n_steps_);
    smooth_sim(x_state_smooths_, cur_step, 
               wgt_state_, z_states_);
    return;
  }

  /// @param[out] x_state_smooths Smoothed state `X_n`.
  /// @param[out] mu_state_smooths Smoothed state mean `mu_n|N`.
  /// @param[out] var_state_smooths Smoothed state variance `Sigma_n|N`.
  /// @param[in] x_state_smooths Smoothed state `X_n+1`.
  /// @param[in] mu_state_smooths Next smoothed state mean `mu_n+1|N`.
  /// @param[in] var_state_smooths Next smoothed state variance `Sigma_n+1|N`.
  /// @param[in] mu_state_filts Current state mean `mu_n|n`.
  /// @param[in] var_state_filts Current state variance `Sigma_n|n`.
  /// @param[in] mu_state_preds Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_preds Predicted state variance `Sigma_n+1|n`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] wgt_state Current state transition matrix `T_n`.
  /// @param[in] z_states 2*n_steps of random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTVODE::smooth(RefMatrixXd x_state_smooths,
                                  RefMatrixXd mu_state_smooths,
                                  RefMatrixXd var_state_smooths,
                                  const int cur_step,
                                  cRefMatrixXd& wgt_state,
                                  cRefMatrixXd& z_states) {
    smooth_mv(mu_state_smooths, var_state_smooths,
              cur_step, wgt_state);
    smooth_sim(x_state_smooths, cur_step, 
               wgt_state, z_states);
    return;
  }
  /// Raw buffer equivalent.
  inline void KalmanTVODE::smooth(double* x_state_smooths,
                                  double* mu_state_smooths,
                                  double* var_state_smooths,
                                  const int cur_step,
                                  const double* wgt_state,
                                  const double* z_states) {
    smooth_mv(mu_state_smooths, var_state_smooths,
              cur_step, wgt_state);
    smooth_sim(x_state_smooths, cur_step, 
               wgt_state, z_states);
    return;
  }

  /// @param[out] x_state_smooths Smoothed state `X_n`.
  /// @param[out] mu_state_smooths Smoothed state mean `mu_n|N`.
  /// @param[out] var_state_smooths Smoothed state variance `Sigma_n|N`.
  /// @param[in] z_states 2*n_steps of random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTVODE::smooth_update(RefMatrixXd x_state_smooths,
                                         RefMatrixXd mu_state_smooths,
                                         RefMatrixXd var_state_smooths,
                                         cRefMatrixXd& wgt_state,
                                         cRefMatrixXd& z_states,
                                         const bool smooths_mv,
                                         const bool smooths_sim) {
    mu_state_smooths.col(n_steps_-1).noalias() = mu_state_filts.col(n_steps_-1);
    var_state_smooths.block(0, n_state_*(n_steps_-1), n_state_, n_state_).noalias() = \
      var_state_filts.block(0, n_state_*(n_steps_-1), n_state_, n_state_);
    state_sim(x_state_smooths.col(n_steps_-1), mu_state_smooths.col(n_steps_-1), 
              var_state_smooths.block(0, n_state_*(n_steps_-1), n_state_, n_state_), z_states.col(n_steps_-1));
    for(int t=n_steps_-2; t>-1; t--) {
      if(smooths_mv && smooths_sim) {
        smooth(x_state_smooths,
               mu_state_smooths,
               var_state_smooths,
               t,
               wgt_state,
               z_states);
      
      } else if(smooths_mv) {
        smooth_mv(mu_state_smooths,
                  var_state_smooths,
                  t,
                  wgt_state);
      } else if(smooths_sim) {
        smooth_sim(x_state_smooths,
                   t,
                   wgt_state,
                   z_states);
      }
    }
    return;
  }
  inline void KalmanTVODE::smooth_update(double* x_state_smooths,
                                         double* mu_state_smooths,
                                         double* var_state_smooths,
                                         const double* wgt_state,
                                         const double* z_states,
                                         const bool smooths_mv,
                                         const bool smooths_sim) {
    MapMatrixXd x_state_smooths_(x_state_smooths, n_state_, n_steps_);
    MapMatrixXd mu_state_smooths_(mu_state_smooths, n_state_, n_steps_);
    MapMatrixXd var_state_smooths_(var_state_smooths, n_state_, n_state_*n_steps_);
    cMapMatrixXd wgt_state_(wgt_state, n_state_, n_state_);
    cMapMatrixXd z_states_(z_states, n_state_, 2*n_steps_);
    smooth_update(x_state_smooths_, mu_state_smooths_,
                  var_state_smooths_, wgt_state_,
                  z_states_, smooths_mv, smooths_sim);
    return;
  }
  /// @param[out] x_state Simulated state.
  /// @param[out] var_meas Variance of simulated measure.
  /// @param[in] mu_state_preds Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_preds Predicted state variance `Sigma_n+1|n`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] z_states 2*n_steps of random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTVODE::chkrebtii_int(RefVectorXd x_state,
                                         const int cur_step,
                                         cRefMatrixXd& wgt_meas,
                                         cRefMatrixXd& z_states) {
    twgt_meas_.noalias() = wgt_meas * var_state_preds.block(0, n_state_*(cur_step+1),
                                                            n_state_, n_state_); // n_meas x n_state
    var_meas.noalias() = twgt_meas_ * wgt_meas.adjoint();
    state_sim(x_state, mu_state_preds.col(cur_step+1),
              var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
              z_states.col(cur_step));
    return;
  }
  inline void KalmanTVODE::chkrebtii_int(double* x_state,
                                         const int cur_step,
                                         const double* wgt_meas,
                                         const double* z_states) {
    MapVectorXd x_state_(x_state, n_state_);
    cMapMatrixXd wgt_meas_(wgt_meas, n_meas_, n_state_);
    cMapMatrixXd z_states_(z_states, n_state_, 2*n_steps_);
    chkrebtii_int(x_state_, cur_step,
                  wgt_meas_, z_states_);
    return;
  }
} // end namespace KalmanTVODE


#endif
