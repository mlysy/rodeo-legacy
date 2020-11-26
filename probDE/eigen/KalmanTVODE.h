/// @file KalmanTVODE.h

#ifndef KalmanTVODE_h
#define KalmanTVODE_h 1

// #undef NDEBUG
// #define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>
#include <iostream>
#include "KalmanTV.h"

namespace kalmantvode {
  using namespace Eigen;

  class KalmanTVODE : public kalmantv::KalmanTV {
  private:
    int n_meas_; ///< Number of measurement dimensions.
    int n_state_; ///< Number of state dimensions.
    int n_steps_; ///< Number of time steps to evaluate.
    // double* mu_state_;
    // double* wgt_state;
    // double* var_state;
    double* x0_state_; ///< Initial x0 value of ODE.
    double* x_state_;
    double* mu_state_;
    double* wgt_state_;
    double* var_state_;
    double* x_meas_;
    double* wgt_meas_;
    double* z_states_;
    double* x_state_smooths_;
    double* mu_state_smooths_;
    double* var_state_smooths_;
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
    /// Default constructor.
    KalmanTVODE(int n_meas, int n_state, int n_steps, double* x0_state, 
                double* x_state, double* mu_state, double* wgt_state,
                double* var_state, double* x_meas, double* wgt_meas,
                double* z_states, double* x_state_smooths,
                double* mu_state_smooths, double* var_state_smooths);
    /// Perform one prediction step of the Kalman filter.
    ///
    /// Calculates `theta_n|n-1` from `theta_n-1|n-1`.
    void predict(const int cur_step);
    /// Perform one update step of the Kalman filter.
    ///
    /// Calculates `theta_n|n` from `theta_n|n-1`.
    void update(const int cur_step);
    /// Perform one step of the Kalman filter.
    /// Combines `predict` and `update` steps to get `theta_n|n` from `theta_n-1|n-1`.
    void filter(const int cur_step);
    /// Perform one step of the Kalman mean/variance smoother.
    ///
    /// Calculates `theta_n|N` from `theta_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`. 
    void smooth_mv(const int cur_step);
    /// Perform one step of the Kalman sampling smoother.
    ///
    /// Calculates a draw `x_n|N` from `x_n+1|N`, `theta_n+1|n+1`, and `theta_n+1|n`. 
    void smooth_sim(const int cur_step);
    /// Perform one step of both Kalman mean/variance and sampling smoothers.
    void smooth(const int cur_step);
    /// Perform the smoothing steps.
    void smooth_update(const bool smooths_mv,
                       const bool smooths_sim);
    /// Perform one step of chkrebtii interrogation.
    void forecast(const int cur_step);
    /// Perform one step of Schobert interrogation.
    void forecast_sch(const int cur_step);
    /// Perform one step of probDE interrogation.
    void forecast_probde(const int cur_step);

  };

  /// @param[in] n_meas Number of measurement variables.
  /// @param[in] n_state Number of state variables.
  /// @param[in] n_steps Number of time steps to evaluate.
  /// @param[in] x0_state Initial value of ODE.
  /// @param[in] x_state Simulated state vector.
  /// @param[in] mu_state Transition_offsets defining the solution prior.
  /// @param[in] wgt_state Transition matrix defining the solution prior.
  /// @param[in] var_state Variance matrix defining the solution prior.
  /// @param[in] x_meas Interrogated measure vector from `x_state`.
  /// @param[in] wgt_meas Transition matrix defining the measure prior.
  /// @param[in] z_states 2*n_steps of random draws from `N(0,1)` for simulating the smoothed state.
  /// @param[in] x_state_smooths Sample solution at time t given observations from each time step.
  /// @param[in] mu_state_smooths Posterior mean of the solution process at each time step.
  /// @param[in] var_state_smooths Posterior variance of the solution process at each time step.
  inline KalmanTVODE::KalmanTVODE(int n_meas, int n_state, int n_steps, double* x0_state, 
                                  double* x_state, double* mu_state, double* wgt_state,
                                  double* var_state, double* x_meas, double* wgt_meas,
                                  double* z_states, double* x_state_smooths,
                                  double* mu_state_smooths, double* var_state_smooths) : KalmanTV(n_meas, n_state) {
    // problem dimensions
    n_meas_ = n_meas;
    n_state_ = n_state;
    n_steps_ = n_steps;
    // initialize memory for all input variables
    x0_state_ = x0_state;
    x_state_ = x_state;
    mu_state_ = mu_state;
    wgt_state_ = wgt_state;
    var_state_ = var_state;
    x_meas_ = x_meas;
    wgt_meas_ = wgt_meas;
    z_states_ = z_states;
    x_state_smooths_ = x_state_smooths;
    mu_state_smooths_ = mu_state_smooths;
    var_state_smooths_ = var_state_smooths;
    // initial vector
    MapVectorXd _x0_state_(x0_state_, n_state_);
    // memory allocation
    mu_meas = VectorXd::Zero(n_meas_);
    var_meas = MatrixXd::Zero(n_meas_, n_meas_);
    mu_state_filts = MatrixXd::Zero(n_state_, n_steps_);
    var_state_filts = MatrixXd::Zero(n_state_, n_state_*n_steps_);
    mu_state_preds = MatrixXd::Zero(n_state_, n_steps_);
    var_state_preds = MatrixXd::Zero(n_state_, n_state_*n_steps_);
    /// initialize mu_state with x0_state
    mu_state_filts.col(0) = _x0_state_;
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
  inline void KalmanTVODE::predict(const int cur_step) {
    MapVectorXd _mu_state_(mu_state_, n_state_);
    MapMatrixXd _wgt_state_(wgt_state_, n_state_, n_state_);
    MapMatrixXd _var_state_(var_state_, n_state_, n_state_);
    KalmanTV::predict(mu_state_preds.col(cur_step+1),
                      var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                      mu_state_filts.col(cur_step),
                      var_state_filts.block(0, n_state_*cur_step, n_state_, n_state_),
                      _mu_state_,
                      _wgt_state_,
                      _var_state_);
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
  inline void KalmanTVODE::update(const int cur_step) {
    MapVectorXd _x_meas_(x_meas_, n_meas_);
    MapMatrixXd _wgt_meas_(wgt_meas_, n_meas_, n_state_);
    KalmanTV::update(mu_state_filts.col(cur_step+1),
                     var_state_filts.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                     mu_state_preds.col(cur_step+1),
                     var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                     _x_meas_,
                     mu_meas,
                     _wgt_meas_,
                     var_meas);
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
  inline void KalmanTVODE::filter(const int cur_step) {
    predict(cur_step);
    update(cur_step);
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
  inline void KalmanTVODE::smooth_mv(const int cur_step) {
    MapMatrixXd _mu_state_smooths_(mu_state_smooths_, n_state_, n_steps_);
    MapMatrixXd _var_state_smooths_(var_state_smooths_, n_state_, n_state_*n_steps_);
    MapMatrixXd _wgt_state_(wgt_state_, n_state_, n_state_);
    KalmanTV::smooth_mv(_mu_state_smooths_.col(cur_step),
                        _var_state_smooths_.block(0, n_state_*cur_step, n_state_, n_state_),
                        _mu_state_smooths_.col(cur_step+1),
                        _var_state_smooths_.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                        mu_state_filts.col(cur_step),
                        var_state_filts.block(0, n_state_*cur_step, n_state_, n_state_),
                        mu_state_preds.col(cur_step+1),
                        var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                        _wgt_state_);
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
  inline void KalmanTVODE::smooth_sim(const int cur_step) {
    MapMatrixXd _x_state_smooths_(x_state_smooths_, n_state_, n_steps_);
    MapMatrixXd _wgt_state_(wgt_state_, n_state_, n_state_);
    MapMatrixXd _z_states_(z_states_, n_state_, 2*n_steps_);
    KalmanTV::smooth_sim(_x_state_smooths_.col(cur_step),
                         _x_state_smooths_.col(cur_step+1),
                         mu_state_filts.col(cur_step),
                         var_state_filts.block(0, n_state_*cur_step, n_state_, n_state_),
                         mu_state_preds.col(cur_step+1),
                         var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
                         _wgt_state_,
                         _z_states_.col(n_steps_ + cur_step));
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
  inline void KalmanTVODE::smooth(const int cur_step) {
    smooth_mv(cur_step);
    smooth_sim(cur_step);
    return;
  }
  /// @param[out] x_state_smooths Smoothed state `X_n`.
  /// @param[out] mu_state_smooths Smoothed state mean `mu_n|N`.
  /// @param[out] var_state_smooths Smoothed state variance `Sigma_n|N`.
  /// @param[in] z_states 2*n_steps of random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTVODE::smooth_update(const bool smooths_mv,
                                         const bool smooths_sim) {
    MapMatrixXd _x_state_smooths_(x_state_smooths_, n_state_, n_steps_);
    MapMatrixXd _mu_state_smooths_(mu_state_smooths_, n_state_, n_steps_);
    MapMatrixXd _var_state_smooths_(var_state_smooths_, n_state_, n_state_*n_steps_);
    MapMatrixXd _z_states_(z_states_, n_state_, 2*n_steps_);
    _mu_state_smooths_.col(n_steps_-1).noalias() = mu_state_filts.col(n_steps_-1);
    _var_state_smooths_.block(0, n_state_*(n_steps_-1), n_state_, n_state_).noalias() = \
      var_state_filts.block(0, n_state_*(n_steps_-1), n_state_, n_state_);
    state_sim(_x_state_smooths_.col(n_steps_-1), _mu_state_smooths_.col(n_steps_-1), 
              _var_state_smooths_.block(0, n_state_*(n_steps_-1), n_state_, n_state_), _z_states_.col(n_steps_-1));
    _mu_state_smooths_.col(0) = mu_state_filts.col(0);
    _x_state_smooths_.col(0) = mu_state_filts.col(0);
    for(int t=n_steps_-2; t>0; t--) {
      if(smooths_mv && smooths_sim) {
        smooth(t);
      } else if(smooths_mv) {
        smooth_mv(t);
      } else if(smooths_sim) {
        smooth_sim(t);
      }
    }
    return;
  }
  /// @param[out] x_state Simulated state.
  /// @param[out] var_meas Variance of simulated measure.
  /// @param[in] mu_state_preds Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_preds Predicted state variance `Sigma_n+1|n`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  /// @param[in] z_states 2*n_steps of random draws from `N(0,1)` for simulating the smoothed state.
  inline void KalmanTVODE::forecast(const int cur_step) {
    MapVectorXd _x_state_(x_state_, n_state_);
    MapMatrixXd _wgt_meas_(wgt_meas_, n_meas_, n_state_);
    MapMatrixXd _z_states_(z_states_, n_state_, 2*n_steps_);
    twgt_meas_.noalias() = _wgt_meas_ * var_state_preds.block(0, n_state_*(cur_step+1),
                                                              n_state_, n_state_); // n_meas x n_state
    var_meas.noalias() = twgt_meas_ * _wgt_meas_.adjoint();
    state_sim(_x_state_, mu_state_preds.col(cur_step+1),
              var_state_preds.block(0, n_state_*(cur_step+1), n_state_, n_state_),
              _z_states_.col(cur_step));
    return;
  }
  /// @param[out] x_state Simulated state.
  /// @param[out] var_meas Variance of simulated measure.
  /// @param[in] mu_state_preds Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_preds Predicted state variance `Sigma_n+1|n`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  inline void KalmanTVODE::forecast_sch(const int cur_step) {
    MapVectorXd _x_state_(x_state_, n_state_);
    _x_state_.noalias() = mu_state_preds.col(cur_step+1);
    return;
  }
  /// @param[out] x_state Simulated state.
  /// @param[out] var_meas Variance of simulated measure.
  /// @param[in] mu_state_preds Predicted state mean `mu_n+1|n`.
  /// @param[in] var_state_preds Predicted state variance `Sigma_n+1|n`.
  /// @param[in] cur_step Current step, n.
  /// @param[in] wgt_meas Current measure transition matrix `W_n`.
  inline void KalmanTVODE::forecast_probde(const int cur_step) {
    MapVectorXd _x_state_(x_state_, n_state_);
    MapMatrixXd _wgt_meas_(wgt_meas_, n_meas_, n_state_);
    twgt_meas_.noalias() = _wgt_meas_ * var_state_preds.block(0, n_state_*(cur_step+1),
                                                              n_state_, n_state_); // n_meas x n_state
    var_meas.noalias() = twgt_meas_ * _wgt_meas_.adjoint();
    _x_state_.noalias() = mu_state_preds.col(cur_step+1);
    return;
  }
} // end namespace KalmanTVODE

#endif
