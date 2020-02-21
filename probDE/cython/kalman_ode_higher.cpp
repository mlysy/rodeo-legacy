#include <Eigen/Dense>
using namespace Eigen;
#include "KalmanTV.h"

// just does smooth
void kalman_ode_chkrebtii(double* mu_state_smooths,
			  double* var_state_smooths,
			  const int n_dim_meas, const int n_dim_state,
			  const int n_eval,			  
			  const double* x0_state,
			  const double tmin, const double tmax,
			  const double* wgt_state,
			  const double* mu_state, const double* var_state,
			  const double* wgt_meas, const double* z_state_sim) {
  // convert to Eigen types, e.g.,
  // note that Eigen only knows about vectors and matrices, so
  // 3d arrays should be converted to matrices by concatenating 3rd dimensions
  // into second.
  MapMatrixXd mu_state_smooths_(mu_state_smooths, n_dim_state, n_eval+1);

  // memory allocation
  VectorXd mu_meas = VectorXd::Zero(n_dim_meas);

  // initialization
  // use (i), (i,j), for elementwise access, and .block() for matrix subblocks.
  // more on all this here: https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
  mu_state_filts.col(0) = x0_state;

  // forward pass
  KalmanTV KFS(n_dim_meas, n_dim_state);
  for(int t=0; t<n_eval; t++) {
    KFS.predict(mu_state_preds.col(t+1),
		var_state_preds.block(0, n_dim_state*(t+1),
				      n_dim_state, n_dim_state);
		mu_state_filts.col(t),
                var_state_filts.block(0, n_dim_state*(t+1),
				      n_dim_state, n_dim_state),
                mu_state,
                wgt_state,
                var_state);
  }
}
