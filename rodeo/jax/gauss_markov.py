import numpy as np


def gauss_markov_mv(A, b, C):
    r"""
    Direct calculation of mean and variance for Gaussian Markov models.

    Let :math:`Y = (Y_0, \ldots, Y_N)` be p-dimensional vectors which follow a Gaussian Markov process:

    .. math::

        Y_0 = b_0 + C_0 \epsilon_0

        Y_n = b_n + A_n Y_{n-1} + C_n \epsilon_n,

    where :math:`\epsilon_n` are independent vectors of p-dimensional iid standard normals.  This function computes the mean and variance of :math:`Y`.

    Args:
        A (ndarray(n_steps, n_dim, n_dim)): Transition matrices in the Gaussian process, i.e., :math:`(A_0, \ldots, A_N)`.
        b (ndarray(n_steps+1, n_dim)): Transition offsets in the Gaussian process, i.e., :math:`(b_0, \ldots, b_N)`.
        C (ndarray(n_steps+1, n_dim, n_dim)): Cholesky factors of the variance matrices in the Gaussian process, i.e., :math:`(C_0, \ldots, C_N)`.

    Returns:
        (tuple):
        - **mean** (ndarray(n_steps+1, n_dim)): Mean of :math:`Y`.
        - **var** (ndarray(n_steps+1, n_dim, n_steps+1, n_dim)): Variance of :math:`Y`.

    """
    # Let AA be an n_tot x n_tot block matrix with blocks of size
    # n_dim x n_dim, such that AA_nm = A_{n:m+1}.
    # Now let us work out a few of the block entries:
    #
    # By row:
    #
    # AA_00 = A_0:1
    #       = I
    # AA_01 = A_{0:2}
    #       = I
    # AA_0m = I
    #
    # By column:
    #
    # AA_00 = I
    # AA_10 = A_{1:1}
    #       = A_1
    # AA_20 = A_{2:1}
    #       = A_2 A_1
    # AA_n0 = A_{n:1}
    #       = A_n ... A_1
    #
    # AA_01 = I
    # AA_11 = A_{1:2}
    #       = I
    # AA_21 = A_{2:2}
    #       = A_2
    # AA_n1 = A_n ... A_2
    #
    # AA_02 = I
    # AA_12 = I
    # AA_22 = I
    # AA_32 = A_3
    # AA_n3 = A_n ... A_3
    #
    # This suggests that we calculate AA by column, with:
    #
    # AA_nm = I for n <= m
    #       = A_n AA_{n-1,m} for n > m
    n_tot, n_dim = b.shape  # n_tot = n_steps + 1
    AA = np.zeros((n_tot, n_tot, n_dim, n_dim))
    for m in range(n_tot):
        # m = column index
        for n in range(n_tot):
            # n = row index
            if n <= m:
                AA[n, m] = np.eye(n_dim)
            else:
                AA[n, m] = AA[n-1, m].dot(A[n-1])
    # Now we can calculate L and u
    L = np.zeros((n_tot, n_dim, n_tot, n_dim))
    for m in range(n_tot):
        # m = column index
        for n in range(m, n_tot):
            # n = row index
            L[n, :, m, :] = AA[n, m].dot(C[m])
    u = np.zeros((n_tot, n_dim))
    for n in range(n_tot):
        for m in range(n+1):
            u[n] = u[n] + AA[n, m].dot(b[m])
    # compute V = LL'
    # to do this need to reshape L
    L = np.reshape(L, (n_tot*n_dim, n_tot*n_dim))
    V = np.reshape(L.dot(L.T), (n_tot, n_dim, n_tot, n_dim))
    return u, V
    # An_m = np.zeros((n_genss, n_genss, n_steps, n_steps+1))
    # for n in range(n_steps):
    #     for m in range(n_steps+1):
    #         if m > n:
    #             An_m[:, :, n, m] = np.eye(n_genss)
    #         elif n == m:
    #             An_m[:, :, n, m] = A[:, :, n]
    #         else:
    #             diff = n-m
    #             wgt_diff = A[:, :, m]
    #             for i in range(diff):
    #                 wgt_diff = np.matmul(A[:, :, m+i+1], wgt_diff)
    #             An_m[:, :, n, m] = wgt_diff

    # L = np.zeros((D, D))
    # gaussian_mu = np.zeros(D)
    # for n in range(n_steps):
    #     for m in range(n, n_steps):
    #         L[m*n_genss:(m+1)*n_genss, n*n_genss:(n+1) *
    #           n_genss] = np.matmul(An_m[:, :, m, n+1], C[:, :, n])
    #     for m in range(n+1):
    #         gaussian_mu[n*n_genss:(n+1)*n_genss] = gaussian_mu[n*n_genss:(n+1)
    #                                                            * n_genss] + An_m[:, :, n, m+1].dot(b[:, m])
    # gaussian_var = L.dot(L.T)
    # return gaussian_mu, gaussian_var


def kalman2gm(wgt_state, mu_state, var_state, wgt_meas, mu_meas, var_meas):
    r"""
    Converts the parameters of the Gaussian state-space model

    .. math::

        x_0 = c_0 + R_0^{1/2} \epsilon_0

        x_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

        y_n = d_n + W_n x_n + V_n^{1/2} \eta_n

    to the parameters of the Gaussian Markov model

    .. math::

        Y_n = b_n + A_n Y_{n-1} + C_n \epsilon_n,

    where :math:`Y_n = (x_n, y_n)`.

    Args:
        wgt_state (ndarray(n_steps, n_state, n_state)): Transition matricesin the state model; denoted by :math:`Q_1, \ldots, Q_N`.
        mu_state (ndarray(n_steps+1, n_state)): Offsets in the state model; denoted by :math:`c_0, \ldots, c_N`.
        var_state (ndarray(n_steps+1, n_state, n_state)): Variance matrices in the state model; denoted by :math:`R_0, \ldots, R_N`.
        wgt_meas (ndarray(n_steps, n_meas, n_state)): Transition matrices in the measurement model; denoted by :math:`W_0, \ldots, W_N`.
        mu_meas (ndarray(n_steps+1, n_meas)): Offsets in the measurement model; denoted by :math:`d_0, \ldots, d_N`.
        var_meas (ndarray(n_steps+1, n_meas, n_meas)): Variance matrices in the measurement model; denoted by :math:`V_0, \ldots, V_N`.

    Returns:
        (tuple):
        - **wgt_gm** (ndarray(n_steps, n_dim, n_dim)): Transition matrices in the Gaussian Markov model, where `n_dim = n_state + n_meas`; denoted by :math:`A_1, \ldots, A_N`.
        - **mu_gm** (ndarray(n_steps+1, n_dim)): Offsets in the Gaussian Markov model; denoted by :math:`b_0, \ldots, b_N`.
        - **chol_gm** (ndarray(n_steps+1, n_dim, n_dim)): Cholesky factors of the variance matrices in the Gaussian Markov model; denoted by :math:`C_0, \ldots, C_N`.

    """
    # dimensions
    n_tot, n_meas, n_state = wgt_meas.shape  # n_tot = n_steps + 1
    n_dim = n_state + n_meas
    # increase dimension of wgt_state to simplify indexing
    wgt_state = np.concatenate([np.zeros((n_state, n_state))[None],
                                wgt_state])
    # useful zero matrices
    zero_sm = np.zeros((n_state, n_meas))
    zero_mm = np.zeros((n_meas, n_meas))
    # initialize outputs
    mu_gm = np.zeros((n_tot, n_dim))
    chol_gm = np.zeros((n_tot, n_dim, n_dim))
    # increase dimension of wgt_gm to simplify indexing
    wgt_gm = np.zeros((n_tot, n_dim, n_dim))
    for i in range(n_tot):
        # mean term
        mu_gm[i] = np.concatenate(
            [mu_state[i],
             mu_meas[i] + wgt_meas[i].dot(mu_state[i])]
        )
        # weight term
        if i > 0:
            wgt_gm[i] = np.block(
                [[wgt_state[i], zero_sm],
                 [wgt_meas[i].dot(wgt_state[i]), zero_mm]]
            )
        # cholesky term
        chol_state = np.linalg.cholesky(var_state[i])
        chol_meas = np.linalg.cholesky(var_meas[i])
        chol_gm[i] = np.block(
            [[chol_state, zero_sm],
             [wgt_meas[i].dot(chol_state), chol_meas]]
        )
    return wgt_gm[1:], mu_gm, chol_gm
