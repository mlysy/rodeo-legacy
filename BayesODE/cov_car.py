import numpy as np


def cov_car(tseq, roots, sigma=1., corr=False):
    delta = np.array(-roots)
    # D = np.diag(delta)
    p = len(roots)
    Q = np.zeros((p, p))

    row = np.ones(p)
    for i in range(p):
        Q[i] = row
        row = row*roots

    Sigma = np.zeros(p)
    Sigma[p-1] = sigma * sigma

    Q_inv = np.linalg.pinv(Q)
    # Gamma = np.linalg.multi_dot([Q, D, Q_inv])  # Q*D*Q^-1
    Sigma_tilde = np.matmul(Q_inv * Sigma, Q_inv.T)  # Q^-1*Sigma*Q^-1'
    # V_tilde_inf
    V_tilde_inf = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            V_tilde_inf[i, j] = Sigma_tilde[i, j] / \
                (delta[i] + delta[j])
            V_tilde_inf[j, i] = V_tilde_inf[i, j]
    if corr:
        V_inf = np.linalg.multi_dot([Q, V_tilde_inf, Q.T])
        sd_inf = np.sqrt(np.diag(V_inf))  # stationary standard deviations

    # covariance matrix
    C = np.zeros((len(tseq), p, p))
    for t in range(len(tseq)):
        # V = np.linalg.multi_dot([Q, V_tilde_inf, Q.T])  # Q*V_tilde_inf*Q.T
        # exp(-Gamma*t) = Q*exp(-D*t)*Q^-1
        exp_Gamma_t = np.matmul(Q_inv.T * np.exp(-delta * tseq[t]), Q.T)
        C[t] = V_inf.dot(exp_Gamma_t)
        if corr:
            C[t] = (C[t] / sd_inf).T
            C[t] = (C[t] / sd_inf).T
    return C
