import numpy as np

def cov_car(tseq, roots, sigma=1, corr=False):
    delta = -roots
    D = np.diag(delta)
    p = len(roots)
    Q = np.zeros((p, p))
    
    row = np.ones(p)
    for i in range(p):
        Q[i] = row
        row = row*roots

    Sigma = np.zeros((p,p))
    Sigma[p-1,p-1] = sigma**2

    Q_inv = np.linalg.pinv(Q)
    Gamma = np.linalg.multi_dot([Q, D, Q_inv]) #Q*D*Q^-1
    Sigma_tilde = np.linalg.multi_dot([Q_inv, Sigma, Q_inv.T]) #Q^-1*Sigma*Q^-1'
    
    cov = np.zeros((len(tseq),p,p))
    for t in range(len(tseq)):
        V_tilde = np.zeros((p,p))
        for i in range(p):
            for j in range(i, p):
                V_tilde[i,j] = Sigma_tilde[i,j] / (delta[i] + delta[j])  #V_tilde_inf
                V_tilde[j,i] = V_tilde[i,j]
    
        V = np.linalg.multi_dot([Q, V_tilde, Q_inv]) #Q*V_tilde_inf*Q^-1
        cov[t] = V.dot(np.exp(Gamma*tseq[t])) #Cov_deltat
    
    return cov