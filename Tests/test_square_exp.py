import unittest
from scipy import integrate
import numpy as np
import sys
import os
from math import exp, erf, sqrt, pi

#Local file
sys.path.insert(0, os.path.abspath('../'))
from BayesODE.cov_fun import cov_vv_se, cov_xv_se, cov_xx_se


class TestIntegrate(unittest.TestCase):
    def R_se(self, t, s):
        return exp(-(s - t)**2/2)

    def RR_se(self, z, t, s):
        return self.R_se(t,z)*self.R_se(s,z)

    def Q_se(self, t,s):
        return sqrt(pi/2)*(erf(s/sqrt(2)) - erf((s-t)/sqrt(2)))

    def QQ_se(self, z, t, s):
        return self.Q_se(t,z)*self.Q_se(s,z)

    def QR_se(self, z, t, s):
        return self.Q_se(t,z)*self.R_se(s,z)

    def cov_vv_se2(self, t,s):
        return integrate.quad(self.RR_se, -np.inf, np.inf, args=(t,s))

    def cov_xx_se2(self, t,s):
        return integrate.quad(self.QQ_se, -np.inf, np.inf, args=(t,s))

    def cov_xv_se2(self, t,s):
        return integrate.quad(self.QR_se, -np.inf, np.inf, args=(t,s))

    def Q_se2(self, t, s):
        return integrate.quad(self.R_se, 0, t, args=(s,))

    def test_Q_se(self):
        t = np.linspace(0.1, 2, 10)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(self.Q_se(t[i],t[j]), self.Q_se2(t[i],t[j])[0])

    def test_cov_vv_se(self):
        t = np.linspace(0.1, 2, 10)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(cov_vv_se(t[i],t[j],1), self.cov_vv_se2(t[i],t[j])[0])


    def test_cov_xv_se(self):
        t = np.linspace(0.1, 2, 10)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(cov_xv_se(t[i],t[j],1), self.cov_xv_se2(t[i],t[j])[0])

if __name__ == '__main__':
    unittest.main()