import unittest
from scipy import integrate
import numpy as np
import sys
import os
from math import exp, erf, sqrt, pi

#Local file
sys.path.insert(0, os.path.abspath('../'))
from BayesODE.cov_fun import cov_vv_re, cov_xv_re, cov_xx_re


class TestIntegrate(unittest.TestCase):
    def R_re(self, t, s):
        if s < t+1 and s > t-1:
            return 1
        else:
            return 0
        
    def RR_re(self, z, t, s):
        return self.R_re(t,z)*self.R_re(s,z)

    def Q_re(self, t, s):
        return max(0, min(s+1,t) - max(0, s-1))

    def QQ_re(self, z, t, s):
        return self.Q_re(t,z)*self.Q_re(s,z)

    def QR_re(self, z, t, s):
        return self.Q_re(t,z)*self.R_re(s,z)

    def cov_vv_re2(self, t,s):
        return integrate.quad(self.RR_re, -np.inf, np.inf, args=(t,s))

    def cov_xx_re2(self, t,s):
        return integrate.quad(self.QQ_re, -1, max(t+1,s+1), args=(t,s))

    def cov_xv_re2(self, t,s):
        return integrate.quad(self.QR_re, -1, max(t+1,s+1), args=(t,s))

    def Q_re2(self, t, s):
        return integrate.quad(self.R_re, 0, t, args=(s,))

    def test_Q_re(self):
        t = np.linspace(1, 2, 5)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(self.Q_re(t[i],t[j]), self.Q_re2(t[i],t[j])[0])

    def test_cov_vv_re(self):
        t = np.linspace(1, 2, 5)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(cov_vv_re(t[i],t[j],1), self.cov_vv_re2(t[i],t[j])[0])


    def test_cov_xv_re(self):
        t = np.linspace(1, 2, 5)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(cov_xv_re(t[i],t[j],1), self.cov_xv_re2(t[i],t[j])[0])

    def test_cov_xx_re(self):
        t = np.linspace(1, 2, 5)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(cov_xx_re(t[i],t[j],1), self.cov_xx_re2(t[i],t[j])[0])

if __name__ == '__main__':
    unittest.main()