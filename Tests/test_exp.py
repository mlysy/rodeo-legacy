import unittest
from scipy import integrate
import numpy as np
import sys
import os
from math import exp, erf, sqrt, pi

#Local file
sys.path.insert(0, os.path.abspath('../'))
from BayesODE.cov_fun import cov_vv_ex, cov_xv_ex, cov_xx_ex


class TestIntegrate(unittest.TestCase):
    
    def cov_xv_ex2(self, t, s):
        return integrate.quad(cov_vv_ex, 0, t, args=(s,1))

    def cov_xx_ex2(self, t, s):
        return integrate.dblquad(cov_vv_ex, 0, t, lambda s: 0, s, args=(1, ))
    
    def test_cov_xv_ex(self):
        t = np.linspace(1/2, 1, 2)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(cov_xv_ex(t[i],t[j],1), self.cov_xv_ex2(t[i],t[j])[0])

    def test_cov_xx_ex(self):
        t = np.linspace(1/2, 1, 2)
        for i in range(len(t)):
            for j in range(len(t)):
                np.testing.assert_almost_equal(cov_xx_ex(1/2,1,1), self.cov_xx_ex2(1/2,1)[0])

 
if __name__ == '__main__':
    unittest.main()