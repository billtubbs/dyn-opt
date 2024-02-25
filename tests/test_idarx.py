import os
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pandas as pd
from dynopt.sysid.idarx import idarx, idarxct


class IdArxTests(unittest.TestCase):

    def setUp(self):

        # Simulation setup
        # Based on homework question from GEL-7063 course
        data_dir = 'tests/data'
        filename = 'TP04_Q3.csv'
        self.id_data = pd.read_csv(os.path.join(data_dir, filename))
        assert self.id_data.shape == (75, 2)

    def test_idarx(self):

        na = 2; nb = 2; nk = 3
        p, covp, Vres = idarx([na, nb, nk], self.id_data.u, self.id_data.y)

        assert np.max(np.abs(p - np.array([
            # Output from Octave/Matlab script idarx1.m
            -1.786574,  0.810195, -0.384260,  0.407383
        ]))) < 1e-6

        assert_array_almost_equal(
            p,
            [-1.7865736424,  0.8101952592, -0.3842601544,  0.4073825150]
        )

    def test_idarxct(self):

        # Based on homework question from GEL-7063 course
        na = 2; nb = 2; nk = 3
        F = np.array([[1.7, 1.7, 1.0, 1.0]])
        G = np.array([[-1.7]])

        p = idarxct([na, nb, nk], self.id_data.u, self.id_data.y, F, G)
        assert np.max(np.abs(p - np.array([
            # Output from Octave/Matlab script idarx1.m
            -1.788782,  0.808797, -0.386398, 0.352373
        ]))) < 1e-4
