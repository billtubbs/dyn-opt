import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal

from dynopt.preprocessing.sim_utils import RealTimeDataLogger

class RealTimeDataLoggerTests(unittest.TestCase):

    def test_initialization(self):
        initial_values = {'x': 0.0, 'y': np.nan}
        rtdl = RealTimeDataLogger(initial_values)

        assert rtdl.k_start == 0
        assert rtdl.nT_init == 1
        assert rtdl.nT_max == 100
        assert rtdl.nT_ahead == 0
        assert rtdl.data.shape == (rtdl.nT_max, 3)
        assert_array_equal(rtdl.data.index.to_numpy(), np.arange(100))
        assert rtdl.data.index.name == 'k'
        assert_series_equal(
            rtdl.data.loc[0][initial_values.keys()], 
            pd.Series(initial_values), 
            check_names=False
        )

        initial_values = {'t': 10.0, 'x': 0.0, 'y': np.nan}
        rtdl = RealTimeDataLogger(initial_values, nT_max=5)
        assert_frame_equal(
            rtdl.data,
            pd.DataFrame(
                np.array([
                    [   10.,     0., np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.Index(range(5), name='k'),
                columns=['t', 'x', 'y']
            )
        )

    def test_append(self):

        initial_values = {'t': 10.0, 'x': 0.0, 'y': np.nan}
        rtdl = RealTimeDataLogger(initial_values, nT_max=5)

        rtdl.append(11, {'x': 0.1, 'y': 1.5})
        assert rtdl.k == 1
        rtdl.append({'t': 12.01, 'x': 0.2, 'y': np.nan})  # Alternative
        assert rtdl.k == 2
        assert_frame_equal(
            rtdl.data,
            pd.DataFrame(
                np.array([
                    [   10.,     0., np.nan],
                    [   11.,    0.1,    1.5],
                    [ 12.01,    0.2, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.Index(range(5), name='k'),
                columns=['t', 'x', 'y']
            )
        )

        rtdl.append([
            (13.01, {'x': 0.3, 'y': 1.7}),
            (13.98, {'x': 0.4, 'y': 1.8})
        ])
        assert rtdl.k == 4
        assert_frame_equal(
            rtdl.data,
            pd.DataFrame(
                np.array([
                    [   10.,     0., np.nan],
                    [   11.,    0.1,    1.5],
                    [ 12.01,    0.2, np.nan],
                    [ 13.01,    0.3,    1.7],
                    [ 13.98,    0.4,    1.8]
                ]),
                index=pd.Index(range(5), name='k'),
                columns=['t', 'x', 'y']
            )
        )

        rtdl.append(pd.Series({'t': 14.97, 'x': 0.5, 'y': 1.9}))
        assert rtdl.k == 5
        assert_frame_equal(
            rtdl.data,
            pd.DataFrame(
                np.array([
                    [   11.,    0.1,    1.5],
                    [ 12.01,    0.2, np.nan],
                    [ 13.01,    0.3,    1.7],
                    [ 13.98,    0.4,    1.8],
                    [ 14.97,    0.5,    1.9]
                ]),
                index=pd.Index(range(1, 6), name='k'),
                columns=['t', 'x', 'y']
            )
        )

        rtdl.reset()
        assert_frame_equal(
            rtdl.data,
            pd.DataFrame(
                np.array([
                    [   10.,     0., np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.Index(range(5), name='k'),
                columns=['t', 'x', 'y']
            )
        )
        assert rtdl.k == rtdl.k_start
