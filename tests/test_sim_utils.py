import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal

from dynopt.preprocessing.sim_utils import DataLogger

class DataLoggerTests(unittest.TestCase):

    def test_initialization(self):

        # Empty data logger - no initial data
        dl = DataLogger(columns=['x', 'y', 'm'])
        assert dl.k_first == 0
        assert dl.nT_init == 0
        assert dl.k == -1
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.full((100, 4), np.nan),
                index=pd.RangeIndex(0, 100, name='k'),
                columns=['t', 'x', 'y', 'm']
            )
        )

        # Empty data logger - alternative intialization
        initial_values = {'x': [], 'y': [], 'm': []}
        dl = DataLogger(initial_values)
        assert dl.k_first == 0
        assert dl.nT_init == 0
        assert dl.k == -1
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.full((100, 4), np.nan),
                index=pd.RangeIndex(0, 100, name='k'),
                columns=['t', 'x', 'y', 'm']
            )
        )

        initial_values = {'x': [0.0], 'y': [np.nan], 'm': ['ok']}
        dl = DataLogger(initial_values)

        assert dl.k_first == 0
        assert dl.nT_init == 1
        assert dl.k == 0
        assert dl.nT_max == 100
        assert dl.nT_ahead == 0
        assert dl.data.shape == (dl.nT_max, 4)
        assert_array_equal(dl.data.index.to_numpy(), np.arange(100))
        assert dl.data.index.name == 'k'
        assert_series_equal(
            dl.data.loc[0][initial_values.keys()], 
            pd.DataFrame(initial_values).loc[0], 
            check_names=False
        )

        # DataFrame only works with one compatible dtype
        # E.g. Value 'm': 'ok' raises FutureWarning:  Setting an item of incompatible 
        # dtype is deprecated and will raise in a future error of pandas. Value 
        # '['ok']' has dtype incompatible with float64 ...
        initial_values = {'t': 15.0, 'x': 0.0, 'y': np.nan}
        dl = DataLogger(initial_values, index=[0], nT_max=5)
        assert dl.k_first == 0
        assert dl.nT_init == 1
        assert dl.k == 0
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [   15.,     0., np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.RangeIndex(0, 5, name='k'),
                columns=['t', 'x', 'y']
            )
        )

        initial_values = {
            't': [15.0, 16.5, 18.001], 
            'x': [0.0, 0.1, 0.2],
            'y': [None, 1.5, None],
            'm': ['bad', 'ok', 'bad']
        }
        dl = DataLogger(initial_values, nT_max=5)
        assert dl.k_first == 0
        assert dl.nT_init == 3
        assert dl.k == 2
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                {
                    't': [   15.0,   16.5, 18.001, np.nan, np.nan],
                    'x': [    0.0,    0.1,    0.2, np.nan, np.nan],
                    'y': [ np.nan,    1.5, np.nan, np.nan, np.nan],
                    'm': [  'bad',   'ok',  'bad', np.nan, np.nan],
                },
                #     [   15.0,   0.0, np.nan],
                #     [   16.5,   0.1,    1.5],
                #     [ 18.001,   0.2, np.nan],
                #     [np.nan, np.nan, np.nan],
                #     [np.nan, np.nan, np.nan]
                # ]),
                index=pd.RangeIndex(0, 5, name='k'),
                columns=['t', 'x', 'y', 'm']
            )
        )

        initial_values = {
            'x': [0.0, 0.1, 0.2],
            'y': [None, 1.5, None]
        }
        dl = DataLogger(initial_values, sample_time=1.5, nT_max=5)
        assert dl.k_first == 0
        assert dl.nT_init == 3
        assert dl.k == 2
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [   0.0,    0.0,  np.nan],
                    [   1.5,    0.1,    1.5],
                    [   3.0,    0.2, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.RangeIndex(0, 5, name='k'),
                columns=['t', 'x', 'y']
            )
        )

        dl = DataLogger(initial_values, sample_time=1.5, nT_max=5,
                        k_first=10)
        assert dl.k_first == 10
        assert dl.nT_init == 3
        assert dl.k == 12
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [   15.0,   0.0, np.nan],
                    [   16.5,   0.1,    1.5],
                    [   18.0,   0.2, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.RangeIndex(10, 15, name='k'),
                columns=['t', 'x', 'y']
            )
        )

        initial_values = {
            'x': [0.0, 0.1, 0.2],
            'y': [None, 1.5, None]
        }
        index = [-2, -1, 0]  # include values before k=0
        dl = DataLogger(initial_values, index=index, sample_time=1.5,
                        nT_max=5)
        assert dl.k_first == -2
        assert dl.nT_init == 3
        assert dl.k == 0
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [   -3.0,   0.0, np.nan],
                    [   -1.5,   0.1,    1.5],
                    [    0.0,   0.2, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.RangeIndex(-2, 3, name='k'),
                columns=['t', 'x', 'y']
            )
        )

        dl = DataLogger(initial_values, sample_time=1.5, k_first=-2,
                        nT_max=5)
        assert dl.k_first == -2
        assert dl.nT_init == 3
        assert dl.k == 0
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [   -3.0,   0.0, np.nan],
                    [   -1.5,   0.1,    1.5],
                    [    0.0,   0.2, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.RangeIndex(-2, 3, name='k'),
                columns=['t', 'x', 'y']
            )
        )


    def test_append(self):

        initial_values = {'t': [15.0], 'x': [0.0], 'y': [np.nan]}
        dl = DataLogger(initial_values, nT_max=5)
        assert dl.k_first == 0
        assert dl.nT_init == 1
        assert dl.k == 0

        dl.append(16.5, {'x': 0.1, 'y': 1.5})
        assert dl.k == 1
        dl.append({'t': 18.001, 'x': 0.2, 'y': np.nan})  # Alternative
        assert dl.k == 2
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [   15.0,    0.0,  np.nan],
                    [   16.5,    0.1,     1.5],
                    [ 18.001,    0.2,  np.nan],
                    [ np.nan,  np.nan,  np.nan],
                    [ np.nan,  np.nan,  np.nan]
                ]),
                index=pd.RangeIndex(0, 5, name='k'),
                columns=['t', 'x', 'y']
            )
        )

        dl.append([
            (19.501, {'x': 0.3, 'y': 1.7}),
            (20.987, {'x': 0.4, 'y': 1.8})
        ])
        assert dl.k == 4
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [   15.0,    0.0,  np.nan],
                    [   16.5,    0.1,     1.5],
                    [ 18.001,    0.2,  np.nan],
                    [ 19.501,    0.3,     1.7],
                    [ 20.987,    0.4,     1.8]
                ]),
                index=pd.RangeIndex(0, 5, name='k'),
                columns=['t', 'x', 'y']
            )
        )

        dl.append(pd.Series({'t': 22.488, 'x': 0.5, 'y': 1.9}))
        assert dl.k == 5
        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [   16.5,    0.1,     1.5],
                    [ 18.001,    0.2,  np.nan],
                    [ 19.501,    0.3,     1.7],
                    [ 20.987,    0.4,     1.8],
                    [ 22.488,    0.5,    1.9]
                ]),
                index=pd.RangeIndex(1, 6, name='k'),
                columns=['t', 'x', 'y']
            )
        )

        dl.reset()
        assert dl.k_first == 0
        assert dl.nT_init == 1
        assert dl.k == 0

        assert_frame_equal(
            dl.data,
            pd.DataFrame(
                np.array([
                    [  15.0,    0.0, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]
                ]),
                index=pd.RangeIndex(0, 5, name='k'),
                columns=['t', 'x', 'y']
            )
        )


if __name__ == "__main__":
    unittest.main()
