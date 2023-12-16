import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_index_equal, assert_series_equal, \
                           assert_frame_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

# Functions to test
from dynopt.preprocessing.utils import (
    add_derivatives_savgol,
    add_differences,
    add_ewmas,
    add_filtered_values_savgol,
    add_previous_values, 
    add_rolling_averages, 
    add_subsequent_values, 
    add_timestep_indices, 
    feature_array_from_expressions, 
    feature_dataframe_from_expressions, 
    name_with_t_inc, 
    polynomial_feature_labels, 
    polynomial_features, 
    savgol_filter, 
    split_name, 
    t_inc_str, 
    var_name_sequences
)


class PreprocessingTests(unittest.TestCase):

    def setUp(self):

        # Dummy dataframe
        self.data = pd.DataFrame({'A': range(50, 55), 'B': range(100, 105)})

    def test_split_name(self):
        self.assertEqual(split_name('T1'), ('T1', None))
        self.assertEqual(split_name('T1[t]'), ('T1', 0))
        self.assertEqual(split_name('T1[t+1]'), ('T1', 1))
        self.assertEqual(split_name('T1[t-12]'), ('T1', -12))
        with self.assertRaises(ValueError):
            split_name('T1[')
        with self.assertRaises(ValueError):
            split_name('T1[t+0.1]')
        with self.assertRaises(ValueError):
            split_name('[t]')

    def test_t_inc_str(self):
        self.assertEqual(t_inc_str(0), '[t]')
        self.assertEqual(t_inc_str(2), '[t+2]')
        self.assertEqual(t_inc_str(-3), '[t-3]')

    def test_var_name_sequences(self):
        name_sequence_test = ['A1[t-2]', 'A1[t]', 'A1[t+2]',
                              'B2[t-2]', 'B2[t]', 'B2[t+2]']
        self.assertListEqual(var_name_sequences(['A1', 'B2'], -2, 4, step=2),
                             name_sequence_test)

    def test_add_previous_or_subsequent_values(self):

        # Test 1
        test_values = pd.DataFrame({
            'A[t]': {0: 50, 1: 51, 2: 52, 3: 53, 4: 54},
            'B[t]': {0: 100, 1: 101, 2: 102, 3: 103, 4: 104},
            'A[t+1]': {0: 51.0, 1: 52.0, 2: 53.0, 3: 54.0, 4: np.nan},
            'B[t+1]': {0: 101.0, 1: 102.0, 2: 103.0, 3: 104.0, 4: np.nan},
            'A[t+2]': {0: 52.0, 1: 53.0, 2: 54.0, 3: np.nan, 4: np.nan},
            'B[t+2]': {0: 102.0, 1: 103.0, 2: 104.0, 3: np.nan, 4: np.nan}
        })
        data_add = add_subsequent_values(self.data, 2)
        assert_frame_equal(data_add, test_values)

        # Test 2
        test_values = pd.DataFrame({
            'A[t]': {0: 50, 1: 51, 2: 52, 3: 53, 4: 54},
            'B[t]': {0: 100, 1: 101, 2: 102, 3: 103, 4: 104},
            'A[t-1]': {0: np.nan, 1: 50.0, 2: 51.0, 3: 52.0, 4: 53.0},
            'B[t-1]': {0: np.nan, 1: 100.0, 2: 101.0, 3: 102.0, 4: 103.0},
            'A[t-2]': {0: np.nan, 1: np.nan, 2: 50.0, 3: 51.0, 4: 52.0},
            'B[t-2]': {0: np.nan, 1: np.nan, 2: 100.0, 3: 101.0, 4: 102.0}
        })
        data_add = add_previous_values(self.data, 2)
        assert_frame_equal(data_add, test_values)

        # Test 3
        test_values = pd.DataFrame({
            'A[t]': {0: 50, 1: 51, 2: 52, 3: 53, 4: 54},
            'B[t]': {0: 100, 1: 101, 2: 102, 3: 103, 4: 104},
            'A[t-1]': {0: np.nan, 1: 50.0, 2: 51.0, 3: 52.0, 4: 53.0},
            'A[t-2]': {0: np.nan, 1: np.nan, 2: 50.0, 3: 51.0, 4: 52.0},
            'A[t-3]': {0: np.nan, 1: np.nan, 2: np.nan, 3: 50.0, 4: 51.0}
        })
        data_add = add_previous_values(self.data, 3, cols=['A'])
        assert_frame_equal(data_add, test_values)

        # Test 4
        data = pd.DataFrame({'A': range(50, 55),
                             'B[t]': range(100, 105),
                             'C[t-1]': list('hello')})
        test_values = pd.DataFrame({
            'A[t]': {0: 50, 1: 51, 2: 52, 3: 53, 4: 54},
            'B[t]': {0: 100, 1: 101, 2: 102, 3: 103, 4: 104},
            'C[t-1]': {0: 'h', 1: 'e', 2: 'l', 3: 'l', 4: 'o'},
            'A[t+1]': {0: 51.0, 1: 52.0, 2: 53.0, 3: 54.0, 4: np.nan},
            'B[t+1]': {0: 101.0, 1: 102.0, 2: 103.0, 3: 104.0, 4: np.nan},
            'C[t]': {0: 'e', 1: 'l', 2: 'l', 3: 'o', 4: None}
        })
        data_add = add_subsequent_values(data, 1)
        assert_frame_equal(data_add, test_values)

        # Test 5
        test_values = pd.DataFrame({
            'A[t]': {3: 53, 4: 54},
            'B[t]': {3: 103, 4: 104},
            'A[t-1]': {3: 52.0, 4: 53.0},
            'B[t-1]': {3: 102.0, 4: 103.0},
            'A[t-2]': {3: 51.0, 4: 52.0},
            'B[t-2]': {3: 101.0, 4: 102.0},
            'A[t-3]': {3: 50.0, 4: 51.0},
            'B[t-3]': {3: 100.0, 4: 101.0}
        })
        data_add = add_previous_values(self.data, 3, dropna=True)
        assert_frame_equal(data_add, test_values)

    def test_add_differences(self):

        # Test 1
        test_values = pd.DataFrame({
            'A[t]': self.data['A'],
            'B[t]': self.data['B'],
            'A[t-1]': [np.nan, 50.0, 51.0, 52.0, 53.0],
            'B[t-1]': [np.nan, 100.0, 101.0, 102.0, 103.0],
            'A_m1[t]': [np.nan, 1.0, 1.0, 1.0, 1.0],
            'B_m1[t]': [np.nan, 1.0, 1.0, 1.0, 1.0]
        })
        data_add = add_differences(self.data)
        assert_frame_equal(data_add, test_values)

        # Test 2
        test_values = pd.DataFrame({
            'A[t]': [50, 51, 52, 53, 54],
            'B[t]': [100, 101, 102, 103, 104],
            'A[t-1]': [np.nan, 50.0, 51.0, 52.0, 53.0],
            'A[t-2]': [np.nan, np.nan, 50.0, 51.0, 52.0],
            'A[t-3]': [np.nan, np.nan, np.nan, 50.0, 51.0],
            'A_m3[t]': [np.nan, np.nan, np.nan, 3.0, 3.0]
        })
        data_add = add_differences(self.data, 3, cols=['A[t]'])
        assert_frame_equal(data_add, test_values)

    def test_add_rolling_averages(self):

        test_values = pd.DataFrame({
            'A': [50, 51, 52, 53, 54],
            'B': [100, 101, 102, 103, 104],
            'A_ra3[t]': [np.nan, np.nan, 51.0, 52.0, 53.0],
            'B_ra3[t]': [np.nan, np.nan, 101.0, 102.0, 103.0]
        })
        data_add = add_rolling_averages(self.data, 3)
        assert_frame_equal(data_add, test_values)

    def test_add_filtered_values_savgol(self):

        data = pd.DataFrame({'x': [2, 2, 5, 2, 1, 0, 1, 4, 9]})
        test_values = pd.DataFrame({
            'x': data['x'],
            'x_sgf[t]': [1.6571429, 3.1714286, 3.5428571, 2.8571429, 0.6571429,
                         0.1714286, 1.0, 4.0, 9.0]
        })
        data_add = add_filtered_values_savgol(data, 5, 2)
        assert_frame_equal(data_add, test_values)

    def test_add_ewmas(self):

        test_values = pd.DataFrame({
            'A': [50, 51, 52, 53, 54],
            'B': [100, 101, 102, 103, 104],
            'A_ewma[t]': [50.0, 50.625, 51.326530612244895, 52.09558823529411,
                          52.92158223455933],
            'B_ewma[t]': [100.0, 100.625, 101.3265306122449, 102.0955882352941,
                          102.92158223455932]
        })
        data_add = add_ewmas(self.data)
        assert_frame_equal(data_add, test_values)

    def test_polynomial_feature_labels(self):

        data = pd.DataFrame({'X1': range(5), 'X2': range(5, 10)})
        exps = polynomial_feature_labels(2, 3, names=data.columns)
        self.assertEqual(exps, ['1', 'X1', 'X2', 'X1**2', 'X1*X2', 'X2**2',
                                'X1**3', 'X1**2*X2', 'X1*X2**2', 'X2**3'])
        test_values = pd.Series({0: 0, 1: 6, 2: 14, 3: 24, 4: 36})
        assert_series_equal(data.eval(exps[4]), test_values)

        for exp in exps:
            data[exp] = data.eval(exp)

        name = 'X1**2*X2'
        test_values = pd.Series({0: 0, 1: 6, 2: 28, 3: 72, 4: 144},
                                name=name)
        assert_series_equal(data[name], test_values)

    def test_features_from_expressions(self):

        data = pd.DataFrame({'x0': range(4), 'x1': range(1, 5)})
        expressions = ['1', 'x0*x1', 'x1**2']
        feature_array = feature_array_from_expressions(data, expressions)
        feature_array_test = np.array([
            [ 1.,  0.,  1.],
            [ 1.,  2.,  4.],
            [ 1.,  6.,  9.],
            [ 1., 12., 16.]
        ])
        assert_array_equal(feature_array, feature_array_test)

        features_df = feature_dataframe_from_expressions(data, expressions)
        self.assertEqual(features_df.columns.tolist(), expressions)
        self.assertEqual(features_df.index.tolist(), data.index.tolist())
        assert_array_equal(features_df.values, feature_array_test)

        data = {'x0': 1, 'x1': 2}
        feature_array = feature_array_from_expressions(data, expressions)
        feature_array_test = np.array([[1., 2., 4.]])
        assert_array_equal(feature_array, feature_array_test)


if __name__ == '__main__':
    unittest.main()
