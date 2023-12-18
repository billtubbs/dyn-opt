import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype


class DataLogger():
    k: int
    data: pd.DataFrame
    time_var: str
    initial_values: dict
    nT_max: int
    nT_ahead: int
    k_start: int
    k_name: str
    nT_init: int

    def __init__(self, initial_data=None, sample_time=None, t_name='t', ts_name='k', 
                 k_first=None, nT_max=100, nT_ahead=0, **kwargs):

        self.initial_data = initial_data
        self.sample_time = sample_time
        self.t_name = t_name
        self.ts_name = ts_name
        self.nT_max = nT_max
        self.nT_ahead = nT_ahead
        self._kwargs = kwargs
        df, k_first, nT_init, k = DataLogger._initialize_df(
            initial_data, sample_time, t_name, ts_name, k_first, 
            nT_max, nT_ahead, **kwargs
        )
        self.data = df
        self.k_first = k_first
        self.nT_init = nT_init
        self.k = k

    def _initialize_df(initial_data, sample_time, t_name, ts_name, k_first, 
                       nT_max, nT_ahead, **kwargs):

        df = pd.DataFrame(initial_data, **kwargs)
        assert is_integer_dtype(df.index), "index values must be integers"
        assert nT_ahead < nT_max, "nT_max too small for look-ahead horizon"
        df.index.name = ts_name
        if df.index.empty:
            nT_init = 0
            if k_first is None:
                k_first = df.index.start
            if t_name not in df:
                df.insert(0, t_name, np.nan)
            new_index = pd.RangeIndex(k_first, k_first + nT_max, name=ts_name)
            kwargs_mod = {k: v for k, v in kwargs.items() if k not in ['index', 'columns']}
            df = pd.DataFrame(np.nan, index=new_index, columns=df.columns, **kwargs_mod)
        else:
            nT_init = df.index.shape[0]
            assert df.index.duplicated().sum() == 0, "index contains duplicates"
            df = df.sort_index()
            if k_first is None:
                k_first = df.index[0]
            else:
                if k_first != df.index[0]:
                    df.index += (k_first - df.index[0])
            if t_name not in df:
                if df.shape[0] > 1:
                    if sample_time is None:
                        raise ValueError("provide time values in initial_data or sample_time")
                    k = df.index.values + (k_first - df.index[0])
                    df.insert(0, t_name, k * sample_time)
                else:
                    df.insert(0, t_name, 0)
            if df.shape[0] < nT_max:
                new_index = pd.RangeIndex(k_first, k_first + nT_max, name=ts_name)
                kwargs_mod = {k: v for k, v in kwargs.items() if k not in ['index', 'columns']}
                new_df = pd.DataFrame(np.nan, index=new_index, columns=df.columns, **kwargs_mod)
                new_df.loc[df.index] = df.values
                df = new_df
        k = k_first + nT_init - 1

        return df, k_first, nT_init, k

    def append(self, *args):
        if len(args) == 2:
            rows = [args]
        elif len(args) == 1:
            if isinstance(args[0], list):
                rows = args[0]
            #TODO: Support dataframes
            else:
                d = dict(args[0])
                rows = [(d[self.t_name], {k: d[k] for k in d if k != self.t_name})]
        else:
            raise TypeError("invalid arguments")
        for t, data in rows:
            self.k += 1
            if self.k > self.data.index[-1] - self.nT_ahead:
                self.data = self.data.shift(-1)
                self.data.index += 1
            for name in data:
                self.data.loc[self.k, name] = data[name]
            self.data.loc[self.k, self.t_name] = t

    def reset(self):

        self.data, self.k_first, self.nT_init, self.k = DataLogger._initialize_df(
            self.initial_data, self.sample_time, self.t_name, self.ts_name, self.k_first, 
            self.nT_max, self.nT_ahead, **self._kwargs
        )