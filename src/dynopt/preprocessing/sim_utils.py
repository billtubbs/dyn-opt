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

    def __init__(self, data=None, sample_time=None, columns=None, dtype=None, 
                 t_name='t', ts_name='k', k_first=None, nT_max=100, nT_ahead=0, 
                 **kwargs):
        df = pd.DataFrame(data, columns=columns, dtype=dtype, **kwargs)
        assert is_integer_dtype(df.index), "index values must be integers"
        assert df.index.duplicated().sum() == 0, "index contains duplicates"
        df = df.sort_index()
        df.index.name = ts_name
        nT_init = df.index.shape[0]
        if k_first is None:
            k_first = df.index[0]
        if t_name not in df:
            if df.shape[0] > 1:
                if sample_time is None:
                    raise ValueError("provide time values in data or sample_time")
                k = df.index.values + (k_first - df.index[0])
                df.insert(0, t_name, k * sample_time)
            else:
                df.insert(0, t_name, 0)
        assert nT_ahead < nT_max, "nT_max too small for look-ahead horizon"
        if df.shape[0] < nT_max:
            new_index = pd.RangeIndex(df.index[0], df.index[0] + nT_max, name=ts_name)
            new_df = pd.DataFrame(np.nan, index=new_index, columns=df.columns, dtype=dtype)
            new_df.loc[df.index] = df.values
            df = new_df
        if k_first != df.index[0]:
            df.index += (k_first - df.index[0])
        self.data = df
        self.sample_time = sample_time
        self.t_name = t_name
        self.ts_name = ts_name
        self.k_first = k_first
        self.nT_max = nT_max
        self.nT_ahead = nT_ahead
        self.nT_init = nT_init
        self.k = k_first + nT_init - 1

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

    # TODO: Is this needed?  Would take up memory storing initial values
    # def reset(self):

    #     self.data.iloc[self.nT_init:] = None
    #     self.k = self.data.index[0] + self.nT_init - 1
