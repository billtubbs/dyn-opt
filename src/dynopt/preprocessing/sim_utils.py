import numpy as np
import pandas as pd


class RealTimeDataLogger():
    k: int
    data: pd.DataFrame
    time_var: str
    initial_values: dict
    nT_max: int
    nT_ahead: int
    k_start: int
    k_name: str
    nT_init: int

    def __init__(self, initial_values, time_var='t', nT_max=100, 
                 nT_ahead=0, k_start=0, k_name='k'):
        self.initial_values = initial_values
        self.time_var = time_var
        self.nT_max = nT_max
        self.nT_ahead = nT_ahead
        self.k_start = k_start
        self.k_name = k_name
        if time_var not in initial_values:
            initial_values[time_var] = 0
        self.reset()

    def reset(self):
        self.k = self.k_start
        nT_init = 1
        # Find length of longest sequence (if any) in initial_values
        for name in self.initial_values:
            try:
                nT_init = max(nT_init, len(self.initial_values[name]))
            except TypeError:
                pass
        self.nT_init = nT_init
        index = np.arange(self.nT_max)
        index += self.k_start - self.nT_init + 1
        self.data = pd.DataFrame(
            np.nan, 
            columns=self.initial_values.keys(),
            index=pd.Index(index, name=self.k_name)
        )
        self.data.insert(self.k_start, self.time_var, self.data.pop(self.time_var))
        if nT_init == 1:
            self.data.loc[self.k] = self.initial_values
        else:
            # Assume sequence values in initial_values are for previous 
            # time instances up to and including the start time
            for name in self.initial_values:
                try:
                    nT = len(self.initial_values[name])
                except TypeError:
                    self.data.loc[self.k, name] = self.initial_values[name]
                else:
                    self.data.loc[self.k-nT+1:self.k, name] = self.initial_values[name]

    def append(self, *args):
        if len(args) == 2:
            rows = [args]
        elif len(args) == 1:
            if isinstance(args[0], list):
                rows = args[0]
            #TODO: Support dataframes
            else:
                d = dict(args[0])
                rows = [(d[self.time_var], {k: d[k] for k in d if k != self.time_var})]
        else:
            raise TypeError("invalid arguments")
        for t, data in rows:
            self.k += 1
            if self.k > self.data.index[-1] - self.nT_ahead:
                self.data = self.data.shift(-1)
                self.data.index += 1
            for name in data:
                self.data.loc[self.k, name] = data[name]
            self.data.loc[self.k, self.time_var] = t

    def __getitem__(self, key):
         return self.data.loc[key]
