import numpy as np
import polars as pl
import pandas as pd

def load_data(file_path):
    data = pl.read_csv(file_path)
    data = data.lazy().with_columns(
        pl.col('timestamp').str.split(' '),
        (pl.col('price') * pl.col('volume')).alias('dollar')
    ).with_columns(
        np.log(pl.col('count') + 1).alias('log_count'),
        np.log(pl.col('volume') + 1).alias('log_volume'),
        np.log(pl.col('dollar') + 1).alias('log_dollar'),
        pl.col('timestamp').list.first().str.to_datetime("%Y-%m-%d").dt.weekday().alias('weekday'),
        pl.col('timestamp').list.last().alias('time')
    ).with_columns(
        (pl.col('weekday') < 6).alias('is_weekday')
    ).collect()
    exchange = data['exchange'][0]
    pair = data['pair'][0]
    return data, exchange, pair


def gamma_calc(data, n=100, gamma_len=1501):
    data = data.values
    sample_num = len(data)
    data_demean = np.zeros(sample_num)
    for i in range(n,sample_num-n):
        data_demean[i] = data[i] - np.mean(data[i-n:i+n])
    data_new = data_demean[n:-n]
    gamma = np.array([np.mean(data_new**2)-np.mean(data_new)**2]+\
                     [np.mean(data_new[i:]*data_new[:-i])-np.mean(data_new[i:])*\
                     np.mean(data_new[:-i]) for i in range(1,gamma_len)]) 
    return gamma

def spec_calc(gamma):
    gamma_inv = gamma[::-1]
    gamma_inv = gamma_inv[:-1]
    gamma_full = np.concatenate((gamma[:-1], gamma_inv))
    spec = np.abs(np.fft.rfft(gamma_full))
    return spec

def freq(x):
    if x >= 1:
        return str(x) + 'min'
    else:
        return str(x * 60) + 's'

def freq_list_gen(gamma_len=1501):
    freq_list = (gamma_len-1) / 30 / np.arange(1, gamma_len)
    freq_list = np.concatenate((['non-periodic'], [freq(_) for _ in freq_list]))
    return freq_list

def var_ratio_calc(spec):
    spec[-1] = spec[-1] / 4
    var = spec * 2
    var[0] = var[0] / 2
    var = var / (2 * len(spec) - 1)
    var_ratio = var / var.sum()
    return pd.Series(var_ratio, index=freq_list_gen(len(var_ratio)))

class Analyzer:
    def __init__(self, data):
        self.data = data
        
    def periodicity_analyzer(self, gamma_len=1501, n=100):
        self.gamma_len = gamma_len
        self.n = n
        self.gamma = gamma_calc(self.data, self.n, self.gamma_len)
        self.spec = spec_calc(self.gamma)
        self.var_ratio = var_ratio_calc(self.spec)

class Data:
    def __init__(self, file_path, vol_type_list=['count', 'volume', 'dollar', 'log_count', 'log_volume', 'log_dollar']):
        self.data, self.exchange, self.pair = load_data(file_path)
        self.vol_type_list = vol_type_list

    def weekday_avg(self):
        vol = self.data.lazy().groupby('is_weekday', 'time').mean().select(pl.col(['is_weekday', 'time'] + self.vol_type_list)).sort(['is_weekday', 'time']).collect().filter(pl.col('is_weekday')).to_pandas()
        return vol
    
    def weekend_avg(self):
        vol = self.data.lazy().groupby('is_weekday', 'time').mean().select(pl.col(['is_weekday', 'time'] + self.vol_type_list)).sort(~['is_weekday', 'time']).collect().filter(pl.col('is_weekday')).to_pandas()
        return vol
