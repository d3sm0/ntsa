import csv
import logging
from functools import wraps
from time import time

import matplotlib.pyplot as plt
# import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger("main")
logger.setLevel(logging.WARN)


def get_config(parsed_args):
    import gin.tf.external_configurables

    gin.parse_config_file('config.gin')

    return parsed_args


def dict_to_str(d):
    return ', '.join("{!s}: {:.2f}".format(k, v) for (k, v) in d.items())


def normalize(x, mu, std):
    return x * std + mu


def one_hot(y):
    if y.ndim > 1:
        y = y.flatten()
    b = np.zeros((y.size, int(y.max() + 1)))
    b[np.arange(y.size), y] = 1
    return b


def plot_report(df, path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(df.loc[:, 'y_hat'].values)
    ax.plot(df.loc[:, 'y'].values)
    ax.set_xlabel('time')
    ax.set_ylabel('y')
    ax.legend(['y_hat', 'y'])
    fig.savefig(path + '_snapshot.png')
    plt.close()


def clf_metrics(y, y_hat):
    labels = np.unique(y_hat)
    p, r, f, s = precision_recall_fscore_support(y, y_hat, labels=labels)
    y = y.flatten()
    y_hat = y_hat.flatten()
    return {
        'accuracy': np.mean(y == y_hat),
        'precision': p,
        'recall': r,
        'f_score': f,
        'support': s
    }


def regr_metrics(y, y_hat):
    assert y.shape == y_hat.shape
    return {
        'mse': np.mean(np.square(y - y_hat)),
        'mae': np.mean(np.abs(y - y_hat)),
        'smape': np.mean(smape(y, y_hat))
    }


# https://www.kaggle.com/tigurius/recuplots-and-cnns-for-time-series-classification
def recurrence_plot(s, eps=0.1, steps=10, metric='euclidean'):
    d = pdist(s, metric=metric)
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z


def smape(y, y_hat, eps=0.1):
    summ = np.maximum(np.abs(y) + np.abs(y_hat) + eps, 0.5 + eps)
    smape = 2. * np.abs(y_hat - y) / summ
    return smape  # tf.losses.compute_weighted_loss(smape, w, loss_collection=None)


def float_to_one_hot(y, depth=1):
    y = np.maximum(0, np.sign(y))
    if depth == 2:
        y = one_hot(y.astype(int))
    return y.astype(int)


def l2(y, y_hat):
    assert y.shape[0] == y_hat.shape[0]
    return np.mean((y - y_hat) ** 2)


def check_shape(y, y_hat):
    assert y.shape == y_hat.shape


def load_pickle(path, as_df=True):
    try:
        if as_df:
            import pandas as pd
            return pd.read_pickle(path)
        else:
            import pickle as pkl
            with open(path, 'rb') as f:
                return pkl.load(f)
    except (IOError, OSError) as e:
        raise e


def std_scale(x, loc=True):
    return (x - x.mean()) / x.std()


def min_max_scale(x, loc=True):
    return (x - x.min()) / (x.max() - x.min())


def one_hot(y):
    if y.ndim > 1:
        y = y.flatten()
    b = np.zeros((y.size, int(y.max() + 1)))
    b[np.arange(y.size), y] = 1
    return b


def window_stack(x, step=1, seq_len=24):
    n = x.shape[0]
    return np.hstack(x[i:1 + n + i - seq_len:step] for i in range(0, seq_len))


def shuffle(data, seq_len):
    idxs = np.split(np.arange(0, data['x'].shape[0]), data['x'].shape[0] // seq_len)
    np.random.shuffle(idxs)
    return {k: v[np.ravel(idxs)] for k, v in data.items()}


# @numba.jit(nopython=True)
# def single_autocorr(series, lag):
#     """
#     Autocorrelation for single data series
#     :param series: traffic series
#     :param lag: lag, days
#     :return:
#     """
#     s1 = series[lag:]
#     s2 = series[:-lag]
#     ms1 = np.mean(s1)
#     ms2 = np.mean(s2)
#     ds1 = s1 - ms1
#     ds2 = s2 - ms2
#     divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
#     return np.sum(ds1 * ds2) / divider if divider != 0 else 0

#
# @numba.jit(nopython=True)
# def batch_autocorr(data, lag, starts, ends, threshold, backoffset=0):
#     """
#     Calculate autocorrelation for batch (many time series at once)
#     :param data: Time series, shape [n_pages, n_days]
#     :param lag: Autocorrelation lag
#     :param starts: Start index for each series
#     :param ends: End index for each series
#     :param threshold: Minimum support (ratio of time series length to lag) to calculate meaningful autocorrelation.
#     :param backoffset: Offset from the series end, days.
#     :return: autocorrelation, shape [n_series]. If series is too short (support less than threshold),
#     autocorrelation value is NaN
#     """
#     n_series = data.shape[0]
#     n_days = data.shape[1]
#     max_end = n_days - backoffset
#     corr = np.empty(n_series, dtype=np.float64)
#     support = np.empty(n_series, dtype=np.float64)
#     for i in range(n_series):
#         series = data[i]
#         end = min(ends[i], max_end)
#         real_len = end - starts[i]
#         support[i] = real_len / lag
#         if support[i] > threshold:
#             series = series[starts[i]:end]
#             c_365 = single_autocorr(series, lag)
#             c_364 = single_autocorr(series, lag - 1)
#             c_366 = single_autocorr(series, lag + 1)
#             # Average value between exact lag and two nearest neighborhs for smoothness
#             corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
#         else:
#             corr[i] = np.NaN
#     return corr  # , support


def to_csv(file_name, data):
    with open(file_name + '.csv', 'w') as fout:
        w = csv.DictWriter(fout, data.keys())
        w.writeheader()
        for row in data.values():
            w.writerow(row)


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        logger.debug(f'F:{f.__name__}: runtime: {end-start}')
        return result

    return wrapper
