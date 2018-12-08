#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gin
import numpy as np
import pandas as pd

import logging
from utils.misc import plot_report


# partial return a new object that when called act like a function

def rbf_kernel(x, l1, sigma, noise=2e-1):
    x1 = np.expand_dims(x, axis=1)
    x2 = np.expand_dims(x, axis=2)
    diff = x1 - x2

    norm = np.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
    norm = np.sum(norm, axis=-1)
    k = np.square(sigma)[:, :, None, None] * np.exp(-0.5 * norm)
    k += (noise ** 2) * np.eye(k.shape[3])
    return k


def gp(samples=10, seq_len=24, x_size=1, y_size=1, l1_scale=.4, sigma_scale=1., as_df=False):
    x = np.random.uniform(-2., 2., size=[samples, seq_len, x_size])
    l1 = np.ones(shape=[samples, y_size, x_size]) * l1_scale
    sigma = np.ones(shape=[samples, y_size]) * sigma_scale
    k = rbf_kernel(x, l1, sigma)
    _k = np.linalg.cholesky(k)
    y = _k @ np.random.normal(size=[samples, y_size, seq_len, 1])
    y = np.transpose(np.squeeze(y, 3), [0, 2, 1])

    x = x.reshape(samples * seq_len, x_size)
    y = y.reshape(samples * seq_len, y_size)
    out = np.concatenate([y, x], axis=1)
    if as_df:
        return pd.DataFrame(out, columns=[f"x_{idx}" for idx in range(x_size)] + [f"y_{idx}" for idx in range(y_size)])
    return y, x


def build_preds(preds, path=None, stats=None):
    assert isinstance(preds, dict)

    preds['dates'] = pd.Series(np.vstack(preds['dates']).flatten(), name='dates')

    preds['y'] = pd.Series(np.vstack(preds['y']).flatten(), name="y")
    preds['y_hat'] = pd.Series(np.vstack(preds['y_hat']).flatten(), name='y_hat')
    assert preds['y_hat'].shape == preds['y'].shape
    preds = pd.concat(preds.values(), axis=1)
    preds.set_index('dates', inplace=True)
    if stats is not None:
        mu, std = stats.values()
        preds = preds * std + mu

    plot_report(preds, path=path)
    if preds.index.duplicated().any():
        logging.log(logging.WARN, "Duplicated found in prediction index")
        preds = preds[~preds.index.duplicated(keep='first')]
    preds.to_csv(path + "_report.csv")
    logging.info(f"Report saved at {path}")
    return preds


def make_features(df, seq_len=24, **kwargs):
    df = df.copy()

    if df.isna().sum().any():
        df.fillna(method='bfill', inplace=True)

    idxs = {
        'x': df.index.tolist(),
        'y': df.index.tolist()
    }

    # x = df.iloc[:, :1].values #[:-seq_len]
    y = df.iloc[:, :1].values  # [seq_len:] # 'temperature_comedor_sensor'
    x_features = df.iloc[:, 1:].values  # [:-seq_len]

    # y_features = np.random.normal(size=(y.shape[0], 1)) # you should change this with the appropriate y features

    x_features = (x_features - x_features.mean(axis=0)) / (x_features.std(axis=0) + 1e-6)

    y_features = df.iloc[:, :1].shift(1).fillna(value=0).values.reshape((-1, 1))
    y_features = (y_features - y_features.mean(axis=0)) / (y_features.std(axis=0) + 1e-6)

    mu = y.mean()
    std = y.std()
    # x = (x - mu) / std
    y = (y - mu) / std

    x = x_features

    assert np.isnan(x).sum() == 0 and np.isnan(y_features).sum() == 0
    assert x.shape[0] == y.shape[0]
    train = dict(x=x, y=y, y_features=y_features)
    return train, idxs, {'mu': mu, 'std': std}


def get_daily_test_set(x, split=.2, seq_len=24):
    days = np.unique(x.index.dayofyear.values)

    ts_idx = np.sort(np.random.choice(days, size=int(days.shape[0] * split) - 1, replace=False))
    mask = x.index.dayofyear.isin(ts_idx)

    tr = x.iloc[~mask].copy()
    ts = x.iloc[mask].copy()
    print('Training set shape {}'.format(tr.shape[0]))
    print('Test set shape {}'.format(ts.shape[0]))

    assert x.index[~mask].intersection(x.index[mask]).__len__() == 0

    return tr, ts


@gin.configurable
def train_test_split(df, test_mode="fixed", split=.7, split_idx=3200):
    if test_mode == "daily" and isinstance(df.index, pd.DatetimeIndex):
        logging.warning("Could not perform daily test. Index is not an instance of pd.DatetimeIndex")
        tr, ts = get_daily_test_set(df, split=split)
    else:
        # split_idx = int(df.shape[0] * split)
        tr = df.iloc[:split_idx, :]
        ts = df.iloc[split_idx:, :]
        # vs = df.iloc[split_idx:, :] # validation set

    return tr, ts


def load_data(path):
    ext = path.split('.')[-1]
    if ext == "csv":
        df = pd.read_csv(path, index_col=0)
    elif ext == "pkl":
        df = pd.read_pickle(path)
    else:
        raise FileNotFoundError("Provide a file path csv or pkl")
    return df
