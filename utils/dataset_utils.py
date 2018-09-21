import os

import numpy as np
import pandas as pd

NASDAQ = './data/nasdaq.pkl'
SML = './data/sml.pkl'


def load_data(dataset):
    if dataset == "sml":
        df = pd.read_pickle(SML)
    elif dataset == "nasdaq":
        df = pd.read_pickle(NASDAQ)
    else:
        raise OSError("Data not Found")
    df.name = dataset
    return df


def get_daily_test_set(x, split=.7):
    days = np.unique(x.index.dayofyear.values)
    tr_idx = np.sort(np.random.choice(days, size=int(days.shape[0] * split) - 1, replace=False))
    mask = x.index.dayofyear.isin(tr_idx)

    tr = x.iloc[mask].copy()
    ts = x.iloc[~mask].copy()
    print('Training set shape {}'.format(tr.shape[0]))
    print('Test set shape {}'.format(ts.shape[0]))

    assert x.index[mask].intersection(x.index[~mask]).__len__() == 0

    return tr, ts


def train_test_split(df, test_mode="fixed", split=.7):
    if test_mode == "daily":
        assert isinstance(df.index, pd.DatetimeIndex)
        tr, ts = get_daily_test_set(df, split=split)
    else:
        split_idx = int(df.shape[0] * split)
        tr = df.iloc[:split_idx, :]
        ts = df.iloc[split_idx:, :]

    return tr, ts


def make_features(df,
                  seq_len=12,
                  preprocess="normalize",
                  lags=(12,),
                  ar=False,
                  use_x=False):
    mu = 0.
    std = 1.
    x = df.iloc[:, :1].values[:-seq_len]
    y = df.iloc[:, :1].values[seq_len:]
    y_features = np.hstack(
        [df.iloc[:, :1].shift(lag).fillna(0).values[:-seq_len] for lag in
         lags])

    dates = {
        'x': df.index[:-seq_len].tolist(),
        'y': df.index[seq_len:].tolist(),
    }
    if preprocess == "normalize":
        mu = x.mean()
        std = x.std()
        x = (x - mu) / std
        y_features = (y_features - y_features.mean()) / y_features.std()

    if not ar:
        x_features = df.iloc[:, 1:].values[:-seq_len]
        if preprocess == "normalize":
            x_features = (x_features - x_features.mean(axis=0)) / (x_features.std(axis=0))
        if use_x:
            x = np.concatenate([x, x_features], axis=-1)
        else:
            x = x_features

    assert np.isnan(x).sum() == 0

    return {
               'x': x,
               'y': y,
               'y_features': y_features,
           }, dates, {'mu': mu, 'std': std}
