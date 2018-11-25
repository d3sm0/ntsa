import logging

import numpy as np

from utils.data_utils import make_features, train_test_split

log = logging.getLogger("dataset")


class Series(object):
    def __init__(self, x, seq_len, name="series", should_pad=False):
        self._x = x
        self._name = name
        self._shape = (seq_len, x.shape[-1])
        self._seq_len = seq_len
        self._should_pad = should_pad

    def __getitem__(self, idx):
        # may manage padding here
        x = self._x[idx: idx + self._seq_len]
        pad_size = self._seq_len - x.shape[0]
        if pad_size and self._should_pad:
            mu = x.mean(axis=0)
            pad = np.repeat(mu, pad_size)
            x = np.concatenate([x, pad], axis=0)
        return x

    def __len__(self):
        return len(self._x)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._x.dtype


# @gin.configurable
class Dataset(object):
    def __init__(self,
                 data,
                 dates=None,
                 stats=None,
                 mode="train",
                 batch_size=32,
                 seq_len=24,
                 pred_len=24,
                 random_start=True,
                 window=1):

        self._mode = mode
        self._pred_len = pred_len if pred_len is not None else seq_len
        self._seq_len = seq_len

        assert list(data.keys()) == ['x', 'y', 'y_features']

        self._data = {k: Series(v, shape, k) for (k, v), shape in
                      zip(data.items(), (self._seq_len, self._pred_len, self._seq_len))}

        self._dates = dates if dates is not None else {"x": np.array([]), "y": np.array([])}
        self._stats = stats
        self._n_batch = ((data['x'].shape[0] - seq_len) // window)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._random_start = random_start
        self._window = window

        self._idx = 0
        self._iterator = None

        self.reset()
        log.info(f"Dataset loaded. Total Batches: {self.batches}")

    def __iter__(self):
        # TODO it works but it shit...fix me

        batch = {k: [] for k in self._data.keys()}
        dates = {"x": [], "y": []}
        should_stop = False
        for idx in range(self._idx, self.size, self._window):
            for k in self._data.keys():
                b = self._data[k][idx]
                if k == 'x' and b.shape[0] < self._seq_len:
                    should_stop = True
                    break
                batch[k].append(b)
            dates["x"].append(self._dates['x'][idx:idx + self._seq_len])
            dates["y"].append(self._dates['y'][idx:idx + self._pred_len])

            if len(batch['x']) == self._batch_size:
                batch = {k: np.stack(v, axis=0) for k, v in batch.items()}
                yield batch, dates
                batch = {k: [] for k in self._data.keys()}
                dates = {"x": [], "y": []}

            if should_stop:
                # yield batch, dates
                return

    def __repr__(self):
        return f'Dataset: {self._mode}, Batches: {self._n_batch}'

    def sample(self):
        self._idx = np.random.randint(0, self.size - self._seq_len * self._batch_size)
        batch = self.__iter__().__next__()
        return batch

    def next(self):
        return self._iterator.__next__()

    def reset(self):
        if self._random_start:
            self._idx = np.random.randint(0, self.size - self._seq_len * self._batch_size)
        else:
            self._idx = 0
        # may add random noise to corrupt the original data at every reset
        self._iterator = self.__iter__()

    @property
    def shape(self):
        return tuple(v.shape for k, v in self._data.items())

    @property
    def types(self):
        return tuple(v.type for k, v in self._data.items())

    @property
    def size(self):
        return len(self._data['x'])

    @property
    def batches(self):
        return self._n_batch

    @property
    def stats(self):
        return self._stats

    @property
    def data(self):
        raise self._data

    @property
    def outputs(self):
        return list(self._data.keys())


def build_train_test_datasets(df, config):
    if config.mode == "predict":
        dataset = Dataset(*make_features(df, seq_len=config.seq_len), mode=config.mode, batch_size=config.batch_size,
                          seq_len=config.seq_len, pred_len=config.pred_len, window=config.window, random_start=False)
        return dataset, None, df

    tr, test_df = train_test_split(df)

    tr, ts = [make_features(t, seq_len=config.seq_len)
              for t in (tr, test_df)]

    train_set = Dataset(*tr, mode="train",
                        batch_size=config.batch_size,
                        seq_len=config.seq_len,
                        pred_len=config.pred_len,
                        window=1)

    test_set = Dataset(*ts, mode="test",
                       batch_size=config.batch_size,
                       seq_len=config.seq_len,
                       pred_len=config.pred_len,
                       random_start=False,
                       window=config.pred_len)

    return train_set, test_set, test_df
