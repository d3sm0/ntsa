# TODO Make me with tf.data.Dataset
import logging

import numpy as np

log = logging.getLogger("tensorflow")


class Series(object):
    def __init__(self, name, x, seq_len, pad=False):
        self._x = x
        self._name = name
        self._shape = (seq_len, x.shape[-1])
        self._seq_len = seq_len
        self._pad = pad

    def __getitem__(self, idx):
        # may manage padding here
        return self._x[idx:idx + self._seq_len]

    def __len__(self):
        return len(self._x)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._x.dtype


class Dataset(object):
    def __init__(self,
                 data,
                 dates,
                 stats=None,
                 mode="train",
                 batch_size=8,
                 seq_len=24,
                 pred_len=1,
                 random_start=True,
                 window=1):

        self._mode = mode
        self._pred_len = pred_len if pred_len is not None else seq_len

        assert list(data.keys()) == ['x', 'y', 'y_features']

        self._data = {k: Series(k, v, shape) for (k, v), shape in
                      zip(data.items(), (seq_len, pred_len, seq_len))}

        self._dates = dates
        self._stats = stats
        self._n_batch = ((data['x'].shape[0] - seq_len) // window)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._random_start = random_start
        self._window = window
        self.reset()
        log.log(logging.INFO, "Dataset loaded. Total Batches: {}".format(self.batches))

    def __iter__(self):

        batch = {k: [] for k in self._data.keys()}
        dates = []
        should_stop = False
        for idx in range(self._idx, self.size, self._window):
            for k in self._data.keys():
                b = self._data[k][idx]
                if k == 'x' and b.shape[0] < self._seq_len:
                    should_stop = True
                    break
                batch[k].append(b)
            dates.append(self._dates['y'][idx:idx + self._pred_len])

            if len(batch['x']) == self._batch_size:
                batch = {k: np.stack(v, axis=0) for k, v in batch.items()}
                yield batch.copy(), dates
                batch = {k: [] for k in self._data.keys()}
                dates = []

            if should_stop:
                return

    def __str__(self):
        return "Dataset: {}, Batches: {}".format(self._mode, self._n_batch)

    def sample(self):
        self._idx = np.random.randint(0, self.size - self._seq_len * self._batch_size)
        batch, _ = self.__iter__().__next__()
        return batch

    def next(self):
        return self._iterator.__next__()

    def reset(self):
        if self._random_start:
            self._idx = np.random.randint(0, self.size - self._seq_len * self._batch_size)
        else:
            self._idx = 0

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
