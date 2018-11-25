#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import datetime
import json
import logging
import os
from collections import OrderedDict
from enum import Enum
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import gin

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def create_logger(path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=path + '/' + __name__ + '.log',
        filemode="w"
    )
    hdlr = logging.StreamHandler()
    hdlr.setLevel(logging.INFO)
    logging.getLogger("main").addHandler(hdlr)


class Folder(Enum):
    train = "train"
    model = "model"
    test = "test"
    report = "report"


class Statistic(object):
    def __init__(self, name):
        self._name = name
        self.history = np.array([])

    def __repr__(self):
        return f"{self.name}:\t{self.mean},\t[{self.history.min()},{self.history.max()}]"

    def __len__(self):
        return len(self.history)

    def append(self, value):
        self.history = np.append(self.history, value)

    def reset(self):
        self.history = np.array([])

    @property
    def name(self):
        return self._name

    @property
    def mean(self):
        return self.history.mean()


@gin.configurable
class Logger(object):
    def __init__(self, base_path="logs", config=None):

        date_dir = datetime.datetime.now().strftime("ntsa-%Y-%m-%d-%H-%M-%S")  # %f if needed

        self.main_path = os.path.join(base_path, config['model'], date_dir)

        self.paths = {}
        for f in Folder:
            os.makedirs(os.path.join(self.main_path, f.value), exist_ok=True)
            self.paths[f.value] = os.path.join(self.main_path, f.value)

        config['restore_path'] = self.main_path
        pprint(config)

        with open(self.main_path + '/config.json', 'w') as f:
            json.dump(config, f)

        self._active_path = self.paths['train'] if config['mode'] == 'train' else self.paths['test']

        self._summary = None
        self._default_stats = None

        create_logger(self.main_path)

    def register(self, stats=()):
        if self._default_stats is None:
            self._default_stats = stats
            with open(self._active_path + '/results.csv', 'w') as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(self._default_stats)
            self.reset()

    def add(self, stats):
        for k in self._summary.keys():
            self._summary[k].append(stats[k])

    def summarize(self):
        summary = {}
        for k, v in self._summary.items():
            summary[k] = v.mean
        return summary

    def log(self):
        logging.info("=" * 24)
        # logger.info("Summary at t {}".format(self._summary['t'][-1]))
        for k, v in self._summary.items():
            logging.info(f"{k}: {v.mean}")
        logging.info("=" * 24)

    def dump(self):
        with open(self._active_path + '/results.csv', 'a') as f:
            writer = csv.writer(f)
            history = [v.history for v in self._summary.values()]
            writer.writerows(zip(*history))
        self._summary = OrderedDict({stat: Statistic(name=stat) for stat in self._default_stats})

    def reset(self, mode='train', stats=None):

        if stats is not None:
            self._default_stats = stats

        if self._summary and len(self._summary[self._default_stats[0]]):
            self.dump()

        self._summary = OrderedDict({stat: Statistic(name=stat) for stat in self._default_stats})
        self._active_path = self.paths['train'] if mode == 'train' else self.paths['test']

    @staticmethod
    def load(path):
        try:
            with open(path + '/config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.ERROR("File not found")

    @staticmethod
    def get_config(path):
        return SimpleNamespace(**Logger.load(path))
