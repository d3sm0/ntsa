import argparse
import logging
import shutil

from base import (Dense, RNN, DARNN, Seq2Seq, SeriesNet, NP)
from config import (dense, rnn, darnn, seq2seq, seriesnet, np_config, trainconfig)
from dataset import Dataset
from train import (Trainer, Runner)
from utils.dataset_utils import (load_data, train_test_split, make_features)
from utils.misc import Logger, dict_to_str
from utils.tf_utils import set_seed

log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)


def build_train_test_datasets(tr, ts, config):
    tr, ts = [make_features(t,
                            seq_len=config.seq_len,
                            preprocess=config.preprocess,
                            lags=config.lags,
                            ar=config.ar,
                            ) for t in (tr, ts)]

    train_set = Dataset(*tr, mode="train" if config.mode != "debug" else "debug",
                        batch_size=config.batch_size,
                        seq_len=config.seq_len,
                        pred_len=config.pred_len,
                        random_start=config.random_start,
                        window=1)

    test_set = Dataset(*ts, mode="test" if config.mode != "debug" else "debug",
                       batch_size=config.batch_size,
                       seq_len=config.seq_len,
                       pred_len=config.test_len,
                       random_start=False,
                       window=config.pred_len)

    return train_set, test_set


def select_model(model_type):
    if model_type == "dense":
        Model = Dense
        config = dense
    elif model_type == "rnn":
        Model = RNN
        config = rnn
    elif model_type == "darnn":
        Model = DARNN
        config = darnn
    elif model_type == "seq2seq":
        Model = Seq2Seq
        config = seq2seq
    elif model_type == "snet":
        Model = SeriesNet
        config = seriesnet
    elif model_type == "np":
        Model = NP
        config = np_config
    else:
        raise NotImplementedError()

    return Model, config


def main(config):
    mode = config.mode
    if config.restore_path is not None:
        config.__dict__.update(**Logger.load(config.restore_path))
        config.mode = mode

    Model, model_config = select_model(config.model)
    if config.restore_path is None:
        config.__dict__.update(**trainconfig)
        config.__dict__.update(**model_config)

    logger = Logger(base_path=config.path + config.model, config=config.__dict__.copy())
    df = load_data(config.dataset)
    tr, ts = train_test_split(df, config.test_mode)

    train_set, test_set = build_train_test_datasets(tr, ts, config)

    set_seed(config.seed)

    model = Model(train_set.shape, config=config)

    trainer = Trainer(model, path=logger.main_path, config=config)
    trainer.init_sess()

    runner = Runner(steps=config.steps, summarize_every=config.summarize_every, test_every=config.test_every,
                    save_every=config.save_every, report_every=config.report_every)

    if config.restore_path is not None:
        trainer.restore(config.restore_path)
    if config.mode != "test":
        try:
            runner.train(trainer, train_set, test_set, logger)
        except KeyboardInterrupt:
            trainer.save()

    df, score = runner.test(trainer, test_set, logger)
    log.log(logging.INFO, "Test metrics: {}".format(dict_to_str(score)))

    if config.mode == "debug":
        try:
            shutil.rmtree(logger.main_path)
        except OSError:
            log.log(logging.ERROR, "Directory not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument('--model', type=str, default="np", help='Model name in base path')
    parser.add_argument('--path', type=str, default='tf/', help='Base path')
    parser.add_argument('--loss', type=str, default="smape", help='Loss type. "clf","smape", mae ')
    parser.add_argument('--mode', type=str, default="train", help='Set mode')
    parser.add_argument('--restore_path', type=str, default=None, help='Base path')
    parser.add_argument('--note', type=str, default=None, help='Some description of the experiment')
    parser.add_argument('--steps', type=int, default=int(1e5), help='Training steps')
    parser.add_argument('--seed', type=int, default=200, help='Seed')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=.8, help='Keep prob')
    parser.add_argument('--clip', type=float, default=20., help='Clip grad')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate')
    parser.add_argument('--random_start', type=bool, default=True, help='Random start')
    main(parser.parse_args())
