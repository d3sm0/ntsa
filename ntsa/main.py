import argparse
import inspect
import logging
import sys
import os
import models
from dataset import build_train_test_datasets
from train import (Trainer, Runner, AdversarialRunner)
from utils.logger import Logger
from utils.misc import dict_to_str, get_config
from utils.data_utils import load_data

log = logging.getLogger("main")
log.setLevel(logging.INFO)


def select_model(model_name):
    # should have a function that finds the model and the config
    classes = inspect.getmembers(sys.modules["models"], inspect.isclass)
    Model = (cls for name, cls in classes if name.lower() == model_name.lower()).__next__()
    return Model


def main(config):
    if config.mode == "test" or config.mode == "predict":
        mode = config.mode
        if config.restore_path is not None:
            config.__dict__.update(**Logger.load(config.restore_path))
            config.mode = mode
            config.window = 1

    if config.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    Model = select_model(config.model)
    logger = Logger(config=config.__dict__.copy())

    df = load_data(config.dataset_path)
    dataset, testset, test_df = build_train_test_datasets(df, config)

    model = Model(dataset.shape, config=config)

    trainer = Trainer(model, path=logger.main_path)
    trainer.init_sess()

    if config.restore_path is not None:
        trainer.restore(config.restore_path)

    if "gan" in config.model:
        runner = AdversarialRunner(steps=config.steps)
    else:
        runner = Runner(steps=config.steps)

    if config.mode == "predict":
        runner.predict(trainer, dataset, logger)
        log.info(f"Prediction saved at {logger.paths['report']}")
    elif config.mode == "train" or config.mode == "test":
        if config.mode == "train":
            try:
                runner.train(trainer, dataset, testset, logger)
            except KeyboardInterrupt:
                trainer.save()
        else:
            df, score = runner.test(trainer, testset, logger)
            log.log(logging.INFO, f"Test metrics: {dict_to_str(score)}")
    else:
        logging.error("Selected Mode does not exist.")
        raise NotImplementedError("Selected mode does not exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument('--model', type=str, default="darnn", help='Model name in base path')
    parser.add_argument('--mode', type=str, default="train", help='Mode: train, test, predict.')
    parser.add_argument('--loss', type=str, default="mae", help='Loss type. "clf","smape", mae ')
    parser.add_argument('--dataset_path', type=str, default="../data/benchmark/sm1_2010.csv", help='Data path')
    parser.add_argument('--restore_path', type=str, default=None, help='Base path')
    parser.add_argument('--note', type=str, default=None, help='Some description of the experiment')
    parser.add_argument('--steps', type=int, default=int(1e5), help='Training steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--alpha', type=int, default=1, help='Set to 1 to use kaf on input')
    parser.add_argument('--beta', type=int, default=1, help='Set to 1 to use kaf on time')
    parser.add_argument('--keep_prob', type=float, default=1., help='Keep Prob')
    parser.add_argument('--pred_len', type=int, default=1, help='Size of the prediction')
    parser.add_argument('--seq_len', type=int, default=10, help='Size of the history')
    parser.add_argument('--window', type=int, default=1, help='Size of the iteration window')
    parser.add_argument("--gpu", default=1, type=int, help="Number of gpus.")
    parser.add_argument("--gpu_ratio", default=1.,type=int, help="percentage of the GPU memory to allocate to the experiment")

    console_args = parser.parse_args()
    config = get_config(console_args)

    main(config)
