import logging
from types import SimpleNamespace

import tensorflow as tf
from tqdm import tqdm

from utils.misc import (clf_metrics, regr_metrics, _build_preds, dict_to_str)
from utils.tf_utils import (ar_inference)

tf.logging.set_verbosity(tf.logging.INFO)


class Trainer(object):
    def __init__(self,
                 model,
                 config,
                 path=None):

        model._build()

        self.to_feed = SimpleNamespace(**{
            'x': model.x,
            'y': model.y,
            'y_features': model.y_features,
            'mu': model.mu,
            'std': model.std,
            'keep_prob': model.keep_prob,
            'is_training': model.is_training,
            'gen_len': model.gen_len,
        })

        self.to_fetch = SimpleNamespace(**{
            'y_hat': model.y_hat,
            'h': model.h,
            'loss': model.loss,
            'train': model.train,
            'summary': model.summary,
            'metrics': model.metrics,
            'global_step': model.global_step,
        })

        self._log_path = path
        self._config = config

    def _get_dict(self, feeds):

        feed_dict = {self.to_feed.__dict__[k]: feeds[k] for k in set(feeds.keys()) & self.to_feed.__dict__.keys()}

        return feed_dict

    def init_sess(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), filename='model.ckpt')
        self.train_writer = tf.summary.FileWriter(logdir=self._log_path + '/train')
        self.test_writer = tf.summary.FileWriter(logdir=self._log_path + '/test')
        self.ep_summary = tf.summary.Summary()

    def fit(self, batch, stats=None):

        feed_dict = self._get_dict({**batch, **stats})
        feed_dict[self.to_feed.keep_prob] = self._config.keep_prob
        loss, _ = self.sess.run([self.to_fetch.loss, self.to_fetch.train], feed_dict=feed_dict)
        return loss

    def score(self, feeds, stats=None):
        y_hat = self.predict(feeds, stats)
        metrics = self.get_metrics(feeds['y'], y_hat, self._config.loss)
        return metrics, y_hat

    def predict(self, batch, stats=None):
        feed_dict = self._get_dict({**batch, **stats})
        y = feed_dict.pop(self.to_feed.y)

        fetches = [self.to_fetch.y_hat, self.to_fetch.h]

        steps = y.shape[1] - self._config.pred_len
        if self._config.ar and steps > 0:
            y_hat = ar_inference(feed_dict, fetches=fetches, steps=steps, sess=self.sess, other=None)
        else:
            y_hat, _ = self.sess.run(fetches, feed_dict=feed_dict)

        return y_hat

    def summarize_model(self, batch, stats=None):
        feed_dict = self._get_dict({**batch, **stats})

        summary, gs = self.sess.run([self.to_fetch.summary, self.to_fetch.global_step],
                                    feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=gs)
        self.train_writer.flush()

    def summarize_metrics(self, summary, mode='train'):
        gs = self.sess.run(self.to_fetch.global_step)
        for k, v in summary.items():
            self.ep_summary.value.add(tag=k, simple_value=v)
        if mode == 'train':
            self.train_writer.add_summary(summary=self.ep_summary, global_step=gs)
        else:
            self.test_writer.add_summary(summary=self.ep_summary, global_step=gs)
        self.test_writer.flush()

    def save(self):
        try:
            self.saver.save(sess=self.sess, save_path=self._log_path + '/model/',
                            global_step=self.to_fetch.global_step,
                            write_meta_graph=False)

            tf.logging.log(tf.logging.INFO,
                           "Model saved at global step {}".format(self.sess.run(tf.train.get_global_step())))
        except (ValueError, TypeError, RuntimeError) as e:
            raise e

    def reset(self):
        self.sess.run(tf.local_variables_initializer())

    def restore(self, path=None):
        try:
            if path is None:
                path = self._log_path + '/model'
            elif 'model' not in path.split('/'):
                path = ''.join([path, '/model'])
            ckpt = tf.train.latest_checkpoint(path)
            self.saver.restore(sess=self.sess, save_path=ckpt)
        except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e:
            raise e

    @staticmethod
    def get_metrics(y, y_hat, metrics="clf"):
        if metrics == "clf":
            metrics = clf_metrics(y, y_hat)
        else:
            metrics = regr_metrics(y, y_hat)
        return metrics

    @property
    def global_step(self):
        return self.sess.run(self.to_fetch.global_step)


class Runner(object):
    def __init__(self, steps=50, summarize_every=100, test_every=10, save_every=20, report_every=20):

        self.steps = steps
        self.summarize_every = summarize_every
        self.test_every = test_every
        self.save_every = save_every
        self.report_every = report_every

    def train(self, trainer, dataset, test_set, logger):
        logging.log(logging.INFO, "Start training for {}".format(self.steps))
        ep = 0
        losses = []
        for steps in tqdm(range(self.steps), desc="Epochs"):
            try:
                batch, _ = dataset.next()

                loss = trainer.fit(batch, stats=dataset.stats)
                losses.append(loss)

            except StopIteration:

                if ep % self.summarize_every == 0:
                    trainer.summarize_model(batch, stats=dataset.stats)

                if ep % self.save_every == 0:
                    trainer.save()
                    tqdm.write("Ep: {}, Avg loss: {}".format(ep, sum(losses) / len(losses)), end='\n')
                if ep % self.test_every == 0:
                    metrics, _ = trainer.score(dataset.sample(), stats=dataset.stats)
                    logger.log(metrics, mode='train')
                    trainer.summarize_metrics(metrics, mode='train')
                    tqdm.write(dict_to_str(metrics), end='\n')
                if ep % self.report_every == 0:
                    _, metrics = self.test(trainer, test_set, logger)
                    logger.log(metrics, mode='test')
                    trainer.summarize_metrics(metrics, mode='test')
                    tqdm.write(dict_to_str(metrics), end='\n')

                ep += 1
                dataset.reset()
                test_set.reset()
                losses = []

        logger.dump()
        trainer.save()
        tf.logging.log(logging.INFO, "Training finished.  Epoches: {}".format(ep))

    def test(self, trainer, dataset, logger):
        preds = dict(dates=[], y_hat=[], y=[])
        batches = []
        for batch, dates in dataset:
            _, y_hat = trainer.score(batch, dataset.stats)
            batches.append(batch)

            for k, v in zip(preds.keys(), [dates, y_hat, batch['y']]):
                preds[k].append(v)

        df = _build_preds(preds, path=logger.paths['report'] + '/{}'.format(trainer.global_step))
        score = trainer.get_metrics(df.y.values, df.y_hat.values, metrics=trainer._config.loss)

        return df, score
