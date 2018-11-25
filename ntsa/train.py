import logging
from types import SimpleNamespace

import gin
from tqdm import tqdm

from utils.misc import (regr_metrics, dict_to_str)
from utils.data_utils import build_preds
from utils.tf_utils import (restore, init_sess)


class Trainer(object):
    def __init__(self, model, path=None):

        model.build()

        # set_seed(config.seed)

        self.to_feed = SimpleNamespace(**{
            'x': model.x,
            'y': model.y,
            'y_features': model.y_features,
            'mu': model.mu,
            'std': model.std,
            'keep_prob': model.keep_prob,
            'is_training': model.is_training,
            'gen_len': model.gen_len,
            'flag': model.flag,
        })

        self.to_fetch = SimpleNamespace(**{
            'y_hat': model.y_hat,
            'h': model.h,
            'summary_list': model.fetch_summary,
            'loss': model.loss,
            'train': model.train,
            'summary': model.summary,
            # 'metrics': model.metrics,
            'global_step': model.global_step,
        })

        self._log_path = path
        # self._config = config
        self._vars = model.vars

    def _get_dict(self, feeds):

        feed_dict = {self.to_feed.__dict__[k]: feeds[k] for k in set(feeds.keys()) & self.to_feed.__dict__.keys()}

        return feed_dict

    def init_sess(self):
        self.sess, self.saver, self.train_writer = init_sess(var_list=self._vars, path=self._log_path + "/train")

    # self.test_writer = tf.summary.FileWriter(logdir=self._log_path + '/test')
    # self.ep_summary = tf.summary.Summary()

    def fit(self, batch, target):

        feed_dict = self._get_dict(batch)
        feed_dict[self.to_feed.keep_prob] = 1.
        feed_dict[self.to_feed.flag] = target
        loss, _ = self.sess.run([self.to_fetch.loss, self.to_fetch.train], feed_dict=feed_dict)
        return loss

    def score(self, batch):
        y_hat = self.predict(batch)
        metrics = self.sess.run(self.to_fetch.summary_list, feed_dict=self._get_dict(batch))

        # metrics = self.get_metrics(batch['y'], y_hat)
        # metrics["loss"] = loss
        return metrics, y_hat

    def predict(self, batch):
        feed_dict = self._get_dict(batch)

        fetches = [self.to_fetch.y_hat, self.to_fetch.h]
        y_hat, _ = self.sess.run(fetches, feed_dict=feed_dict)

        return y_hat

    def summarize_model(self, batch):
        feed_dict = self._get_dict(batch)

        summary, gs = self.sess.run([self.to_fetch.summary, self.to_fetch.global_step],
                                    feed_dict=feed_dict)
        self.train_writer.add_summary(summary, global_step=gs)
        self.train_writer.flush()

    def summarize_metrics(self, summary, mode='train'):
        raise NotImplementedError("Check log/*/model-date/report for summaries.")

        # gs = self.sess.run(self.to_fetch.global_step)
        # for k, v in summary.items():
        #     self.ep_summary.value.add(tag=k, simple_value=v)
        # if mode == 'train':
        #     self.train_writer.add_summary(summary=self.ep_summary, global_step=gs)
        # else:
        #     self.test_writer.add_summary(summary=self.ep_summary, global_step=gs)
        # self.test_writer.flush()

    def save(self):
        try:
            self.saver.save(sess=self.sess, save_path=self._log_path + '/model/',
                            global_step=self.to_fetch.global_step,
                            write_meta_graph=False)

            logging.info(f"Model saved at global step {self.global_step}")
        except (ValueError, TypeError, RuntimeError) as e:
            raise e

    def reset(self):
        pass

    def restore(self, path=None):
        restore(self.saver, self.sess, path + '/model')

    @staticmethod
    def get_metrics(y, y_hat):
        metrics = regr_metrics(y, y_hat)
        return metrics

    @property
    def global_step(self):
        return self.sess.run(self.to_fetch.global_step)


@gin.configurable
class Runner(object):
    def __init__(self, steps=50, summarize_every=100, test_every=10, save_every=20, report_every=20):

        self.steps = steps
        self.summarize_every = summarize_every
        self.test_every = test_every
        self.save_every = save_every
        self.report_every = report_every

    def train(self, trainer, dataset, testset, logger):
        logging.log(logging.INFO, "Start training for {}".format(self.steps))
        ep = 0
        logger.register(stats=tuple(trainer.to_fetch.summary_list.keys()))
        for step in tqdm(range(self.steps), desc="steps"):
            try:
                batch, _ = dataset.next()
            except StopIteration:
                dataset.reset()
                ep += 1
            else:
                trainer.fit(batch, False)

                if step % self.summarize_every == 0:
                    trainer.summarize_model(batch)
                    logging.log(logging.INFO, f'Model summary at {step}')

                if step % self.save_every == 0:
                    trainer.save()
                    logging.log(logging.INFO, f'Model saved at {step}')

                if step % self.test_every == 0:
                    batch, _ = dataset.sample()
                    metrics, _ = trainer.score(batch)
                    logger.add(metrics)
                    # trainer.summarize_metrics(metrics, mode='train')
                    tqdm.write(f'TEST:t:{step}\t' + dict_to_str(metrics), end='\n')

                if step % self.report_every == 0:
                    _, metrics = self.test(trainer, testset, logger)
                    # trainer.summarize_metrics(metrics, mode='test')
                    tqdm.write(f'REPORT:t:{step}\t' + dict_to_str(metrics), end='\n')
                    testset.reset()
                    logger.reset()

        logger.dump()
        trainer.save()
        logging.info(f'Training Finished. Total epochs {ep}')

    def test(self, trainer, dataset, logger):
        logging.log(logging.INFO, "Start testing")
        preds = dict(dates=[], y_hat=[], y=[])
        logger.register(stats=tuple(trainer.to_fetch.summary_list.keys()))
        logger.reset(mode="test")
        t = 0
        while True:
            try:
                batch, dates = dataset.next()
                t += 1
            except StopIteration:
                break
            else:
                metrics, y_hat = trainer.score(batch)
                logger.add(metrics)
                for k, v in zip(preds.keys(), [dates['y'], y_hat, batch['y']]):
                    preds[k].append(v)

        df = build_preds(preds, path=logger.paths['report'] + '/{}'.format(trainer.global_step), stats=dataset.stats)
        score = trainer.get_metrics(df.y.values, df.y_hat.values)
        logger.dump()
        return df, score

    def predict(self, trainer, dataset, logger):
        logging.log(logging.INFO, "Start prediction")
        preds = dict(dates=[], y_hat=[], y=[])
        t = 0
        while True:
            try:
                batch, dates = dataset.next()
                t += 1
            except StopIteration:
                break
            else:
                y_hat = trainer.predict(batch)
                for k, v in zip(preds.keys(), [dates['y'], y_hat, batch['y']]):
                    preds[k].append(v)

        df = build_preds(preds, path=logger.paths['report'] + '/{}'.format(trainer.global_step), stats=dataset.stats)
        return df


@gin.configurable
class AdversarialRunner(Runner):
    def __init__(self, steps, g_steps=5, d_steps=1):
        super(AdversarialRunner, self).__init__(steps)
        self.g_steps = g_steps
        self.d_steps = d_steps

    def train(self, trainer, dataset, testset, logger):

        logging.log(logging.INFO, "Start training for {}".format(self.steps))
        ep = 0
        logger.register(stats=tuple(trainer.to_fetch.summary_list.keys()))

        for step in tqdm(range(self.steps)):
            try:
                batch, _ = dataset.next()
            except StopIteration:
                dataset.reset()
                ep += 1
            else:
                for d in range(self.d_steps):
                    trainer.fit(batch, target=False)

                for g in range(self.g_steps):
                    trainer.fit(batch, target=True)

                if step % self.summarize_every == 0:
                    trainer.summarize_model(batch)
                    logging.log(logging.INFO, f'Model summary at {step}')

                if step % self.save_every == 0:
                    trainer.save()
                    logging.log(logging.INFO, f'Model saved at {step}')

                if step % self.test_every == 0:
                    batch, _ = dataset.sample()
                    metrics, _ = trainer.score(batch)
                    logger.add(metrics)
                    # trainer.summarize_metrics(metrics, mode='train')
                    tqdm.write(f'TEST:t:{step}\t' + dict_to_str(metrics), end='\n')

                if step % self.report_every == 0:
                    _, metrics = self.test(trainer, testset, logger)
                    # trainer.summarize_metrics(metrics, mode='test')
                    tqdm.write(f'REPORT:t:{step}\t' + dict_to_str(metrics), end='\n')
                    testset.reset()
                    logger.reset()

            logger.dump()
            trainer.save()
            logging.log(logging.INFO, "Training finished.  Epoches: {}".format(ep))
