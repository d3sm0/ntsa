import logging

import gin.tf

from .blocks import FullyConnected, NeuralDecoder, NeuralEncoder, StochasticNeuralDecoder, Recurrent, \
    RecurrentEncoder, RecurrentDecoder, RGenerator, RDiscriminator, CDiscriminator, CGenerator, tf
from utils.tf_utils import sequence_loss, clip_grads, regr_metrics, summary_op, mask_output, reg_rnn, which_loss, \
    sn_block, get_z, train_fn


class Base(object):
    def __init__(self, input_shapes, config, scope="model"):
        self._input_shapes = input_shapes
        self._output_shape = input_shapes[1][-1]
        self._seq_len = input_shapes[0][0]
        self._pred_len = input_shapes[1][0]
        self._scope = scope
        self._config = config
        self._summary_dict = {}
        self._vars = []
        self._to_fetch = {}
        self._to_feed = {}
        self._built = False
        self._global_step = None

    def build(self, *args, **kwargs):
        self._global_step = tf.Variable(initial_value=0, dtype=tf.int32, name="global_step", trainable=False)
        self._ph_op()
        self._graph_op(*args, **kwargs)
        self._predict_op()
        self._vars = tf.trainable_variables()
        self._loss_op()
        self._train_op()
        self._summary_op()
        self._built = True
        tf.logging.log(logging.INFO, "Built model with scope {}".format(self._scope))

    def _ph_op(self):
        with tf.name_scope("init_ph"):
            x, y, y_feature = self._input_shapes

            # x driving series
            self.x = tf.placeholder(dtype=tf.float32, shape=(None,) + x, name='x')
            # future values of driving series
            self.y = tf.placeholder(dtype=tf.float32, shape=(None,) + y, name='y')
            # future values of the ancillary series
            self.y_features = tf.placeholder(dtype=tf.float32, shape=(None,) + y_feature, name='y_features')

            self.mu = tf.placeholder_with_default(0., shape=(), name='mu')
            self.std = tf.placeholder_with_default(1., shape=(), name='std')

            self.keep_prob = tf.placeholder_with_default(1., shape=(), name='keep_prob')
            self.is_training = tf.placeholder_with_default(True, shape=(), name='is_training')
            self.gen_len = tf.placeholder_with_default(1, shape=(), name='gen_len')
            self.flag = tf.placeholder(shape=(), dtype=tf.bool)

    def _build_op(self, *args, **kwargs):
        return

    def _graph_op(self, *args, **kwargs):
        with tf.variable_scope(self._scope, reuse=False):
            self._build_op(*args, **kwargs)

    def _predict_op(self):
        with tf.name_scope("predict_op"):
            self.y_hat = self.h

    def _loss_op(self):
        with tf.name_scope("loss_op"):

            weights = tf.ones_like(self.y, name='weights')
            self.loss = sequence_loss(self.y_hat, self.y, weights=weights, loss_fn=which_loss(self._config.loss))
            self._summary_dict.update({"loss": self.loss})

            if hasattr(self, '_reg'):
                reg = tf.reduce_sum(self._reg)
                self.loss += reg
                self._summary_dict.update({"loss": self.loss, "reg": reg})
            else:
                self._summary_dict.update({"loss": self.loss})

    @gin.configurable
    def _train_op(self):
        with tf.name_scope("train_op"):
            opt = train_fn(global_step=self._global_step)
            gvs, norm = clip_grads(self.loss, self.vars)

            # self.train = opt.apply_gradients(gvs, global_step=self._global_step)
            self.train = opt.minimize(self.loss, var_list=self.vars, global_step=self._global_step)
            self._summary_dict.update({"norm": norm})

    def _summary_op(self):
        with tf.name_scope("summary_op"):
            # self._summary_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            metrics = regr_metrics(y=self.y, y_hat=self.y_hat)
            metrics = {k: tf.reduce_mean(v) for k, v in metrics.items()}
            self._summary_dict.update(metrics)
            self.summary = summary_op(t_dict=self._summary_dict)

    def _project_output(self, h):
        if self._pred_len != self._seq_len:
            h = mask_output(h, pred_len=self._pred_len, seq_len=self._seq_len)
        return h

    @property
    def vars(self):
        return self._vars

    @property
    def global_step(self):
        return self._global_step

    @property
    def fetch_summary(self):
        return self._summary_dict

    @property
    def fetch_all(self):
        return self._to_fetch

    @property
    def config(self):
        return self._config

    @property
    def scope(self):
        return self._scope


class Dense(Base):
    def __init__(self, input_shapes, config):
        super(Dense, self).__init__(input_shapes, config=config, scope="dense")

    def _build_op(self):
        h = FullyConnected(output_shape=self._output_shape, scope=self._scope)(self.x, self.keep_prob)

        h = tf.layers.dense(tf.concat([h, self.y_features], axis=2), units=128, activation=tf.nn.tanh)
        h = tf.layers.dense(h, units=1, activation=None)

        self.h = self._project_output(h)


class SeriesNet(Base):
    def __init__(self, input_shapes, config):
        super(SeriesNet, self).__init__(input_shapes, config=config, scope="seriesnet")

        self._dilation = [2 ** idx for idx in range(self._config.d_rate)]

    def _build_op(self):
        h = self.x
        skips = []
        for d in self._dilation:
            h, skip = sn_block(h,
                               filters=self._config.filters,
                               kernel_size=self._config.kernel_size,
                               dilation=d,
                               scope="sn_block_{}".format(d))
            skips.append(skip)
            # may apply dropout to latest skip connection

        h = tf.add_n(skips)
        h = tf.nn.leaky_relu(h, alpha=0.1)
        self.h = self._project_output(h)


@gin.configurable
class CNP(Base):
    """The CNP model."""

    def __init__(self, input_shapes, config, cp=12, sample_cp=True, scope="cnp"):
        super(CNP, self).__init__(input_shapes, config, scope)
        self._cp = cp
        self._sample_cp = sample_cp

    @staticmethod
    def sample(samples, minval=3, maxval=24):
        return tf.distributions.Categorical(probs=(1 / (maxval - minval),) * (maxval - minval)).sample(samples)

    def _split(self):
        if self._sample_cp:
            idxs = self._sample(samples=self._cp, maxval=self._config.seq_len)
            c_x = tf.gather(self.x[:, :, 1:], idxs)
            c_y = tf.gather(self.x[:, :, :1], idxs)
        else:
            c_x = self.x[:, :self._cp, 1:]
            c_y = self.x[:, :self._cp, :1]
        return c_x, c_y

    def _build_op(self):

        x = self.x[:, :, 1:]
        c_x, c_y = self._split()

        r = NeuralEncoder(output_shape=None, scope="encoder")(c_x, c_y, keep_prob=self.keep_prob)

        r = tf.tile(r[:, None, :], (1, self.y_features.get_shape().as_list()[1], 1))
        h = tf.layers.dense(tf.concat([r, self.y_features], axis=2), units=128, activation=tf.nn.tanh)
        h = tf.layers.dense(h, units=1, activation=None)
        r = tf.reduce_mean(h, axis=1)

        self.d = NeuralDecoder(scope="decoder")(r, x, keep_prob=self.keep_prob)

        self.h = self.d.loc

    def _loss_op(self):
        self.loss = -tf.reduce_mean(self.d.log_prob(self.y))
        self._summary_dict.update({"loss": self.loss})


@gin.configurable
class NP(CNP):
    def __init__(self, input_shape, config, z_samples=1, latent=4, scope='np'):
        super(NP, self).__init__(input_shape, config, scope=scope)
        self.z_samples = z_samples
        self.latent = latent

    def _build_op(self):
        with tf.variable_scope(self._scope):
            c_x, c_y = self._split()
            x = self.x[:, :, 1:]
            y = self.x[:, :, :1]

            r = NeuralEncoder(scope="encoder")(c_x, c_y)
            self.c_z = get_z(r, self.latent, dist="normal")

            z = NeuralEncoder(scope="encoder")(x, y)
            self.z = get_z(z, latent_shape=self.latent)

            r = self.c_z.sample(self.z_samples)
            # r = tf.transpose(r, (1, 0, 2))
            # r = tf.reduce_mean(r, axis=1)

            self.d = StochasticNeuralDecoder(scope="decoder")(r, x)
            self.h = self.d.loc

    def _loss_op(self):
        kl = tf.reduce_sum(self.z.kl_divergence(self.c_z))
        log_lik = -tf.reduce_mean(self.d.log_prob(self.y))
        self.loss = log_lik + kl
        self._summary_dict.update({"loss": log_lik, "kl": kl})


class RNN(Base):
    def __init__(self, input_shapes, config, scope="rnn"):
        super(RNN, self).__init__(input_shapes, config=config, scope=scope)

    def _build_op(self):
        h = self.embedding(self.x, 128)

        encoder_output, encoder_state, states = Recurrent(output_shape=self._output_shape)(h, keep_prob=self.keep_prob)

        encoder_output = tf.layers.dense(tf.concat([encoder_output, self.y_features], axis=2), units=self._output_shape,
                                         activation=None)

        self.h = self._project_output(encoder_output)
        self._reg = self.reg_op([states])

    @classmethod
    def reg_op(cls, tensors):
        return reg_rnn(tensors)

    @classmethod
    def embedding(cls, h, units=None, kernel_size=None):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            if units == 0:
                return h
            if kernel_size is not None:
                h = conv1d(h, filters=units, kernel_size=kernel_size, act=tf.nn.tanh)
            else:
                h = tf.layers.dense(h, units=units, activation=tf.nn.tanh,
                                    kernel_initializer=tf.variance_scaling_initializer)
            return h


@gin.configurable
class Seq2Seq(RNN):
    def __init__(self, input_shapes, config, attn=None, scope="seq2seq"):
        super(Seq2Seq, self).__init__(input_shapes, config=config, scope=scope)
        self._attn = attn

    def _build_op(self):
        h = tf.layers.dense(self.x, units=128)

        encoder = RecurrentEncoder(output_shape=None, scope="encoder", attn=False)
        encoder_output, encoder_state, encoder_states = encoder(h, keep_prob=self.keep_prob)

        decoder = RecurrentDecoder(output_shape=self._output_shape, seq_len=self._pred_len, attn=self._attn, scope="decoder")

        decoder_output, decoder_state, decoder_states = decoder(x=self.x[:, -1, :1],
                                                                keep_prob=self.keep_prob,
                                                                encoder_states=encoder_states,
                                                                init_state=encoder_state,
                                                                y_features=self.y_features)
        self.h = self._project_output(decoder_output)

        self._reg = self.reg_op([encoder_states, decoder_states])


class DARNN(Seq2Seq):
    def __init__(self, input_shapes, config, attn="aligned"):
        super(DARNN, self).__init__(input_shapes,
                                    attn=attn,
                                    config=config,
                                    scope="darnn")
        self._config = config


@gin.configurable
class CGAN(Base):
    def __init__(self, input_shapes, config, latent=4, g_lr=1e-3, d_lr=1e-1, scope="cgan"):
        super(CGAN, self).__init__(input_shapes, config, scope=scope)
        self.latent = latent
        self.g_lr = g_lr
        self.d_lr = d_lr

    def _build_op(self):
        batch_size = tf.shape(self.y)[0]
        seq_len = self.y.get_shape().as_list()[1]
        x_shape = self.y.get_shape().as_list()[2]
        shape = [batch_size, seq_len, self.latent]
        z = tf.distributions.Normal(loc=0., scale=1.).sample(sample_shape=shape)

        g = CGenerator(output_shape=x_shape, scope="generator")
        x_fake = g(z)
        x_fake = tf.nn.tanh(x_fake)
        d = CDiscriminator(output_shape=1, scope="discriminator")  # 2 classes true or false
        self._true_d = d(self.y)
        self._fake_d = d(x_fake)
        self.h = x_fake[:, :, :1]
        self.x_fake = x_fake
        self.d = d

    @classmethod
    def _reg(cls, batch_size, d, x, x_fake, beta=1e-1):
        alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
        interpolates = alpha * x + (1 - alpha) * x_fake
        int_d = d(interpolates)
        gradients = tf.gradients(int_d, [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        return beta * tf.reduce_mean((slopes - 1) ** 2)

    @classmethod
    def _gen_norm(cls, y, y_hat):
        return tf.norm(y - y_hat)

    def _loss_op(self):
        with tf.name_scope("loss_op"):
            self.d_loss = tf.reduce_mean(self._fake_d) - tf.reduce_mean(self._true_d)
            self.g_loss = -tf.reduce_mean(self._fake_d)

            # reg = self._reg(tf.shape(self.x)[0], self.d, self.x, self.x_fake)
            # self.d_loss += reg
            self.loss = [self.d_loss, self.g_loss]

    def _train_op(self):
        with tf.name_scope("train_op"):
            d_opt = tf.train.GradientDescentOptimizer(self.d_lr)
            var_list = tf.trainable_variables(self.scope + "/discriminator")

            gvs, d_norm = clip_grads(self.d_loss, var_list)
            self.d_train = d_opt.minimize(self.d_loss, var_list=var_list, global_step=self._global_step)

            g_opt = tf.train.AdamOptimizer(self.g_lr)
            var_list = tf.trainable_variables(self.scope + "/generator")

            gvs, g_norm = clip_grads(self.g_loss, var_list)
            self.g_train = g_opt.minimize(self.g_loss, var_list=var_list, global_step=self._global_step)

            # g_train = g_opt.apply_gradients(gvs, global_step=self._global_step)

            self.train = tf.cond(self.flag, lambda: self.g_train, lambda: self.d_train)

            self._summary_dict.update(
                {"distance": self._gen_norm(self.x_fake, self.y),
                 "g_norm": g_norm,
                 "d_norm": d_norm,
                 "g_loss": self.g_loss,
                 "d_loss": self.d_loss
                 })


class RGAN(CGAN):
    def __init__(self, input_shapes, config, scope="rgan"):
        super(RGAN, self).__init__(input_shapes, config, scope=scope)

    def _build_op(self):
        batch_size = tf.shape(self.y)[0]
        seq_len = self.y.get_shape()[1]
        y_shape = self.y.get_shape()[2]
        y = self.y

        shape = [batch_size, seq_len, self.latent]
        z = tf.distributions.Normal(loc=0., scale=1.).sample(sample_shape=shape)
        # z = tf.concat([z, x_features], axis=2)
        g = RGenerator(output_shape=y_shape, scope="generator")
        x_fake, _, _ = g(z, keep_prob=self.keep_prob)
        x_fake = tf.nn.tanh(x_fake)  # i guess this is to bound the output of the lstm
        d = RDiscriminator(output_shape=1, scope="discriminator")

        self._true_d, _, _ = d(y)
        self._fake_d, _, _ = d(x_fake)
        self.h = x_fake[:, :, :1]
        self.x_fake = x_fake

    def _loss_op(self):
        with tf.name_scope("loss_op"):
            # labels = tf.distributions.Uniform(low=0.7, high=1.2).sample(tf.shape(self._true_d))
            labels = tf.ones_like(self._true_d)
            d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self._true_d, labels=labels,
            ))

            # labels = tf.distributions.Uniform(low=0., high=0.3).sample(tf.shape(self._fake_d))
            labels = tf.zeros_like(self._fake_d)
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self._fake_d, labels=labels
            ))

            self.d_loss = d_loss_true + d_loss_fake

            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self._fake_d, labels=tf.ones_like(self._fake_d)
            ))
            self.loss = [self.d_loss, self.g_loss]


class CRGAN(RGAN):
    def __init__(self, input_shapes, config, scope="crgan"):
        super(CRGAN, self).__init__(input_shapes, config, scope=scope)

    def _build_op(self):
        batch_size = tf.shape(self.y)[0]
        seq_len = self.y.get_shape()[1]
        y_shape = self.y.get_shape()[2]
        y = self.y

        shape = [batch_size, seq_len, self.latent]
        z = tf.distributions.Normal(loc=0., scale=1.).sample(sample_shape=shape)
        z = tf.concat([z, self.y_features], axis=2)
        g = RGenerator(output_shape=y_shape, scope="generator")
        x_fake, _, _ = g(z, keep_prob=self.keep_prob)
        x_fake = tf.nn.tanh(x_fake)  # i guess this is to bound the output of the lstm
        d = RDiscriminator(output_shape=1, scope="discriminator")

        self._true_d, _, _ = d(tf.concat([y, self.y_features], axis=2))
        self._fake_d, _, _ = d(tf.concat([x_fake, self.y_features], axis=2))
        self.h = x_fake[:, :, :1]
        self.x_fake = x_fake
