import logging

import tensorflow as tf

from utils.tf_utils import (summary_op, selu, clf_metrics, fc_block, sn_block, clip_grads, smape, regr_metrics,
                            conv1d_v2)

if not tf.VERSION == '1.10.1':
    tf.logging.log(tf.logging.WARN, "You should update tensorflow to 1.10.1")


class Base(object):
    def __init__(self, input_shapes, config, scope="model"):
        self._input_shapes = input_shapes
        self._proj_shape = 2 if config.loss == "clf" else 1  # suppose binary classification vs regression
        self._scope = scope
        self._config = config
        self._summary_list = []
        self._model_attr = {}

    def _build(self):
        self._global_step = tf.Variable(initial_value=0, dtype=tf.int32, name="global_step", trainable=False)
        self._init_ph()
        self._init_graph()
        self._predict_op()
        self._loss_op()
        self._train_op()
        self._summary_op()
        tf.logging.log(logging.INFO, "Built model with scope {}".format(self._scope))

    def _init_ph(self):
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

    def _init_graph(self):
        logging.log(logging.ERROR, "This method must be implemented by a subclass")
        raise NotImplementedError()

    def _predict_op(self):
        with tf.name_scope("predict_op"):
            if self._config.loss == "clf":
                self.y_hat = tf.argmax(self.h, axis=2, name='y_hat')
            else:
                self.y_hat = self.h * self.std + self.mu

    def _loss_op(self):
        with tf.name_scope("loss_op"):

            y = self.y
            if self._config.loss == "clf":
                y = tf.cast(y, dtype=tf.int32)
                y = tf.reshape(y, (-1,))
                h = tf.reshape(self.h, (-1, self._proj_shape))
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=h)
            else:
                h = self.h * self.std + self.mu
                h = tf.reshape(h, (-1, 1))
                y = tf.reshape(y, (-1, 1))
                if self._config.loss == "smape":
                    loss = smape(y, h)
                elif self._config.loss == "mae":
                    loss = tf.abs(y - h)
                elif self._config.loss == "mse":
                    loss = tf.square(y - h)
                else:
                    raise NotImplementedError()

            self.loss = tf.reduce_mean(loss, name=self._config.loss)

            if hasattr(self, '_reg'):
                reg = tf.reduce_sum(tf.add_n(self._reg, name="reg"))
                self.loss += reg
                self._summary_list += [self.loss, reg]
            else:
                self._summary_list += [self.loss]

    def _train_op(self):
        with tf.name_scope("train_op"):
            lr = tf.train.polynomial_decay(
                global_step=self.global_step,
                learning_rate=self._config.lr,
                end_learning_rate=1e-4,
                decay_steps=self._config.decay_steps,
                power=1 - self._config.decay_rate,
            )
            opt = tf.train.AdamOptimizer(learning_rate=lr)
            gvs, norm = clip_grads(self.loss, self.vars, clip=self._config.clip)
            self.train = opt.apply_gradients(gvs, global_step=self._global_step)
            self._summary_list += [norm]

    def _summary_op(self):
        with tf.name_scope("summary_op"):
            self._summary_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.summary = summary_op(t_list=self._summary_list)

            y = tf.reshape(self.y, (-1, self._config.pred_len))
            y_hat = tf.reshape(self.y_hat, (-1, self._config.pred_len))

            if self._config.loss == "clf":
                metrics = clf_metrics(y=y, y_hat=y_hat)
            else:
                metrics = regr_metrics(y=y, y_hat=y_hat)

            self.metrics = {k: tf.reduce_mean(v) for k, v in metrics.items()}

    def _project_output(self, h):
        if self._config.pred_len != self._config.seq_len:
            h = mask_output(h, pred_len=self._config.pred_len, seq_len=self._config.seq_len)
        if self._config.loss == "clf":
            h = tf.layers.dense(h, units=self._proj_shape, activation=None, name="logits", reuse=tf.AUTO_REUSE)
        return h

    @staticmethod
    def _reg_op(tensors, beta):
        return [beta * tf.nn.l2_loss(t) for t in tensors if 'kernel' in t.name or 'weight' in t.name],

    @property
    def vars(self):
        return tf.trainable_variables(scope=self._scope)

    @property
    def global_step(self):
        return self._global_step

    @property
    def model_attr(self):
        return self._model_attr


class Dense(Base):
    def __init__(self, input_shapes, config):
        super(Dense, self).__init__(input_shapes, config=config, scope="dense")

        self._config = config

    def _init_graph(self):
        with tf.variable_scope(self._scope):
            h = self.x
            for idx, (units, rate) in enumerate(self._config.layers):
                h = fc_block(h, units=units, act=selu, keep_prob=self.keep_prob, init=tf.initializers.variance_scaling,
                             scope="fc_block_{}".format(idx))

            h = mask_output(h, pred_len=self._config.pred_len, seq_len=self._config.seq_len)
            self.h = tf.layers.dense(h, units=self._proj_shape, kernel_initializer=tf.initializers.variance_scaling)


class SeriesNet(Base):
    def __init__(self, input_shapes, config):
        super(SeriesNet, self).__init__(input_shapes, config=config, scope="seriesnet")

        self._dilation = [2 ** idx for idx in range(self._config.d_rate)]

    def _init_graph(self):
        h = self.x
        skips = []
        with tf.variable_scope(self._scope):
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


class WaveNet(Base):
    # TODO debug me
    def __init__(self, input_shape, output_shape="clf"):
        super(WaveNet, self).__init__(input_shape, output_shape, scope="wavenet")

        self._dilation = [2 ** idx for idx in range(self._config.d_rate)] * self._config.layers
        self._kernel = [2 for idx in range(self._config.d_rate)] * self._config.layers

    def _init_graph(self):
        # x and x_features
        x = tf.reshape(self.x, shape=(-1, self._config.seq_len, self.x.get_shape()[-1]))
        x_fetures = tf.reshape(self.x_features, shape=(-1, self._config.seq_len, self.x_features.get_shape()[-1]))
        y_features = tf.reshape(self.y_features, shape=(-1, self._config.pred_len, self.y_features.get_shape()[-1]))

        y_hat, conv_inputs = self._encode(tf.concat([x, x_fetures], axis=2), scope="encode")
        # x + y_features
        self._encode(tf.concat([x, y_features], axis=1), scope="decode")
        self.y_hat = self._decode(y_hat, conv_inputs, y_features)

    def _encode(self, x, scope="encode"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = fc_block(x, units=self._config.res, act=tf.nn.tanh, scope="x_proj")
            skips = []
            conv_inputs = [x]
            for i, (dilation, filter_width) in enumerate(zip(self._dilation, self._kernel)):
                h = conv1d_v2(x, 2 * self._config.res, kernel_size=filter_width, dilation_rate=dilation,
                              scope='conv_{}'.format(i))

                f, g = tf.split(h, 2, axis=2)
                g = tf.nn.tanh(f) * tf.nn.sigmoid(g)

                h = fc_block(g, units=self._config.res + self._config.skip, scope="fc_{}".format(i))

                skip, residual = tf.split(h, [self._config.skip, self._config.res], axis=2)

                x += residual
                skips.append(skip)
                conv_inputs.append(x)

            h = tf.nn.relu(tf.concat(skips, axis=2))
            h = fc_block(h, units=self._config.encoder, act=tf.nn.relu, scope="proj_1")
            y_hat = fc_block(h, units=1, act=None, scope="proj_2")
            return y_hat, conv_inputs[:-1]

    def _decode(self, x, conv_inputs, features):
        batch_size = tf.shape(x)[0]

        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(conv_inputs, self._dilation)):
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            batch_idx = tf.reshape(batch_idx, [-1])

            queue_begin_time = self._config.seq_len - dilation - 1
            temporal_idx = tf.expand_dims(queue_begin_time, 1) + tf.expand_dims(tf.range(dilation), 0)
            temporal_idx = tf.reshape(temporal_idx, [-1])

            idx = tf.stack([batch_idx, temporal_idx], axis=1)
            slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, conv_input.get_shape()[2]))
            # shape(conv_input, 2)))

            layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation + self._config.pred_len)
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        features_ta = tf.TensorArray(dtype=tf.float32, size=self._config.pred_len)
        features_ta = features_ta.unstack(tf.transpose(features, (1, 0, 2)))

        # initialize output tensor array
        emit_ta = tf.TensorArray(size=self._config.pred_len, dtype=tf.float32)

        # initialize other loop vars
        elements_finished = 0 >= self._config.pred_len
        time = tf.constant(0, dtype=tf.int32)

        # get initial x input
        current_idx = tf.stack([tf.range(tf.shape(self._config.seq_len)[0]), self._config.seq_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx)
        dilations = self._dilation

        def loop_fn(time, current_input, queues):
            current_features = features_ta.read(time)
            current_input = tf.concat([current_input, current_features], axis=1)

            with tf.variable_scope('decode/x_proj', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                x_proj = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            skip_outputs, updated_queues = [], []
            for i, (conv_input, queue, dilation) in enumerate(zip(conv_inputs, queues, dilations)):
                state = queue.read(time)
                with tf.variable_scope('decode/conv_{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights'.format(i))
                    b_conv = tf.get_variable('biases'.format(i))
                    dilated_conv = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(x_proj, w_conv[1, :, :]) + b_conv
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

                with tf.variable_scope('decode/fc_{}'.format(i), reuse=True):
                    w_proj = tf.get_variable('weights'.format(i))
                    b_proj = tf.get_variable('biases'.format(i))
                    concat_outputs = tf.matmul(dilated_conv, w_proj) + b_proj
                skips, residuals = tf.split(concat_outputs, [self._config.skip, self._config.res], axis=1)

                x_proj += residuals
                skip_outputs.append(skips)
                updated_queues.append(queue.write(time + dilation, x_proj))

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            with tf.variable_scope('decode/proj_1', reuse=True):
                w_h = tf.get_variable('weights')
                b_h = tf.get_variable('biases')
                h = tf.nn.relu(tf.matmul(skip_outputs, w_h) + b_h)

            with tf.variable_scope('decode/proj_2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                y_hat = tf.matmul(h, w_y) + b_y

            elements_finished = (time >= self._config.pred_len)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 1], dtype=tf.float32),
                lambda: y_hat
            )
            next_elements_finished = (time >= self._config.pred_len - 1)

            return (next_elements_finished, next_input, updated_queues)

        def condition(unused_time, elements_finished, *_):
            return tf.logical_not(tf.reduce_all(elements_finished))

        def body(time, elements_finished, emit_ta, *state_queues):
            (next_finished, emit_output, state_queues) = loop_fn(time, initial_input, state_queues)

            emit = tf.where(elements_finished, tf.zeros_like(emit_output), emit_output)
            emit_ta = emit_ta.write(time, emit)

            elements_finished = tf.logical_or(elements_finished, next_finished)
            return [time + 1, elements_finished, emit_ta] + list(state_queues)

        returned = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[time, elements_finished, emit_ta] + state_queues
        )

        outputs_ta = returned[2]
        y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))
        return y_hat


class NP(Base):
    def __init__(self, input_shape, config, scope='np'):
        super(NP, self).__init__(input_shape, config, scope=scope)
        self._config.dist = "bernoulli" if self._config.loss == "clf" else "normal"
        # TODO this doesn't quite work yet

    def _init_graph(self):
        with tf.variable_scope(self._scope):
            x = self.x
            y = self.y
            y = (y - self.mu) / self.std

            batch_size = tf.shape(x)[0]
            # there is a better way to sample
            c = tf.distributions.Uniform(1., tf.cast(batch_size, dtype=tf.float32)).sample()
            c = tf.cast(c, dtype=tf.int32)
            s = (tf.range(0, batch_size, 1) < c)

            x_context, x_target = mask_tensor(x, s)
            y_context, y_target = mask_tensor(y, s)

            with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
                r = map_xr(x, y, units=self._config.encoder, output_shape=self._config.embedding,
                           h_dim=self._config.h_dim)
                z_all = get_z(r, z=self._config.latent)

                r = map_xr(x_context, y_context, units=self._config.encoder, output_shape=self._config.embedding,
                           h_dim=self._config.h_dim)
                z_context = get_z(r, z=self._config.latent)

                q = get_dist(z_all, dist=self._config.dist)
                p = get_dist(z_context, dist=self._config.dist)
                self.z_sample = q.sample(sample_shape=self._config.samples)

            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                g = map_zx(self.z_sample, x_target, units=self._config.decoder)
                g = self._project_output(g)
                z_hat = get_z(g, self._proj_shape, dist=self._config.dist)
                d = get_dist(z_hat, dist=self._config.dist)

            with tf.variable_scope("loss"):
                kl = tf.reduce_sum(q.kl_divergence(p))
                log_lik = tf.reduce_mean(tf.reduce_sum(d.log_prob(y_target), axis=0))
                self.loss = -log_lik + kl
                self._reg = kl
                self._summary_list += [log_lik, kl]

    def _loss_op(self):
        # Loss computation are carried in init graph
        pass

    def _predict_op(self):
        with tf.variable_scope(self._scope):
            # x is current value
            # y is the estimate
            x = self.x  # assuming y_features is like x but lagged of 12 values
            y = self.x[:, -1:, :1]
            y = tf.tile(y, (1, self.x.get_shape()[1], 1))
            x_target = self.x

            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                # i need x,y to sample the latent factor
                h = map_xr(x, y, units=self._config.encoder, output_shape=self._config.embedding,
                           h_dim=self._config.h_dim)
                z_all = get_z(h, z=self._config.latent)
                z_sample = get_dist(z_all, dist=self._config.dist).sample(self._config.samples)
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                g = map_zx(z_sample, x_target, units=self._config.decoder)
                g = self._project_output(g)
                z_hat = get_z(g, self._proj_shape, dist=self._config.dist)
                self.y_hat = z_hat[0] * self.std + self.mu
                self.h = z_hat[0]


def map_xr(x, y, units, output_shape, h_dim=3):
    seq_len = x.get_shape()[1]
    pred_len = y.get_shape()[1]
    if seq_len != pred_len:
        y = tf.tile(y, (1, seq_len, 1))

    h = tf.concat([x, y], axis=2)

    h = tf.layers.dense(h, units, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, name='h_0')
    h = tf.layers.dense(h, output_shape, activation=None, reuse=tf.AUTO_REUSE, name='h_1')

    h = tf.reduce_mean(h, axis=tf.range(0, h_dim, delta=1))  # last dimension is the "time" in the latent space
    h = tf.reshape(h, (-1, 1))

    return h


def map_zx(z, x_target, units):
    # [n_draws, batch_size, seq_len,dim_z]

    batch_size = tf.shape(x_target)[0]
    samples = z.get_shape()[0]

    z_tile = x_target.get_shape()[1] - z.get_shape()[1] + 1

    z = tf.expand_dims(z, axis=1)
    z = tf.tile(z, (1, batch_size, z_tile, 1))
    x_target = tf.expand_dims(x_target, axis=0)
    x_target = tf.tile(x_target, (samples, 1, 1, 1))

    h = tf.concat([x_target, z], axis=3)

    # change decoder here
    h = tf.layers.dense(h, units, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, name='d_0')
    h = tf.reduce_mean(h, axis=0)  # average over samples
    return h


def get_z(r, z, dist="normal"):
    if dist == "normal":
        mu = tf.layers.dense(r, z, name='mu', reuse=tf.AUTO_REUSE)
        sigma = tf.layers.dense(r, z, name='std', reuse=tf.AUTO_REUSE, activation=tf.nn.softplus)
        return (mu, sigma)
    elif dist == "bernoulli":
        mu = tf.layers.dense(r, z, name='mu', reuse=tf.AUTO_REUSE)
        return (mu,)
    else:
        raise NotImplementedError()


class RNN(Base):
    def __init__(self, input_shapes, config, scope="rnn"):

        super(RNN, self).__init__(input_shapes, config=config, scope=scope)

    def _init_graph(self):
        if self._config.loss == "clf" and self._config.ar is True:
            raise ValueError("Can't use classification loss with autoregressive model, change to mae.")

        with tf.variable_scope(self._scope):
            h = self.x
            encoder_output, encoder_state, states = self._encoder(h)
            self.h = self._project_output(encoder_output)
        self._reg = self._reg_op([states], self._config.alpha, self._config.beta)

    def _encoder(self, h, encoder_state=None):

        with tf.variable_scope('encoder'):
            cell = tf.contrib.rnn.GRUBlockCellV2(num_units=self._config.encoder)

            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,
                                                 state_keep_prob=self.keep_prob)
            if not encoder_state:
                encoder_state = cell.zero_state(batch_size=tf.shape(h)[0], dtype=tf.float32)

            if self._config.attn:
                attn = InputAttention(output_shape=self._proj_shape, memory=h, use_bias=True,
                                      project_output=True)
            else:
                attn = Attention(output_shape=self._proj_shape, use_bias=True,
                                 project_output=True)

            encoder_output, encoder_state, states = encode(h, cell, encoder_state=encoder_state,
                                                           seq_len=self._config.seq_len, time_first=False,
                                                           attn=attn)
            encoder_output.set_shape((None, self._config.seq_len, encoder_output.get_shape()[-1]))
        return encoder_output, encoder_state, states

    @staticmethod
    def _reg_op(tensors, alpha=0., beta=0.):
        return [(activation_loss(t, alpha), stability_loss(t, beta)) for t in tensors]


class Seq2Seq(RNN):
    def __init__(self, input_shapes, config, scope="seq2seq"):
        super(Seq2Seq, self).__init__(input_shapes, config=config, scope=scope)

    def _decoder(self, h, last_y, decoder_state=None, states=None):
        with tf.variable_scope('decoder'):
            cell = tf.contrib.rnn.GRUBlockCellV2(num_units=self._config.decoder)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob,
                                                 state_keep_prob=self.keep_prob)
            if decoder_state is None:
                decoder_state = cell.zero_state(batch_size=tf.shape(h)[0], dtype=tf.float32)

            if self._config.attn and states is not None:
                attn = TimeAttention(input_shape=self._config.seq_len, output_shape=self._proj_shape,
                                     memory=states)
            else:
                attn = Attention(output_shape=self._proj_shape, project_output=True)

            decoder_output, decoder_state, decoder_states = decode(cell,
                                                                   last_y=last_y,
                                                                   h=h,
                                                                   decoder_state=decoder_state,
                                                                   seq_len=self._config.pred_len,
                                                                   attn=attn)
            return decoder_output, decoder_state, decoder_states

    def _init_graph(self):
        with tf.variable_scope(self._scope):
            h = self.x
            h, last_state, encoder_states = self._encoder(h)
            # decoder takes last_x as initial value, make sure the main feature in the first column
            out, last_state, decoder_states = self._decoder(self.y_features,
                                                            self.x[:, -1, :1], decoder_state=last_state,
                                                            states=encoder_states)

            self.h = self._project_output(out)

        self._reg = RNN._reg_op([encoder_states, decoder_states], alpha=self._config.alpha, beta=self._config.beta)


class DARNN(Seq2Seq):
    def __init__(self, input_shapes, config):
        super(DARNN, self).__init__(input_shapes,
                                    config=config,
                                    scope="darnn")
        self._config = config


class Attention(object):
    def __init__(self, output_shape=1, memory=None, scale=False, use_bias=True, project_output=False,
                 name='attention'):
        self._memory = memory
        self._scale = scale
        self._name = name
        self._output_shape = output_shape
        self._project = self.projection_layer(output_shape, use_bias=use_bias) if project_output else lambda x: x

    def apply(self, query, state):
        return query

    def project(self, state):
        return self._project(state)

    @staticmethod
    def projection_layer(output_shape, use_bias=True):
        with tf.variable_scope("proj_layer"):
            h = tf.layers.Dense(units=output_shape, use_bias=use_bias, name='fc',
                                kernel_initializer=tf.variance_scaling_initializer)
            return h


class InputAttention(Attention):

    def __init__(self,
                 output_shape,
                 memory,
                 scale=False,
                 name='input_attention',
                 use_bias=False,
                 project_output=False):
        super(InputAttention, self).__init__(
            output_shape=output_shape,
            memory=tf.transpose(memory, (0, 2, 1)),
            name=name,
            scale=scale,
            project_output=project_output
        )

        self._memory_layer = tf.layers.Dense(units=1, use_bias=use_bias, name='memory', _reuse=tf.AUTO_REUSE)
        self._input_shape = memory.get_shape()[-1]

    def apply(self, query, state):
        with tf.variable_scope(self._name, values=[query]):
            x = tf.concat([
                tf.tile(tf.expand_dims(state, axis=1), (1, self._input_shape, 1)),
                self._memory], axis=2)  # batch_size, input_shape, cell_size + seq_len
            x = self._memory_layer(x)
            x = tf.squeeze(x, axis=-1)
            alpha = tf.nn.softmax(x)
            x_tilde = tf.multiply(alpha, query)
            return x_tilde


class TimeAttention(Attention):

    def __init__(self, input_shape, output_shape, memory, scale=False, name='time_attention', use_bias=False):
        super(TimeAttention, self).__init__(output_shape, memory, scale=scale, name=name, project_output=True,
                                            use_bias=True)
        self._input_shape = input_shape
        self._memory_layer = self.memory_layer(input_shape, use_bias=use_bias)
        self._query_layer = tf.layers.Dense(units=1, use_bias=use_bias, name='query_layer',
                                            kernel_initializer=tf.variance_scaling_initializer)

    def apply(self, query, state):
        # batch_size, seq_len, encoder + decoder_state
        context = self.context(state)
        y_tilde = self._query_layer(tf.concat([query, context], axis=1))
        return y_tilde

    def context(self, state):
        # none, 25, 64
        x = tf.concat([
            tf.tile(tf.expand_dims(state, axis=1), (1, self._input_shape, 1)),
            self._memory], axis=2)
        beta = self._memory_layer(x)  # None, 25
        context = tf.matmul(tf.expand_dims(tf.squeeze(beta, axis=-1), axis=1), self._memory)[:, 0, :]  # None, cell_size
        return context

    @staticmethod
    def memory_layer(input_shape, use_bias=False):
        with tf.variable_scope('beta', reuse=tf.AUTO_REUSE, initializer=tf.initializers.variance_scaling):
            fc_0 = tf.layers.Dense(units=input_shape, use_bias=use_bias, activation=tf.nn.tanh, name='fc_0')
            fc_1 = tf.layers.Dense(units=1, use_bias=use_bias, activation=tf.nn.softmax, name='fc_1')
        return lambda x: fc_1(fc_0(x))


def encode(h, cell, encoder_state=None, seq_len=24, time_first=False, attn=None):
    def cond_stop(time, prev_state, output, states):
        return time < seq_len

    def loop_fn(time, prev_state, output, states):
        x_tilde = attn.apply(h[:, time, :], prev_state)  # input_attention(time, prev_state)
        _, state = cell(x_tilde, prev_state)
        out = attn.project(state)
        output = output.write(time, out)
        states = states.write(time, state)
        return time + 1, state, output, states

    init_cond = [tf.constant(0, dtype=tf.int32),
                 encoder_state,
                 tf.TensorArray(dtype=tf.float32, size=seq_len),
                 tf.TensorArray(dtype=tf.float32, size=seq_len)
                 ]
    _, state, output, states = tf.while_loop(cond_stop, loop_fn, init_cond, parallel_iterations=32)
    output = output.stack()
    states = states.stack()
    if not time_first:
        output = tf.transpose(output, (1, 0, 2))
        states = tf.transpose(states, (1, 0, 2))

    return output, state, states


def decode(cell, last_y, h=None, decoder_state=None, seq_len=24, time_first=False, attn=None):
    def cond_stop(time, prev_output, prev_state, output, states):
        return time < seq_len

    def loop_fn(time, prev_output, prev_state, output, states):
        y_tilde = attn.apply(tf.concat([prev_output, h[:, time, :]], axis=1), prev_state)
        # y_tilde = attn.apply(prev_output, prev_state)
        out, state = cell(y_tilde, prev_state)
        out = attn.project(out)
        output = output.write(time, out)
        states = states.write(time, state)
        return time + 1, out, state, output, states

    init_cond = [tf.constant(0, dtype=tf.int32),
                 last_y,
                 decoder_state,
                 tf.TensorArray(dtype=tf.float32, size=seq_len),
                 tf.TensorArray(dtype=tf.float32, size=seq_len)
                 ]
    _, _, state, output, states = tf.while_loop(cond_stop, loop_fn, init_cond, parallel_iterations=32)
    output = output.stack()
    states = states.stack()
    if not time_first:
        output = tf.transpose(output, (1, 0, 2))
        states = tf.transpose(states, (1, 0, 2))
    return output, state, states


def activation_loss(h, beta):
    if beta == 0.0:
        return 0.0
    else:
        return tf.nn.l2_loss(h) * (beta)


def stability_loss(h, beta):
    if beta == 0.0:
        return 0.0
    else:
        l2 = tf.sqrt(tf.reduce_sum(tf.square(h), axis=-1))
        return beta * tf.reduce_mean(tf.square(l2[1:] - l2[:-1]))


def get_dist(params, dist="normal"):
    if dist == "normal":
        return tf.distributions.Normal(*params)
    elif dist == "bernoulli":
        return tf.distributions.Bernoulli(logits=params[0], dtype=tf.float32)
    else:
        raise NotImplementedError()


def mask_tensor(x, s):
    not_x = tf.boolean_mask(x, tf.logical_not(s))
    x = tf.boolean_mask(x, s)
    return x, not_x


def mask_output(h, seq_len, pred_len):
    assert h.get_shape().ndims == 3
    if seq_len != pred_len:
        h = h[:, -pred_len:, :]
    return h

