import logging

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from utils.tf_utils import (summary_op, selu, clf_metrics, fc_block, sn_block, clip_grads, smape, regr_metrics,
                            conv1d)


def assert_shape(x, template):
    if x.get_shape().tolist() != template:
        raise ValueError("Tensors has different shape")


def sequence_loss(y, y_hat, weights, loss_fn, avg_time=True, avg_batch=True):
    loss = loss_fn(y, y_hat)
    total_size = tf.convert_to_tensor(1e-12)
    if avg_batch and avg_time:
        loss = tf.reduce_sum(loss)
        total_size += tf.reduce_sum(weights)
    elif avg_batch and not avg_time:
        loss = tf.reduce_sum(loss, axis=0)
        total_size += tf.reduce_sum(loss, axis=0)
    else:
        loss = tf.reduce_sum(loss, axis=1)
        total_size = tf.reduce_sum(loss, axis=1)

    loss = tf.divide(loss, total_size, name=loss.name)
    return tf.losses.add_loss(loss)


class Base(object):
    def __init__(self, input_shapes, config, scope="model"):
        self._input_shapes = input_shapes
        self._output_shape = input_shapes[1][-1]
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
            self.y_hat = self.h * self.std + self.mu

    def _loss_op(self):
        with tf.name_scope("loss_op"):

            weights = tf.ones_like(self.y, name='weights')
            self.loss = sequence_loss(self.y, self.y_hat, weights=weights, loss_fn=which_loss(self._config.loss))

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
            opt = tf.train.RMSPropOptimizer(learning_rate=lr)
            gvs, norm = clip_grads(self.loss, self.vars, clip=self._config.clip)
            self.train = opt.apply_gradients(gvs, global_step=self._global_step)
            self._summary_list += [norm]

    def _summary_op(self):
        with tf.name_scope("summary_op"):
            self._summary_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.summary = summary_op(t_list=self._summary_list)

            y = tf.reshape(self.y, (-1, self._config.pred_len))
            y_hat = tf.reshape(self.y_hat, (-1, self._config.pred_len))

            metrics = regr_metrics(y=y, y_hat=y_hat)

            self.metrics = {k: tf.reduce_mean(v) for k, v in metrics.items()}

    def _project_output(self, h):
        if self._config.pred_len != self._config.seq_len:
            h = mask_output(h, pred_len=self._config.pred_len, seq_len=self._config.seq_len)
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

    def _init_graph(self):
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            h = self.x
            h = FullyConnected(config=self._config, scope=self._scope)(h)
            self.h = self._project_output(h)


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
    def __init__(self, input_shapes, config, scope="wavenet"):
        super(WaveNet, self).__init__(input_shapes, config=config, scope=scope)

        self._dilation = [2 ** idx for idx in range(self._config.d_rate)] * self._config.layers
        self._kernel = [self._config.kernel_size for idx in range(self._config.d_rate)] * self._config.layers

    def _init_graph(self):
        # x and x_features
        x = self.x[:, :, :1]
        x_features = self.x[:, :, 1:]
        y_features = self.y_features

        with tf.variable_scope(self._scope):
            y_hat, conv_inputs = self._encode(tf.concat([x, x_features], axis=2), scope="encode")
            # x + y_features
            self._encode(tf.concat([x, y_features], axis=2), scope='decode')
            # self._decode(tf.concat([x, y_features], axis=2), scope="decode")
            h = self._decoder(y_hat, conv_inputs, y_features)
            self.h = self._project_output(h)

    def _encode(self, x, scope="encode"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = fc_block(x, units=self._config.res, act=tf.nn.tanh, scope="x_proj")
            skips = []
            conv_inputs = [x]
            for i, (dilation, filter_width) in enumerate(zip(self._dilation, self._kernel)):
                h = conv1d(x, 2 * self._config.res, kernel_size=filter_width, dilation_rate=dilation,
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

    def _decode(self, x, scope='decode'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs = fc_block(
                x,
                units=self._config.res,
                act=tf.nn.tanh,
                scope='x-proj'
            )

            skip_outputs = []
            conv_inputs = [inputs]
            for i, (dilation, filter_width) in enumerate(zip(self._dilation, self._kernel)):
                dilated_conv = conv1d(
                    inputs,
                    filters=2 * self._config.res,
                    kernel_size=filter_width,
                    dilation_rate=[dilation],
                    scope='conv_{}'.format(i)
                )
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
                dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

                outputs = fc_block(
                    dilated_conv,
                    self._config.skip + self._config.res,
                    scope='fc_{}'.format(i)
                )
                skips, residuals = tf.split(outputs, [self._config.skip, self._config.res], axis=2)

                inputs += residuals
                conv_inputs.append(inputs)
                skip_outputs.append(skips)

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
            h = fc_block(skip_outputs, 128, scope='proj_1', act=tf.nn.relu)
            y_hat = fc_block(h, 1, scope='proj_2')
            return y_hat

    def _decoder(self, x, conv_inputs, features):
        batch_size = tf.shape(x)[0]
        encoder_len = tf.tile(tf.expand_dims(x.get_shape()[1], 0), (batch_size,))
        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(conv_inputs, self._dilation)):
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            batch_idx = tf.reshape(batch_idx, [-1])

            queue_begin_time = encoder_len - dilation - 1
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
        current_idx = tf.stack([tf.range(tf.shape(encoder_len)[0]), encoder_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx)
        dilations = self._dilation

        def loop_fn(time, current_input, queues):
            current_features = features_ta.read(time)
            current_input = tf.concat([current_input, current_features], axis=1)

            with tf.variable_scope('decode/x_proj', reuse=True):
                w_x_proj = tf.get_variable('kernel')
                b_x_proj = tf.get_variable('bias')
                x_proj = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            skip_outputs, updated_queues = [], []
            for i, (conv_input, queue, dilation) in enumerate(zip(conv_inputs, queues, dilations)):
                state = queue.read(time)
                with tf.variable_scope('decode/conv_{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('kernel'.format(i))
                    b_conv = tf.get_variable('bias'.format(i))
                    dilated_conv = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(x_proj, w_conv[1, :, :]) + b_conv
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter) * tf.nn.sigmoid(conv_gate)

                with tf.variable_scope('decode/fc_{}'.format(i), reuse=True):
                    w_proj = tf.get_variable('kernel'.format(i))
                    b_proj = tf.get_variable('bias'.format(i))
                    concat_outputs = tf.matmul(dilated_conv, w_proj) + b_proj
                skips, residuals = tf.split(concat_outputs, [self._config.skip, self._config.res], axis=1)

                x_proj += residuals
                skip_outputs.append(skips)
                updated_queues.append(queue.write(time + dilation, x_proj))

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            with tf.variable_scope('decode/proj_1', reuse=True):
                w_h = tf.get_variable('kernel')
                b_h = tf.get_variable('bias')
                h = tf.nn.relu(tf.matmul(skip_outputs, w_h) + b_h)

            with tf.variable_scope('decode/proj_2', reuse=True):
                w_y = tf.get_variable('kernel')
                b_y = tf.get_variable('bias')
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


class CNP(Base):
    """The CNP model."""

    def __init__(self, input_shapes, config, scope="cnp"):
        super(CNP, self).__init__(input_shapes, config, scope)

    def _init_graph(self):
        context_point = tf.random_uniform(shape=(self._config.context_points,), minval=3, maxval=self._config.seq_len)

        x = self.x[:, :, 1:]
        y = self.x[:, :, 1:]

        c_x = tf.gather(x, indices=context_point, axis=1)
        c_y = tf.gather(y, indices=context_point, axis=1)

        r = NeuralEncoder(self._config)(c_x, c_y)
        self.h, self.sigma = NeuralDecoder(self._config)(r, x)

    def _loss_op(self):
        dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=self.h, scale_diag=self.sigma)

        self.loss = tf.reduce_mean(-dist.log_prob(self.y))


class NP(Base):
    def __init__(self, input_shape, config, scope='np'):
        super(NP, self).__init__(input_shape, config, scope=scope)
        # self._config.dist = "bernoulli" if self._config.loss == "clf" else "normal"

    def _init_graph(self):
        with tf.variable_scope(self._scope):
            context_point = tf.random_uniform(shape=(self._config.context_points,), minval=3,
                                              maxval=self._config.seq_len)

            x = self.x[:, :, 1:]
            y = self.x[:, :, :1]

            c_x = tf.gather(x, indices=context_point, axis=1)
            c_y = tf.gather(y, indices=context_point, axis=1)

            r = NeuralEncoder(self._config)(c_x, c_y)
            self.c_z = get_z(r, 1)
            self.z = NeuralEncoder(self._config)(x, y)
            r = self.c_z.sample(1)

            self.d = NeuralDecoder(self._config)(r, x)

    def _loss_op(self):
        kl = tf.reduce_sum(self.z.kl_divergence(self.c_z))
        log_lik = tf.reduce_mean(-self.d.log_prop(self.y))
        self.loss = log_lik + kl


# tf.layers.Dense
# Learner/ Architecture/ Model/ Units


def build_cell(config):
    Cell = tf.nn.rnn_cell.LSTMCell if config.layers.cell == "lstm" else tf.nn.rnn_cell.GRUCell
    cells = []
    for layer, units in enumerate(config.layers):
        cell = Cell()
        if config.dropout is not None:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell)
        cells.append(cell)

    if len(config.layers) > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    return cell


# kernel init, func activation, cell type, dropout
class RNN(Base):
    def __init__(self, input_shapes, config, scope="rnn"):
        super(RNN, self).__init__(input_shapes, config=config, scope=scope)

    def _init_graph(self):
        with tf.variable_scope(self._scope):
            encoder_output, encoder_state, states = Recurrent(output_shape=self._output_shape,
                                                              config=self._config)(self.x)
            self.h = self._project_output(encoder_output)
        self._reg = self._reg_op([states], self._config.alpha, self._config.beta)

    @staticmethod
    def _reg_op(tensors, alpha=0., beta=0.):
        return [(activation_loss(t, alpha), stability_loss(t, beta)) for t in tensors]


class Seq2Seq(RNN):
    def __init__(self, input_shapes, config, scope="seq2seq"):
        super(Seq2Seq, self).__init__(input_shapes, config=config, scope=scope)

    def _init_graph(self):
        with tf.variable_scope(self._scope):
            encoder_output, encoder_state, encoder_states = Recurrent(output_shape=self._output_shape,
                                                                      config=self._config)(self.x)
            # decoder takes last_x as initial value, make sure the main feature in the first column
            decoder_output, decoder_state, decoder_states = RecurrentDecoder(output_shape=self._output_shape,
                                                                             config=self._config)(self.y_features,
                                                                                                  self.x[:, -1, :1],
                                                                                                  encoder_states)
            self.h = self._project_output(decoder_output)

        self._reg = RNN._reg_op([encoder_states, decoder_states], alpha=self._config.alpha, beta=self._config.beta)


class DARNN(Seq2Seq):
    def __init__(self, input_shapes, config):
        super(DARNN, self).__init__(input_shapes,
                                    config=config,
                                    scope="darnn")
        self._config = config


class Attention(object):
    def __init__(self, output_shape=None, memory=None, use_bias=True, name='attention'):
        self._memory = memory
        self._name = name
        self._output_shape = output_shape
        self._project = self.projection_layer(output_shape, use_bias=use_bias) if output_shape is not None else lambda \
                x: x

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
                 name='input_attention',
                 use_bias=False):
        super(InputAttention, self).__init__(
            output_shape=output_shape,
            memory=tf.transpose(memory, (0, 2, 1)),
            name=name,
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

    def __init__(self, input_shape, output_shape, memory, name='time_attention', use_bias=False):
        super(TimeAttention, self).__init__(output_shape=output_shape,
                                            memory=memory,
                                            use_bias=True,
                                            name=name)
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


class Block(object):
    def __init__(self, config, scope="block"):
        self._built = False
        self._config = config
        self._scope = scope
        self._vars = []
        self._endpoints = {}  # all the outputs of the model
        self._output_shape = config.layers[-1]

    def __call__(self, x, *args, **kwargs):
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            if not self._built:
                self.build()
            out = self.call(x, *args, **kwargs)
            return out

    def call(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented locally")

    def build(self):
        self._built = True

    @property
    def scope(self):
        return self._scope

    @property
    def config(self):
        return self._config

    @property
    def vars(self):
        return self._vars


class FullyConnected(Block):
    def __init__(self, config, scope="dense_block"):
        super(FullyConnected, self).__init__(config, scope)

    def call(self, x):
        h = x
        for layer, unit in enumerate(self.config.layers):
            h = tf.layers.dense(x, units=unit, activation=self.config.act, kernel_initializer=self.config.kernel_init,
                                name=f'fc_"{layer}"')
        return h


class Conv1D(Block):
    def __init__(self, config, scope="conv_block"):
        super(Conv1D, self).__init__(config, scope)

    def call(self, x):
        h = x
        for layer, filters, kernel_size in enumerate(self.config.layers):
            h = conv1d(h, filters, kernel_size, scope=f'conv_"{layer}"')
        return h


class Recurrent(Block):
    def __init__(self, output_shape, config, scope="recurrent_encoder"):
        # unit Recurrent, encode(h)
        super(Recurrent, self).__init__(config, scope)
        self._output_shape = output_shape
        self._cell = build_cell(config)

    def call(self, x):
        attn = InputAttention(output_shape=self._output_shape, memory=x) if self._config.attn else Attention(
            output_shape=self._output_shape)
        init_state = self._cell.zero_state(batch_size=tf.shape(x), dtype=tf.float32)
        output, state, states = encode(x, self._cell, encoder_state=init_state, time_first=True, attn=attn,
                                       seq_len=self._config.seq_len)
        return output, state, states


class RecurrentEncoder(Recurrent):
    def __init__(self, output_shape, config, scope="recurrent_encoder"):
        # unit Recurrent, encode(h)
        super(RecurrentEncoder, self).__init__(output_shape, config, scope)


class RecurrentDecoder(Block):
    def __init__(self, output_shape, config, scope="recurrent_decoder"):
        # unit Recurrent, encode(h)
        super(RecurrentDecoder, self).__init__(config, scope)
        self._output_shape = output_shape
        self._cell = build_cell(config)

    def call(self, inputs):
        x, last_y, encoder_states = inputs
        attn = TimeAttention(input_shape=x.get_shape()[-1],
                             output_shape=self._output_shape,
                             memory=encoder_states) if self._config.attn else Attention(
            output_shape=self._output_shape)

        init_state = self._cell.zero_state(batch_size=tf.shape(x), dtype=tf.float32)

        output, state, states = decode(x, last_y, self._cell, decoder_state=init_state, attn=attn,
                                       seq_len=self._config.seq_len)
        return output, state, states


class NeuralEncoder(Block):
    def __init__(self, config, scope="neural_encoder"):
        super(NeuralEncoder, self).__init__(config, scope)

    def call(self, x, y):
        h = tf.concat([x, y], axis=2)
        for layer, unit in enumerate(self.config.layers):
            h = tf.layers.dense(x, units=unit, activation=self.config.act,
                                kernel_initializer=self.config.kernel_init,
                                name=f'fc_"{layer}"')

        h = tf.reduce_mean(h, axis=1)
        return h


class NeuralDecoder(Block):
    def __init__(self, config, scope="neural_decoder"):
        super(NeuralDecoder, self).__init__(config, scope)

    def call(self, r, x):
        seq_len = x.get_shape().as_list()[1]
        r = tf.tile(
            tf.expand_dims(r, axis=1), [1, seq_len, 1])

        h = tf.concat([r, x], axis=2)

        for layer, unit in enumerate(self.config.layers):
            h = tf.layers.dense(x, units=unit, activation=self.config.act,
                                kernel_initializer=self.config.kernel_init,
                                name=f'fc_"{layer}"')

        return get_z(h, latent_shape=2)

        # mu, log_sigma = tf.split(h, 2, axis=-1)
        #
        # # Bound the variance
        # sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
        # return mu, sigma


def map_xr(x, y, units, output_shape, h_dim=3):
    seq_len = x.get_shape()[1]
    pred_len = y.get_shape()[1]
    s = seq_len // pred_len
    if s > 1:
        y = tf.tile(y, (1, s, 1))

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

    seq_len = x_target.get_shape()[1]
    pred_len = z.get_shape()[1]
    z_tile = seq_len // pred_len
    z = tf.expand_dims(z, axis=1)
    z = tf.tile(z, (1, batch_size, z_tile, 1))
    x_target = tf.expand_dims(x_target, axis=0)
    x_target = tf.tile(x_target, (samples, 1, 1, 1))

    h = tf.concat([x_target, z], axis=3)

    # change decoder here
    h = tf.layers.dense(h, units, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, name='d_0')
    h = tf.reduce_mean(h, axis=0)  # average over samples
    return h


def get_z(h, latent_shape, dist="normal"):
    h = tf.layers.dense(h, latent_shape * 2, name='mu', activation=None)
    mu, log_sigma = tf.split(h, 2, axis=-1)
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
    return tfd.MultivariateNormalDiag(mu, sigma)

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
    # if not time_first:
    #     output = tf.transpose(output, (1, 0, 2))
    #     states = tf.transpose(states, (1, 0, 2))

    output.set_shape((None, seq_len, output.get_shape()[-1]))

    return output, state, states


def decode(h, last_y, cell, decoder_state=None, seq_len=24, time_first=False, attn=None):
    def cond_stop(time, prev_output, prev_state, output, states):
        return time < seq_len

    def loop_fn(time, prev_output, prev_state, output, states):
        y_tilde = attn.apply(tf.concat([prev_output, h[:, time, :]], axis=1), prev_state)
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
    # if not time_first:
    #     output = tf.transpose(output, (1, 0, 2))
    #     states = tf.transpose(states, (1, 0, 2))

    output.set_shape((None, seq_len, output.get_shape()[-1]))
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


def mask_tensor(x, s):
    not_x = tf.boolean_mask(x, tf.logical_not(s))
    x = tf.boolean_mask(x, s)
    return x, not_x


def mask_output(h, seq_len, pred_len):
    assert h.get_shape().ndims == 3
    if seq_len != pred_len:
        h = h[:, -pred_len:, :]
    return h
