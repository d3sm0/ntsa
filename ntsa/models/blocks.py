#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gin.tf.external_configurables

from utils.tf_utils import fc_block, conv1d_block, rnn_block, get_z, tf, Kaf


class Block(object):
    def __init__(self, output_shape=None, scope="block"):
        self._built = False
        self._scope = scope
        self._output_shape = output_shape
        self._vars = []
        self._endpoints = {}  # all the outputs of the model

    def apply(self, x, *args, **kwargs):
        return

    def __call__(self, x, *args, **kwargs):
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            out = self.apply(x, *args, **kwargs)
            if not self._built:
                self._built = True
                self._vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            return out

    @property
    def scope(self):
        return self._scope

    @property
    def vars(self):
        return self._vars


class FullyConnected(Block):
    def __init__(self, output_shape, scope="fully_connected"):
        super(FullyConnected, self).__init__(output_shape=output_shape, scope=scope)

    def apply(self, x, keep_prob):
        h = fc_block(x, output_shape=self._output_shape, keep_prob=keep_prob)
        return h


class Conv1D(Block):
    def __init__(self, output_shape, scope="conv1d"):
        super(Conv1D, self).__init__(output_shape, scope)

    def apply(self, x, *args, **kwargs):
        h = conv1d_block(x, output_shape=self._output_shape)
        return h


@gin.configurable
class Recurrent(Block):
    def __init__(self, output_shape, attn=False, scope="recurrent"):
        # unit Recurrent, encode(h)
        super(Recurrent, self).__init__(output_shape, scope)
        self._attn = attn
        self._cell = None

    def build(self):
        self._built = True

    def apply(self, x, keep_prob=None):
        cell = rnn_block(keep_prob=keep_prob)

        if self._attn is not None:
            attn = InputAttention(output_shape=self._output_shape, memory=x)
        else:
            attn = Attention(output_shape=self._output_shape)

        init_state = cell.zero_state(batch_size=tf.shape(x)[0], dtype=tf.float32)

        output, state, states = encode(x, cell, encoder_state=init_state, time_first=False, attn=attn,
                                       seq_len=x.get_shape().as_list()[1])
        return output, state, states


@gin.configurable
class RecurrentEncoder(Recurrent):
    def __init__(self, output_shape, attn=False, scope="encoder"):
        # unit Recurrent, encode(h)
        super(RecurrentEncoder, self).__init__(output_shape, attn, scope=scope)


@gin.configurable
class RecurrentDecoder(Recurrent):
    def __init__(self, output_shape, seq_len, attn=False, scope="decoder"):
        super(RecurrentDecoder, self).__init__(output_shape, attn, scope=scope)
        self._seq_len = seq_len

    def apply(self,
              x,
              keep_prob=1.,
              encoder_states=None,
              init_state=None,
              y_features=None):
        cell = rnn_block(keep_prob=keep_prob)

        if self._attn == "time":
            attn = TimeAttention(input_shape=y_features.get_shape().as_list()[1],
                                 output_shape=self._output_shape,
                                 memory=encoder_states,
                                 y_features=y_features)
        elif self._attn == "aligned":
            attn = AlignedAttention(input_shape=y_features.get_shape().as_list()[1],
                                    output_shape=self._output_shape,
                                    memory=encoder_states,
                                    y_features=y_features)
        else:
            attn = DecoderAttention(output_shape=self._output_shape, y_features=y_features)

        if init_state is None:
            init_state = cell.zero_state(batch_size=tf.shape(x)[0], dtype=tf.float32)
        output, state, states = decode(x, cell, decoder_state=init_state, attn=attn,
                                       seq_len=self._seq_len)
        return output, state, states


class RGenerator(Recurrent):
    def __init__(self, output_shape, scope="generator"):
        super(RGenerator, self).__init__(output_shape, scope=scope)


class RDiscriminator(Recurrent):
    def __init__(self, output_shape, scope="discriminator"):
        super(RDiscriminator, self).__init__(output_shape, scope=scope)


class CGenerator(Conv1D):
    def __init__(self, output_shape, scope="generator"):
        super(CGenerator, self).__init__(output_shape=output_shape, scope=scope)


class CDiscriminator(Conv1D):
    def __init__(self, output_shape, scope="discriminator"):
        super(CDiscriminator, self).__init__(output_shape=output_shape, scope=scope)


class NeuralEncoder(Block):
    def __init__(self, output_shape=None, scope="encoder"):
        super(NeuralEncoder, self).__init__(output_shape, scope)

    def apply(self, x, y, keep_prob=1.):
        h = tf.concat([x, y], axis=2)

        h = fc_block(h, keep_prob=keep_prob)

        h = tf.reduce_mean(h, axis=1)
        return h


class NeuralDecoder(Block):
    def __init__(self, scope="decoder"):
        super(NeuralDecoder, self).__init__(scope)

    def apply(self, r, x, keep_prob=1.):
        r = tf.tile(tf.expand_dims(r, axis=1), [1, tf.shape(x)[1], 1])

        h = tf.concat([r, x], axis=2)
        h = fc_block(h, keep_prob=keep_prob)

        d = get_z(h, latent_shape=1, dist="multi")
        return d


class StochasticNeuralDecoder(NeuralDecoder):
    def __init__(self, scope="decoder"):
        super(NeuralDecoder, self).__init__(scope)

    def apply(self, z, x, keep_prob=1.):
        # [n_draws, batch_size, seq_len, dim_z]

        samples = z.get_shape()[0]

        seq_len = x.get_shape()[1]

        z = tf.expand_dims(z, axis=2)
        z = tf.tile(z, (1, 1, seq_len, 1))

        x_target = tf.expand_dims(x, axis=0)
        x_target = tf.tile(x_target, (samples, 1, 1, 1))

        h = tf.concat([x_target, z], axis=3)

        h = fc_block(h, keep_prob=keep_prob)

        h = tf.reduce_mean(h, axis=0)  # average over samples
        z = get_z(h, latent_shape=1, dist="multi")
        return z


class Attention(object):
    def __init__(self, output_shape=None, memory=None, use_bias=True, name='attention'):
        self._memory = memory
        self._name = name
        self._output_shape = output_shape
        self._project = self.projection_layer(output_shape, use_bias=use_bias) if output_shape is not None else lambda \
                x: x

    def apply(self, query, state, t):
        return query, state

    def project(self, state):
        return self._project(state)

    @staticmethod
    def projection_layer(output_shape, use_bias=True):
        with tf.variable_scope("proj_layer"):
            h = tf.layers.Dense(units=output_shape, use_bias=use_bias, name='fc',
                                kernel_initializer=tf.variance_scaling_initializer)
            return h


class DecoderAttention(Attention):
    def __init__(self, output_shape=None, y_features=None, name='decoder_attention'):
        super(DecoderAttention, self).__init__(output_shape, memory=y_features, name=name)

    def apply(self, query, state, t):
        return self._memory[:, t, :], state


@gin.configurable
class InputAttention(Attention):

    def __init__(self,
                 output_shape,
                 memory,
                 name='input_attention',
                 use_bias=False,
                 use_kaf=0):
        super(InputAttention, self).__init__(
            output_shape=output_shape,
            memory=tf.transpose(memory, (0, 2, 1)),
            name=name,
        )
        if use_kaf:
            k = Kaf(input_shape=self._memory.get_shape()[1])
            self._act = lambda x: tf.nn.softmax(k(x) + x)
        else:
            self._act = tf.nn.softmax
        self._memory_layer = tf.layers.Dense(units=1, use_bias=use_bias, name='memory', activation=None)
        self._input_shape = memory.get_shape()[-1]

    def apply(self, query, state, t):
        # batch_size, input_shape, cell_size + seq_len
        x = tf.concat([tf.tile(tf.expand_dims(state, axis=1), (1, self._input_shape, 1)), self._memory], axis=2)
        x = self._memory_layer(x)
        x = tf.squeeze(x, axis=2)
        alpha = self._act(x)
        x_tilde = tf.multiply(alpha, query)
        return x_tilde, state

@gin.configurable
class TimeAttention(Attention):

    def __init__(self, input_shape, output_shape, memory, y_features, name='time_attention', use_bias=False, use_kaf=False):
        super(TimeAttention, self).__init__(output_shape=output_shape,
                                            memory=memory,
                                            use_bias=True,
                                            name=name)
        # input_shape == encoder hidden size 128
        self._input_shape = memory.get_shape().as_list()[1]
        self._y_features = y_features
        if use_kaf:
            k = Kaf(input_shape=self._output_shape)
            self._act = lambda x: tf.nn.softmax(k(x) + x)
        else:
            self._act = tf.nn.softmax
        self._memory_layer = self.memory_layer(memory.get_shape().as_list()[-1], use_bias=use_bias, act=self._act)
        self._query_layer = tf.layers.Dense(units=1, use_bias=use_bias, name='query_layer',
                                            kernel_initializer=tf.variance_scaling_initializer)

    # TODO replace here with an estimate of y[t-1]
    def apply(self, query, state, time):
        # batch_size, seq_len, encoder + decoder_state
        context = self.context(state)
        y_tilde = self._query_layer(tf.concat([self._y_features[:, time, :], context], axis=1))
        return y_tilde, state

    def context(self, state):
        # none, 25, 64
        x = tf.concat([tf.tile(tf.expand_dims(state, axis=1), (1, self._input_shape, 1)), self._memory], axis=2)
        beta = self._memory_layer(x)  # None, 25
        context = tf.matmul(tf.expand_dims(tf.squeeze(beta, axis=-1), axis=1), self._memory)[:, 0, :]  # None, cell_size
        return context

    @staticmethod
    def memory_layer(input_shape, use_bias=False, act=tf.nn.softmax):
        with tf.variable_scope('beta', reuse=tf.AUTO_REUSE, initializer=tf.initializers.variance_scaling):
            fc_0 = tf.layers.Dense(units=input_shape, use_bias=use_bias, activation=tf.nn.tanh, name='fc_0')
            fc_1 = tf.layers.Dense(units=1, use_bias=use_bias, activation=act, name='fc_1')
        return lambda x: fc_1(fc_0(x))


class AlignedAttention(Attention):

    def __init__(self, input_shape, output_shape, memory, y_features=None, name='time_attn', use_bias=False):
        super(AlignedAttention, self).__init__(output_shape=output_shape,
                                               memory=memory,
                                               use_bias=True,
                                               name=name)
        self._input_shape = input_shape

        self._memory_layer = tf.layers.Dense(units=1, name="memory_layer",
                                             kernel_initializer=tf.initializers.variance_scaling)

        self._query_layer = tf.layers.Dense(units=memory.get_shape()[-1],
                                            use_bias=use_bias, name='query_layer',
                                            kernel_initializer=tf.variance_scaling_initializer, activation=tf.nn.tanh)
        self._memory_length = None
        self._y_features = y_features

    def apply(self, query, state, t):
        x = tf.concat([tf.tile(tf.expand_dims(state, axis=1), (1, self._input_shape, 1)), self._memory], axis=2)
        aligned = self._memory_layer(x)
        aligned = tf.squeeze(aligned, axis=2)

        w = tf.expand_dims(tf.nn.softmax(aligned), axis=-1)
        c = tf.reduce_sum(w * self._memory, 1)
        s_tilde = self._query_layer(tf.concat([state, c], axis=1))

        query = tf.concat([self._y_features[:, t, :], query], axis=-1)
        return query, s_tilde


#
def encode(h, cell, encoder_state=None, seq_len=24, time_first=False, attn=None):
    def cond_stop(time, prev_state, output, states):
        return time < seq_len

    def loop_fn(time, prev_state, output, states):
        x_tilde, prev_state = attn.apply(h[:, time, :], prev_state, time)  # input_attention(time, prev_state)
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

    output.set_shape((None, seq_len, output.get_shape()[-1]))
    states.set_shape((None, seq_len, output.get_shape()[-1]))
    #
    return output, state, states


def decode(last_y, cell, decoder_state=None, seq_len=24, time_first=False, attn=None):
    def cond_stop(time, prev_output, prev_state, output, states):
        return time < seq_len

    def loop_fn(time, prev_output, prev_state, output, states):
        # y_tilde = attn.apply(tf.concat([prev_output, h[:, time, :]], axis=1), prev_state)
        y_tilde, s_tilde = attn.apply(prev_output, prev_state, time)
        out, state = cell(y_tilde, s_tilde)
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

    output.set_shape((None, seq_len, output.get_shape()[-1]))
    states.set_shape((None, seq_len, states.get_shape()[-1]))
    return output, state, states
