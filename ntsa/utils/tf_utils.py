#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import logging

import gin.tf
import numpy as np
import tensorflow as tf

from sdtw.soft_dtw_fast import _jacobian_product_sq_euc, _soft_dtw, _soft_dtw_grad

logging.getLogger("tensorflow")


@gin.configurable
def set_seed(seed=0):
    tf.set_random_seed(seed)


def reset():
    with tf.get_default_graph() as g:
        tf.reset_default_graph()


def assert_shape(x, template):
    if x.get_shape().tolist() != template:
        raise ValueError("Tensors has different shape")


def mask_tensor(x, s):
    not_x = tf.boolean_mask(x, tf.logical_not(s))
    x = tf.boolean_mask(x, s)
    return x, not_x


def mask_output(h, seq_len, pred_len):
    assert h.get_shape().ndims == 3
    if seq_len != pred_len:
        h = h[:, -pred_len:, :]
    return h


@gin.configurable(module='tf.train')
def train_fn(lr=1e-2, opt=tf.train.RMSPropOptimizer, use_decay=False, global_step=None, decay_steps=1., decay_rate=0.):
    if use_decay:
        lr = tf.train.polynomial_decay(
            global_step=global_step,
            learning_rate=lr,
            end_learning_rate=1e-4,
            decay_steps=decay_steps,
            power=1 - decay_rate,
        )
    opt = opt(lr)
    return opt


# @gin.configurable
def get_z(h, latent_shape, name="latent", dist="normal"):
    h = tf.layers.dense(h, latent_shape * 2, name=name, activation=None, reuse=tf.AUTO_REUSE)
    mu, log_sigma = tf.split(h, 2, axis=-1)
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
    if dist == "multi":
        return tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)
    else:
        return tf.distributions.Normal(mu, sigma)


@gin.configurable
def conv1d(x, filters, kernel_size, strides=1, padding='causal', dilation_rate=1, act=None,
           init=None, scope="conv1d", use_bias=True):
    batch_size, seq_len, h = x.get_shape().as_list()
    # Taken from keras, there is a faster version from magenta
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # assert seq_len % dilation_rate == 0

        w = tf.get_variable('kernel', shape=(kernel_size, h, filters), dtype=tf.float32, initializer=init)

        if padding == 'causal':
            # causal (dilated) convolution:
            left_pad = dilation_rate * (kernel_size - 1)
            pattern = [[0, 0], [left_pad, 0], [0, 0]]
            x = tf.pad(x, pattern)
            padding = 'VALID'

        out = tf.nn.convolution(
            input=x,
            filter=w,
            dilation_rate=(dilation_rate,),
            strides=(strides,),
            padding=padding)
        if use_bias:
            b = tf.get_variable('bias', shape=(filters,), dtype=tf.float32, initializer=tf.initializers.zeros)
            out = tf.add(out, b)
        if act is not None:
            return act(out)
    return out


@gin.configurable(module='tf.train')
def fc(h, units, act=tf.nn.leaky_relu, keep_prob=None, init=tf.initializers.variance_scaling, scope="fc_block",
       is_training=None):
    h = tf.layers.dense(h, units=units, kernel_initializer=init, name=scope, reuse=tf.AUTO_REUSE)
    if is_training is not None:
        h = tf.layers.batch_normalization(h, training=is_training)
    if keep_prob is not None:
        h = tf.nn.dropout(h, keep_prob=keep_prob)
    if act is not None:
        h = act(h)
    return h


@gin.configurable
def fc_block(h, layers=(), keep_prob=1., output_shape=None):
    for layer, unit in enumerate(layers):
        h = fc(h, unit, keep_prob=keep_prob, scope=f'fc_{layer}')
    if output_shape is not None:
        h = tf.layers.dense(h, units=output_shape, activation=None, name=f'out')
    return h


@gin.configurable
def conv1d_block(h, layers=(), output_shape=None):
    for layer, (filters, kernel_size, stride) in enumerate(layers):
        h = conv1d(h, filters, kernel_size, strides=stride, dilation_rate=2 ** layer, scope=f'conv_{layer}')
    if output_shape is not None:
        h = tf.layers.dense(h, units=output_shape, activation=None, name=f'out')
    return h


@gin.configurable
def rnn_block(keep_prob=None, layers=(32,), cell_type="gru", use_residual=True):
    Cell = tf.nn.rnn_cell.LSTMCell if cell_type == "lstm" else tf.nn.rnn_cell.GRUCell
    cells = []
    for layer, units in enumerate(layers):
        cell = Cell(num_units=units,
                    name=f"{cell_type}_cell_{layer+1}",
                    kernel_initializer=tf.initializers.variance_scaling,
                    reuse=tf.AUTO_REUSE)
        if use_residual:
            cell = tf.nn.rnn_cell.ResidualWrapper(cell)
        if keep_prob is not None:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)

    if len(layers) > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    return cell


def sn_block(x, filters, kernel_size, dilation, scope="sn_block"):
    with tf.variable_scope(scope):
        residual = x
        h = conv1d(x, filters=filters, kernel_size=kernel_size, dilation_rate=dilation, scope="conv_1", use_bias=False)
        h = tf.nn.leaky_relu(h, alpha=0.1)
        skip_out = conv1d(h, filters=1, kernel_size=1, scope="skip", use_bias=False)
        network_in = conv1d(h, filters=1, kernel_size=1, scope="network_in", use_bias=False)

        residual += network_in

    return residual, skip_out


@gin.configurable
def batch_norm(h, training, decay_rate=0.997, eps=1e-5):
    return tf.layers.batch_normalization(h, momentum=decay_rate, epsilon=eps, training=training,
                                         center=True, fused=True)


@gin.configurable
def clip_grads(loss, params, clip=20.):
    grads = tf.gradients(ys=loss, xs=params)
    clipped_grads, norm = tf.clip_by_global_norm(grads, clip)
    gvs = [(g, v) for (g, v) in zip(clipped_grads, params)]
    return gvs, norm


def summary_op(t_dict):
    ops = []
    for k, t in t_dict.items():
        name = t.name.replace(':', '_')
        if t.get_shape().ndims < 1:
            op = tf.summary.scalar(name=name, tensor=t)
        else:
            op = tf.summary.histogram(name=name, values=t)
        ops.append(op)
    return tf.summary.merge(ops)


def reg_fc(tensors, reg):
    return tf.add_n([reg * tf.nn.l2_loss(t) for t in tensors if 'weight' in t.name])


def reg_conv(tensors, reg):
    return tf.add_n([reg * tf.nn.l2_loss(t) for t in tensors if 'kernel' in t.name])


def reg_rnn(tensors):
    return tf.add_n([(activation_loss(t), stability_loss(t)) for t in tensors])


@gin.configurable
def activation_loss(h, beta):
    if beta == 0.0:
        return 0.0
    else:
        return tf.nn.l2_loss(h) * beta


@gin.configurable
def stability_loss(h, beta):
    if beta == 0.0:
        return 0.0
    else:
        l2 = tf.sqrt(tf.reduce_sum(tf.square(h), axis=-1))
        return beta * tf.reduce_mean(tf.square(l2[1:] - l2[:-1]))


def historgram_loss(y, y_hat, k=100., sigma=1 / 2):
    raise NotImplementedError()
    ps = 0.
    w = 1 / k
    y = tf.squeeze(y, axis=2)
    # y_hat = tf.layers.flatten(y_hat)
    k = np.linspace(0., 1., k)
    s = (tf.erf((1. - y) / (tf.sqrt(2.) * sigma)) - tf.erf((0. - y) / (tf.sqrt(2.) * sigma)))
    for idx, j in enumerate(k):
        u = tf.erf(((j + w - y) / (tf.sqrt(2.) * sigma)))
        l = tf.erf(((j - y) / (tf.sqrt(2.) * sigma)))
        p = (u - l) / (2 * s + 1e-6)
        f_x = tf.log(y_hat[:, :, idx])
        ps += p * tf.where(tf.is_nan(f_x), tf.zeros_like(f_x), f_x)
    return tf.reduce_mean(-ps)


def quantile_loss(y, y_hat, k=4):
    k = np.linspace(0., 1., k)
    loss = 0.
    y = tf.squeeze(y, axis=2)
    for idx, q in enumerate(k):
        error = tf.subtract(y, y_hat[:, :, idx])
        loss += tf.reduce_mean(tf.maximum(q * error, (q - 1) / error), axis=-1)
    return tf.reduce_mean(loss)


@gin.configurable
def _sdtw_loss(D, gamma=0.01):
    #    x,y = inputs
    m, n = D.shape
    R = np.zeros((m + 2, n + 2), dtype=np.float64)
    _soft_dtw(D, R, gamma=gamma)
    return R


def _sdtw_jacobian(y_hat, y, E):
    G = np.zeros_like(y_hat)
    _jacobian_product_sq_euc(y_hat, y, E, G)
    return G


def _sdtw_grad(y_hat, y, D, R, gamma=0.01):
    E = _sdtw_loss_grad(D, R, gamma=gamma)
    return _sdtw_jacobian(y_hat, y, E)


def _sdtw_loss_grad(D, R, gamma):
    m, n = D.shape
    D = np.vstack((D, np.zeros(n)))
    D = np.hstack((D, np.zeros((m + 1, 1))))
    E = np.zeros((m + 2, n + 2))

    _soft_dtw_grad(D, R, E, gamma=gamma)
    return E[1:-1, 1:-1]


def batch_sdtw_loss(D, gamma=0.01):
    R = np.array(list(map(partial(_sdtw_loss, gamma=gamma), D)))
    return R


def batch_sdtw_grad(y_hat, y, D, R, gamma=0.01):
    grad = np.array(list(map(partial(_sdtw_grad, gamma=gamma), y_hat, y, D, R)))
    return grad


def batched_cosine_distance(y_hat, y):
    assert y_hat.get_shape().ndims == 3 and y.get_shape().ndims == 3

    D = tf.matmul(y, y_hat, transpose_b=True)
    A = l2_norm(y_hat, axis=2)
    B = l2_norm(y, axis=2)
    return D / (tf.matmul(A, B, transpose_b=True))


def l2_norm(x, axis=2):
    squared = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
    norm = tf.sqrt(tf.maximum(squared, 1e-6))
    return norm


def batched_euclidean_distance(y_hat, y, squared=True):
    assert y_hat.get_shape().ndims == 3 and y.get_shape().ndims == 3
    a = tf.square(tf.reduce_sum(y, axis=2))[:, :, None]
    b = tf.square(tf.reduce_sum(y_hat, axis=2))[:, None, :]
    D = tf.matmul(y, y_hat, transpose_b=True)
    d = a + b - 2 * D
    return tf.sqrt(d) if not squared else d


@tf.custom_gradient
@gin.configurable
def sdtw_loss(y_hat, y, gamma=0.01):
    y_hat = tf.cast(y_hat, tf.float64)
    y = tf.cast(y, tf.float64)
    D = batched_euclidean_distance(y, y_hat)
    R = tf.py_func(batch_sdtw_loss, inp=[D, gamma], Tout=tf.float64)

    m = D.get_shape()[1]
    loss = tf.reduce_mean(R[:, m, m])
    loss.set_shape(shape=())
    loss = tf.cast(loss, dtype=tf.float32)

    def grad(dy):
        _grad = tf.py_func(batch_sdtw_grad, inp=[y_hat, y, D, R, gamma], Tout=tf.float64)
        return tf.cast(_grad, dtype=tf.float32), tf.zeros_like(y, dtype=tf.float32)

    return loss, grad


def mae(y, y_hat):
    return tf.abs(y - y_hat, name="mae")


def mse(y, y_hat):
    return tf.square(y - y_hat, name="mse")


@gin.configurable
def smape(y, y_hat, eps=0.1):
    summ = tf.maximum(tf.abs(y) + tf.abs(y_hat) + eps, 0.5 + eps)
    smape = tf.div(2. * tf.abs(y_hat - y), summ, name="smape")
    return smape


def which_loss(loss_op):
    if loss_op == "mae":
        return mae
    elif loss_op == "mse":
        return mse
    elif loss_op == "smape":
        return smape
    elif loss_op == "sdtw":
        return sdtw_loss
    else:
        raise NotImplementedError("Loss function not implemented")


@gin.configurable
def sequence_loss(y_hat, y, weights, loss_fn, avg_time=True, avg_batch=True):
    loss = loss_fn(y_hat, y)
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

    loss = tf.divide(loss, total_size, name="seq_loss")
    tf.losses.add_loss(loss)
    return loss


def queue_np(x, y):
    d = np.arange(0, y.shape[1])
    x = np.concatenate([x, y], axis=1)
    x = np.delete(x, d, axis=1)
    return x


def selu(x):
    """
    SELU activation
    https://arxiv.org/abs/1706.02515
    :param x:
    :return:
    """
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


# TODO this is model specific for rnn, make me general
def ar_inference(feed_dict, fetches, steps, sess, other=None):
    raise NotImplementedError()
    x = [k for k in feed_dict.keys() if k.name == 'init_ph/x:0'][0]

    y_hat, h = sess.run(fetches, feed_dict=feed_dict)

    for step in range(0, steps, y_hat.shape[1]):
        # append k elements at last and remove the first k
        feed_dict[x] = queue_np(feed_dict[x], h)

        y, h = sess.run(fetches, feed_dict=feed_dict)
        y_hat = np.concatenate([y_hat, y], axis=1)
    # assert y_hat.shape[1] == self._config.test_len
    return y_hat


def acc(y, y_hat, name='accuracy'):
    return tf.cast(tf.equal(y, y_hat, name=name), dtype=tf.int32)


def false_p(y, y_hat, name='fp'):
    return tf.logical_and(
        tf.equal(y, False), tf.equal(y_hat, True), name=name)


def false_n(y, y_hat, name='fn'):
    return tf.logical_and(
        tf.equal(y, True), tf.equal(y_hat, False), name=name)


def true_n(y, y_hat, name='tn'):
    return tf.logical_and(
        tf.equal(y, False), tf.equal(y_hat, False), name=name)


def true_p(y, y_hat, name='tp'):
    return tf.logical_and(
        tf.equal(y, True), tf.equal(y_hat, True), name=name)


def precision(y, y_hat, name='precision'):
    tp = true_p(y, y_hat)
    fp = false_p(y, y_hat)
    return tf.where(
        tf.greater(tp + fp, 0), tf.div(tp, tp + fp), 0, name)


def recall(y, y_hat, name='recall'):
    tp = true_p(y, y_hat)
    fn = false_p(y, y_hat)
    return tf.where(
        tf.greater(tp + fn, 0),
        tf.div(tp, tp + fn), 0, name)


def clf_metrics(y, y_hat):
    clf_ops = {
        'accuracy': acc(y, y_hat),
        'precision': precision(y, y_hat),
        'recall': recall(y, y_hat),
        'true_negatives': true_n(y, y_hat),
        'true_positives': true_p(y, y_hat)
    }
    return clf_ops


def regr_metrics(y, y_hat):
    regr_ops = {
        'mse': mse(y, y_hat),
        'mae': mae(y, y_hat),
        'smape': smape(y, y_hat),
        'rmse': tf.sqrt(mse(y, y_hat))
    }
    return regr_ops


def gauss_kernel(x, D, gamma=1.):
    x = tf.expand_dims(x, axis=-1)
    if x.get_shape().ndims < 4:
        D = tf.reshape(D, (1, 1, -1))
    else:
        D = tf.reshape(D, (1, 1, 1, 1, -1))

    return tf.exp(- gamma * tf.square(x - D))


def gauss_kernel2D(x, Dx, Dy, gamma=1.):
    h_size = (x.get_shape()[-1].value) // 2

    x = tf.expand_dims(x, axis=-1)
    if x.get_shape().ndims < 4:
        Dx = tf.reshape(Dx, (1, 1, -1))
        Dy = tf.reshape(Dy, (1, 1, -1))
        x1, x2 = x[:, :h_size], x[:, h_size:]
    else:
        Dy = tf.reshape(Dy, (1, 1, 1, 1, -1))
        Dx = tf.reshape(Dx, (1, 1, 1, 1, -1))
        x1, x2 = x[:, :, :, :h_size], x[:, :, :, h_size:]
    gauss_kernel = tf.exp(-gamma * tf.square(x1 - Dx)) + tf.exp(- gamma * tf.square(x2 - Dy))
    return gauss_kernel


class RidgeInit(object):
    def __init__(self, K, d, reg=1e-1, dtype=tf.float32):
        self.K = K
        self.d = tf.reshape(d, (1, -1, 1))
        self.reg = reg
        self.dtype = dtype

    def __call__(self, shape=None, dtype=None, partition_info=None):
        out = tf.matrix_solve_ls(self.K + self.reg * tf.eye(self.K.get_shape()[-1].value), self.d)
        out = tf.tile(out, (1, 1, shape[1]))
        return tf.reshape(out, shape)

    def get_config(self):
        return {
            'reg': self.reg,
            'dtype': self.dtype.name
        }


class Kaf(object):
    def __init__(self, input_shape, dict_size=(-1., 1., 20), gamma=None):
        self.d = tf.linspace(*dict_size)
        if gamma is None:
            self.gamma = .5 / tf.square(2 * (self.d[-1] - self.d[0]))  # (d_stop - d_start)*2
        else:
            self.gamma = gamma
        self.alpha = tf.get_variable('alpha', shape=(1, input_shape, self.d.get_shape()[0]),
                                     initializer=RidgeInit(gauss_kernel(self.d, self.d, self.gamma), self.d))

    def __call__(self, x):

        K = gauss_kernel(x, self.d, self.gamma)
        return tf.reduce_sum(tf.multiply(K, self.alpha), axis=-1)


def kaf(linear, name, kernel='rbf', D=None, gamma=None):
    if D is None:
        D = tf.linspace(start=-2., stop=2., num=20)

    with tf.variable_scope('kaf', reuse=tf.AUTO_REUSE):
        if kernel == "rbf":
            K = gauss_kernel(linear, D, gamma=gamma)
            alpha = tf.get_variable(name, shape=(1, linear.get_shape()[-1], D.get_shape()[0]),
                                    initializer=tf.random_normal_initializer(stddev=0.1))
        elif kernel == 'rbf2d':
            Dx, Dy = tf.meshgrid(D, D)
            K = gauss_kernel2D(linear, Dx, Dy, gamma=gamma)

            alpha = tf.get_variable(name,
                                    shape=(1, linear.get_shape()[-1] // 2, D.get_shape()[0] * D.get_shape()[0]),
                                    initializer=tf.random_normal_initializer(stddev=0.1))
        else:
            raise NotImplementedError()
        act = tf.reduce_sum(tf.multiply(K, alpha), axis=-1)
        # act = tf.squeeze(act, axis=0)
    return act


def init_sess(var_list=None, path=None):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = None
    writer = None
    if var_list is not None:
        saver = tf.train.Saver(var_list=var_list, filename='model.ckpt')
    if path is not None:
        writer = tf.summary.FileWriter(logdir=path)
    return sess, saver, writer


def restore(saver, sess, path):
    try:
        ckpt = tf.train.latest_checkpoint(path)
        saver.restore(sess=sess, save_path=ckpt)
    except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e:
        raise e
