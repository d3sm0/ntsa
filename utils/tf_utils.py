# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


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


def mae(y, y_hat):
    return tf.abs(y - y_hat, name="mae")


def mse(y, y_hat):
    return tf.square(y - y_hat, name="mse")


def smape(y, y_hat, eps=0.1):
    summ = tf.maximum(tf.abs(y) + tf.abs(y_hat) + eps, 0.5 + eps)
    smape = tf.div(2. * tf.abs(y_hat - y), summ, name="smape")
    return smape


def set_seed(seed=0):
    np.random.seed(seed)
    tf.set_random_seed(seed)


def fc_block(h, units, act=tf.nn.relu, keep_prob=None, init=tf.initializers.variance_scaling, scope="fc_block",
             is_training=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        h = tf.layers.dense(h, units=units, kernel_initializer=init, name="fc")
        if is_training is not None:
            h = tf.layers.batch_normalization(h, training=is_training)
        if keep_prob is not None:
            h = tf.nn.dropout(h, keep_prob=keep_prob)
        if act is not None:
            h = act(h)
    return h


def batch_norm(h, training, norm_decay=_BATCH_NORM_DECAY, batch_norm_eps=_BATCH_NORM_EPSILON):
    return tf.layers.batch_normalization(h, momentum=norm_decay, epsilon=batch_norm_eps, training=training,
                                         center=True, fused=True)


def sn_block(x, filters, kernel_size, dilation, scope="sn_block"):
    with tf.variable_scope(scope):
        residual = x
        h = conv1d(x,
                   filters=filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation,
                   use_bias=False,
                   scope="conv_1")
        h = tf.nn.leaky_relu(h, alpha=0.1)
        skip_out = conv1d(h,
                          filters=1,
                          kernel_size=1,
                          use_bias=False,
                          scope="skip")
        network_in = conv1d(h,
                            filters=1,
                            kernel_size=1,
                            use_bias=False,
                            scope="network_in")

        residual += network_in

    return residual, skip_out


def conv1d(x, filters, kernel_size, strides=1, padding='causal', dilation_rate=1, activation=lambda x: x,
           init=tf.initializers.variance_scaling, scope="conv1d", use_bias=True):
    batch_size, seq_len, h = x.get_shape().as_list()
    # Taken from keras, there is a faster version from magenta
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # assert seq_len % dilation_rate == 0

        w = tf.get_variable('kernel', shape=(kernel_size, h, filters), dtype=tf.float32, initializer=init)
        if use_bias:
            b = tf.get_variable('bias', shape=(filters,), dtype=tf.float32, initializer=tf.initializers.zeros)

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
            out = tf.add(out, b)

    return activation(out)


def conv1d_v2(x, filters, kernel_size, strides=1, padding='causal', dilation_rate=1, activation=lambda x: x,
              init=tf.initializers.variance_scaling, use_bias=True, keep_prob=None, scope="conv1d_v2"):
    raise NotImplementedError()
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        batch_size, seq_len, features = x.get_shape()

        if padding == "causal":
            shift = (kernel_size // 2) + (dilation_rate - 1) // 2
            pad = tf.zeros([batch_size, shift, features])
            x = tf.concat([pad, x], axis=1)

        w = tf.get_variable("kernel",
                            initializer=init,
                            shape=[kernel_size, features, filters])
        z = tf.nn.convolution(x, w, padding="SAME", dilation_rate=dilation_rate)

        if use_bias:
            b = tf.get_variable("bias", initializer=tf.zeros_initializer, shape=(filters,))
            z = z + b

        if activation is not None:
            z = activation(z)
        if keep_prob is not None:
            z = tf.nn.dropout(z, keep_prob)
        if padding == "causal":
            z = z[:, :-shift, :]
        return z


def clip_grads(loss, params, clip=20.):
    grads = tf.gradients(ys=loss, xs=params)
    clipped_grads, norm = tf.clip_by_global_norm(grads, clip)
    gvs = [(g, v) for (g, v) in zip(clipped_grads, params)]
    return gvs, norm


def summary_op(t_list):
    ops = []
    for t in t_list:
        name = t.name.replace(':', '_')
        if t.get_shape().ndims < 1:
            op = tf.summary.scalar(name=name, tensor=t)
        else:
            op = tf.summary.histogram(name=name, values=t)
        ops.append(op)
    return tf.summary.merge(ops)


def reg_conv(vars, reg):
    return tf.add_n([reg * tf.nn.l2_loss(v) for v in vars if 'kernel' in v.name])


def which_loss(loss_type="clf"):
    if loss_type == "clf":
        return tf.nn.softmax_cross_entropy_with_logits_v2
    elif loss_type == "mse":
        return mse
    elif loss_type == "mae":
        return mae
    elif loss_type == "smape":
        return smape
    elif loss_type == "dtw":
        # TODO https://github.com/mblondel/soft-dtw
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def queue_np(x, y):
    d = np.arange(0, y.shape[1])
    x = np.concatenate([x, y], axis=1)
    x = np.delete(x, d, axis=1)
    return x


# TODO this is model specific for rnn, make me general
def ar_inference(feed_dict, fetches, steps, sess, other=None):
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
    return tf.equal(y, y_hat, name=name)


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
        'smape': smape(y, y_hat)
    }
    return regr_ops


def gauss_kernel(x, D, gamma=1.):
    x = tf.expand_dims(x, axis=-1)
    if x.get_shape().ndims < 4:
        D = tf.reshape(D, (1, 1, -1))
    else:
        D = tf.reshape(D, (1, 1, 1, 1, -1))

    gauss_kernel = tf.exp(- gamma * tf.square(x - D))
    return gauss_kernel


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


def kaf(linear, name, kernel='rbf', D=None, gamma=1., ):
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
