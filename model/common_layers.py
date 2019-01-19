# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.python.framework import function

_NEG_INF = -1e9


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """Identity operation whose gradient is converted to a `Tensor`.
      Currently, the gradient to `tf.concat` is particularly expensive to
      compute if dy is an `IndexedSlices` (a lack of GPU implementation
      forces the gradient operation onto CPU).  This situation occurs when
      the output of the `tf.concat` is eventually passed to `tf.gather`.
      It is sometimes faster to convert the gradient to a `Tensor`, so as
      to get the cheaper gradient for `tf.concat`.  To do this, replace
      `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.
      Args:
        x: A `Tensor`.
      Returns:
        The input `Tensor`.
    """
    return x


def transformer_prepare_encoder(x, embeddings, name):
    with tf.name_scope(name or "prepare_encoder"):
        x_emb = get_embedding(x, embeddings)
        x_pad = get_padding(x)
        att_bias = attention_bias_ignore_padding(x_pad)
        x_emb = add_timing_signal_1d(x_emb)
        return x_emb, att_bias, x_pad


def transformer_prepare_decoder(x, embeddings, name):
    with tf.name_scope(name or "prepare_decoder"):
        x_emb = get_embedding(shift_right_2d(x), embeddings)
        att_bias = get_decoder_self_attention_bias(
            shape_list(x_emb)[1])
        x_emb = add_timing_signal_1d(x_emb)
        return x_emb, att_bias


def get_embedding(x, embedding):
    """
    Args:
        x: int tensor
        embedding:
    Returns:
        index embedding of x
    """
    with tf.name_scope("embedding"):
        # embedding = convert_gradient_to_tensor(embedding)
        mask = tf.to_float(tf.not_equal(x, 0))
        x_emb = tf.gather(embedding, x)
        x_emb *= tf.expand_dims(mask, -1)
        return x_emb


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.
      Args:
        x: int tensor with any shape
        padding_value: int value that
      Returns:
        float tensor with same shape as x containing values 0 or 1.
          0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.
      Each channel of the input Tensor is incremented by a sinusoid of a different
      frequency and phase.
      This allows attention to learn to use absolute and relative positions.
      Timing signals should be added to some precursors of both the query and the
      memory inputs to attention.
      The use of relative position is possible because sin(x+y) and cos(x+y) can be
      expressed in terms of y, sin(x) and cos(x).
      In particular, we use a geometric sequence of timescales starting with
      min_timescale and ending with max_timescale.  The number of different
      timescales is equal to channels / 2. For each timescale, we
      generate the two sinusoidal signals sin(timestep/timescale) and
      cos(timestep/timescale).  All of these sinusoids are concatenated in
      the channels dimension.
      Args:
        length: scalar, length of timing signal sequence.
        channels: scalar, size of timing embeddings to create. The number of
            different timescales is equal to channels / 2.
        min_timescale: a float
        max_timescale: a float
        start_index: index of first position
      Returns:
        a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        tf.maximum(tf.to_float(num_timescales) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
      Each channel of the input Tensor is incremented by a sinusoid of a different
      frequency and phase.
      This allows attention to learn to use absolute and relative positions.
      Timing signals should be added to some precursors of both the query and the
      memory inputs to attention.
      The use of relative position is possible because sin(x+y) and cos(x+y) can be
      experessed in terms of y, sin(x) and cos(x).
      In particular, we use a geometric sequence of timescales starting with
      min_timescale and ending with max_timescale.  The number of different
      timescales is equal to channels / 2. For each timescale, we
      generate the two sinusoidal signals sin(timestep/timescale) and
      cos(timestep/timescale).  All of these sinusoids are concatenated in
      the channels dimension.
      Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
        start_index: index of first position
      Returns:
        a Tensor the same shape as x.
    """
    with tf.name_scope("add_pos_encoding"):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale,
                                      start_index)
        return x + signal


def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.
      Args:
        memory_padding: a float `Tensor` with shape [batch, memory_length].
      Returns:
        a `Tensor` with shape [batch, 1, 1, memory_length].
    """
    with tf.name_scope("attention_bias"):
        ret = memory_padding * _NEG_INF
        return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregressive property.
      Creates a tensor that masks out locations that correspond to illegal
      connections, so prediction at position i cannot draw information from future
      positions.
      Args:
        length: int length of sequences in batch.
      Returns:
        float tensor of shape [1, 1, length, length]
    """
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def shift_right_3d(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    with tf.name_scope("shift_targets"):
        if pad_value is None:
            shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        else:
            shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
        return shifted_targets


def shift_right_2d(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    with tf.name_scope("shift_targets"):
        if pad_value is None:
            shifted_targets = tf.pad(x, [[0, 0], [1, 0]])[:, :-1]
        else:
            shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1]
        return shifted_targets


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def sample_with_temperature(logits, temperature):
    """Either argmax or random sampling.
      Args:
        logits: a Tensor.
        temperature: a float  0.0=argmax 1.0=random
      Returns:
        a Tensor with one fewer dimension than logits.
      """
    if temperature == 0.0:
        # TF argmax doesn't handle >5 dimensions, so we reshape here.
        logits_shape = shape_list(logits)
        argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=1)
        return tf.reshape(argmax, logits_shape[:-1])
    else:
        assert temperature > 0.0
        reshaped_logits = (
            tf.reshape(logits, [-1, shape_list(logits)[-1]]) / temperature)
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices,
                             shape_list(logits)[:logits.get_shape().ndims - 1])
        return choices


def log_prob_from_logits(logits, reduce_axis=-1):
    return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y


def padded_cross_entropy_loss(logits, labels, smoothing=0.0):
    """Calculate cross entropy loss while ignoring padding.
      Args:
        logits: Tensor of size [batch_size, length_logits, vocab_size]
        labels: Tensor of size [batch_size, length_labels]
        smoothing: Label smoothing constant, used to determine the on and off values
      Returns:
        Returns the cross entropy loss and weight tensors: float32 tensors with
          shape [batch_size, max(length_logits, length_labels)]
      """
    with tf.name_scope("loss", values=[logits, labels]):
        # logits, labels = _pad_tensors_to_same_length(logits, labels)

        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
            if smoothing > 0.:
                confidence = 1.0 - smoothing
                vocab_size = tf.shape(logits)[-1]
                low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
                soft_targets = tf.one_hot(
                    tf.cast(labels, tf.int32),
                    depth=vocab_size,
                    on_value=confidence,
                    off_value=low_confidence)
                xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=soft_targets)

                # Calculate the best (lowest) possible value of cross entropy, and
                # subtract from the cross entropy loss.
                normalizing_constant = -(
                    confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
                    low_confidence * tf.log(low_confidence + 1e-20))
                xentropy -= normalizing_constant

            else:
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)

        weights = tf.to_float(tf.not_equal(labels, 0))
        return xentropy * weights, weights


