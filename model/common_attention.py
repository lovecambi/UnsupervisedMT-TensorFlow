# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, train):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.
        Args:
          x: A tensor with shape [batch_size, length, hidden_size]
        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y.
        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.
        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)


class FeedFowardNetwork(tf.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad):
        super(FeedFowardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(
            filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name="output_layer")

    def call(self, x, padding=None):
        """Return outputs of the feedforward network.
        Args:
          x: tensor with shape [batch_size, length, hidden_size]
          padding: (optional) If set, the padding values are temporarily removed
            from x (provided self.allow_pad is set). The padding values are placed
            back in the output tensor in the same locations.
            shape [batch_size, length]
        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """
        padding = None if not self.allow_pad else padding

        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope("remove_padding"):
                # Flatten padding to [batch_size*length]
                pad_mask = tf.reshape(padding, [-1])

                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions.
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)
        if self.train:
            output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
        output = self.output_dense_layer(output)

        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size]
                )
                output = tf.reshape(output, [batch_size, length, self.hidden_size])
        return output
