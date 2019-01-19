# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

import unsupervised_iterator

_SHUFFLE_BUFFER = 1000000

UNK_ID = unsupervised_iterator.UNK_ID


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer",
                            "iterator"))):
    pass


def get_iterator_txt(is_training,
                     lang1_data_file,
                     lang2_data_file,
                     lang1_vocab_file,
                     lang2_vocab_file,
                     batch_size,
                     bucket_width,
                     num_repeat):

    def get_num_gpus():
        from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
        local_device_protos = device_lib.list_local_devices()
        return sum([1 for d in local_device_protos if d.device_type == "GPU"])

    num_gpus = get_num_gpus()

    lang1_dataset = tf.data.TextLineDataset(lang1_data_file)
    lang2_dataset = tf.data.TextLineDataset(lang2_data_file)
    lang1_vocab_table = tf.contrib.lookup.index_table_from_file(lang1_vocab_file, default_value=UNK_ID)
    lang2_vocab_table = tf.contrib.lookup.index_table_from_file(lang2_vocab_file, default_value=UNK_ID)

    dataset = tf.data.Dataset.zip((lang1_dataset, lang2_dataset))

    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffles records before repeating to respect epoch boundaries.
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_repeat)

    dataset = dataset.map(
        lambda lang1, lang2: (
            tf.string_split([lang1]).values,
            tf.string_split([lang2]).values),
        num_parallel_calls=8).prefetch(1000000)

    dataset = dataset.map(
        lambda lang1, lang2: (
            lang1_vocab_table.lookup(lang1),
            lang2_vocab_table.lookup(lang2)),
        num_parallel_calls=8).prefetch(10000000)

    if is_training:
        dataset = dataset.filter(
            lambda lang1, lang2: tf.logical_and(tf.size(lang1) <= 100, tf.size(lang2) <= 100))

    if bucket_width > 1 and is_training:
        def key_func(lang1, lang2):
            bucket_id = tf.constant(0, dtype=tf.int32)
            bucket_id = tf.maximum(bucket_id, tf.size(lang1) // bucket_width)
            bucket_id = tf.maximum(bucket_id, tf.size(lang2) // bucket_width)
            return tf.to_int64(bucket_id)

        def reduce_func(unused_key, windowed_data):
            return windowed_data.padded_batch(batch_size, padded_shapes=([None], [None]))

        def window_size_func(key):
            key += 1  # For bucket_width == 1, key 0 is unassigned.
            size = batch_size // (key * bucket_width)
            return tf.to_int64(size)

        dataset = dataset.apply(
            tf.data.experimental.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size_func=window_size_func))
    else:
        dataset = dataset.padded_batch(batch_size * num_gpus, padded_shapes=([None], [None]))

    iterator = dataset.make_initializable_iterator()

    return BatchedInput(
        initializer=iterator.initializer,
        iterator=iterator)

