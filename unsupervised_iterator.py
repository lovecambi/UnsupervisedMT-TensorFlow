# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import multiprocessing
import tensorflow as tf
import numpy as np

from tensorflow.contrib.data.python.ops import threadpool

_SHUFFLE_BUFFER = 10000

_DEFAULT_WINDOW_SIZE = 3
_DEFAULT_DEL_PROB = 0.1
_DEFAULT_MASK_PROB = 0.2
_DEFAULT_MASK_TOKEN = b"<mask>"

# Suppose using bpe tokenization
# e.g.,  I ha@@ ve an app@@ le ., 
# if using SPM, 
# please reimplement add_word_shuffle/delete/mask
_DEFAULT_BPE_SEP = b"@@"

UNK_ID = 2  # not protect member
_EOS_ID = 1
_PAD_ID = 0  # equal to tf default padding value


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer",
                            "lang1_iterator",
                            "lang2_iterator"))):
    pass


def add_word_shuffle(tokens, window=0):
    """This is word level shuffle, NOT token level.
    Args:
        tokens: A 1-D string ``np.ndarray`` of tokens (after tokenization, not necessary to be words).
        window: ``int``, permutation window size, new position of each word
            can be at most ``window`` shifted from the original position.
    Return:
        A new 1-D string ``np.ndarray`` after shuffling.
    """
    length = len(tokens)
    if window > 0 and length > 0:
        bpe_end = np.array([_DEFAULT_BPE_SEP not in token for token in tokens])
        word_idx = bpe_end[::-1].cumsum()[::-1]
        word_idx = word_idx.max() - word_idx

        offset = np.random.uniform(0, window, size=length)
        new_pos = np.argsort(word_idx + offset[word_idx] + 1e-6 * np.arange(length))
        tokens = np.take(tokens, new_pos)
    return tokens


def add_word_delete(tokens, del_prob=0.):
    """This is word level deletion, NOT token level.
    Args:
        tokens: A 1-D string ``np.ndarray`` of tokens (after tokenization, not necessary to be words).
        del_prob: ``float``, probability to delete a word.
    Return:
        A new 1-D string ``np.ndarray`` after shuffling.
    """
    length = len(tokens)
    if del_prob > 0. and length > 0:
        bpe_end = np.array([_DEFAULT_BPE_SEP not in token for token in tokens])
        word_idx = bpe_end[::-1].cumsum()[::-1]
        word_idx = word_idx.max() - word_idx

        keep = np.random.rand(length) > del_prob
        new_keep = keep[word_idx]
        if np.count_nonzero(new_keep) == 0:
            ind = np.random.randint(0, word_idx[-1] + 1)
            keep[ind] = True
            new_keep = keep[word_idx]
        tokens = np.take(tokens, new_keep.nonzero())[0]
    return tokens


def add_word_mask(tokens, mask_prob=0., mask_token=b"<mask>"):
    """This is word level deletion, NOT token level.
    Args:
        tokens: A 1-D string ``np.ndarray`` of tokens (after tokenization, not necessary to be words).
        mask_prob: ``float``, probability to delete a word.
        mask_token: A string for special token mask.
    Return:
        A new 1-D string ``np.ndarray`` after shuffling.
    """
    length = len(tokens)
    if mask_prob > 0. and length > 0:
        bpe_end = np.array([_DEFAULT_BPE_SEP not in token for token in tokens])
        word_idx = bpe_end[::-1].cumsum()[::-1]
        word_idx = word_idx.max() - word_idx

        mask = np.random.rand(length) <= mask_prob
        new_mask = mask[word_idx]
        tokens = np.where(new_mask, mask_token, tokens)
    return tokens


def add_sentence_noise(tokens):
    tokens = add_word_shuffle(tokens, _DEFAULT_WINDOW_SIZE)
    tokens = add_word_delete(tokens, _DEFAULT_DEL_PROB)
    tokens = add_word_mask(tokens, _DEFAULT_MASK_PROB, _DEFAULT_MASK_TOKEN)
    return tokens


def parse_sentence(sentence, vocab_table, is_training, dtype=tf.int64):
    """
    Args:
        sentence:
        vocab_table:
        is_training:
        dtype:
    Return:
        token ids
    """
    tokens = tf.string_split([sentence]).values
    _tokens_shape = tokens.get_shape()
    noise_tokens = tf.py_func(add_sentence_noise, [tokens], tf.string)
    noise_tokens.set_shape(_tokens_shape)

    token_ids = vocab_table.lookup(tokens)
    token_ids = tf.concat([token_ids, [_EOS_ID]], 0)

    noise_token_ids = vocab_table.lookup(noise_tokens)
    noise_token_ids = tf.concat([noise_token_ids, [_EOS_ID]], 0)
    return tf.cast(token_ids, dtype), tf.cast(noise_token_ids, dtype)


def get_padded_shapes(dataset):
    """Returns the padded shapes for ``tf.data.Dataset.padded_batch``.
    Args:
        dataset: The dataset that will be batched with padding.
    Returns:
        The same structure as ``dataset.output_shapes`` containing the padded
        shapes.
    """
    return tf.contrib.framework.nest.map_structure(
        lambda shape: shape.as_list(),
        dataset.output_shapes)


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           vocab_table,
                           num_repeat=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           num_parallel_batches=1,
                           bucket_width=1):
    """Given a Dataset with raw records, return an iterator over the records.
    Args:
        dataset: A Dataset representing raw records
        is_training: A boolean denoting whether the input is for training.
        batch_size: The number of samples per batch.
        shuffle_buffer: The buffer size to use when shuffling records. A larger
          value results in better randomness, but smaller values reduce startup
          time and use less memory.
        parse_record_fn: A function that takes a raw record and returns the
          corresponding (image, caption) pair.
        vocab_table: vocab for parsing caption.
        num_repeat: The number to repeat the dataset.
        dtype: Data type to use for images/features.
        datasets_num_private_threads: Number of threads for a private
          threadpool created for all datasets computation.
        num_parallel_batches: Number of parallel batches for tf.data.
        bucket_width:
    Returns:
        Dataset of (image, caption) pairs ready for iteration.
    """
    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if False: # is_training:
        # the shuffle is turned off, we will globally shulffle data with subprocess
        # before each epoch via cmd ``shuf``. 
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_repeat)

    # Parses the raw records into images and captions.
    dataset = dataset.map(
        lambda value: parse_record_fn(value, vocab_table, is_training, dtype),
        num_parallel_calls=batch_size * num_parallel_batches)

    if is_training:
        dataset = dataset.filter(
            lambda tokens, noise_tokens: tf.logical_and(
                tf.size(tokens) <= 175, tf.size(tokens) > 0))

    padded_shapes = get_padded_shapes(dataset)

    if is_training:
        def _key_func(token_ids, noise_token_ids):
            bucket_id = tf.constant(0, dtype=tf.int32)
            bucket_id = tf.maximum(bucket_id, tf.size(token_ids) // bucket_width)
            return tf.to_int64(bucket_id)

        def _reduce_func(unused_key, windowed_data):
            return windowed_data.padded_batch(
                batch_size,
                padded_shapes=padded_shapes)

        def _window_size_func(key):
            key += 1  # For bucket_width == 1, key 0 is unassigned.
            size = batch_size // (key * bucket_width)
            return tf.to_int64(size)

        if batch_size >= 1024:  # too large, it is very likely token based batch size.
            dataset = dataset.apply(
                tf.data.experimental.group_by_window(
                    key_func=_key_func, reduce_func=_reduce_func, window_size_func=_window_size_func))
        else:
            dataset = dataset.apply(
                tf.data.experimental.group_by_window(
                    key_func=_key_func, reduce_func=_reduce_func, window_size=batch_size))

    else:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            drop_remainder=False)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
        tf.logging.info('datasets_num_private_threads: %s',
                        datasets_num_private_threads)
        dataset = threadpool.override_threadpool(
            dataset,
            threadpool.PrivateThreadPool(
                datasets_num_private_threads,
                display_name='input_pipeline_thread_pool'))

    return dataset


def get_num_gpus():
    from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
    local_device_protos = device_lib.list_local_devices()
    return sum([1 for d in local_device_protos if d.device_type == "GPU"])


def get_datasets_num_private_threads():
    cpu_count = multiprocessing.cpu_count()  # Logical CPU cores
    gpu_count = get_num_gpus()
    per_gpu_thread_count = 1
    total_gpu_thread_count = per_gpu_thread_count * gpu_count
    num_monitoring_threads = 2 * gpu_count
    return cpu_count - total_gpu_thread_count - num_monitoring_threads


def get_iterator_txt(is_training,
                     lang1_data_file,
                     lang2_data_file,
                     lang1_vocab_file,
                     lang2_vocab_file,
                     batch_size,
                     bucket_width,
                     lang1_num_repeat=1,
                     lang2_num_repeat=1):
    # We prefer not to shuffle the data in the iterator, since the
    # dataset.shuffle is locally shuffled within buffer size.
    # Thus, we use the cmd process to shuffle before each epoch.

    datasets_num_private_threads = get_datasets_num_private_threads()

    lang1_dataset = process_record_dataset(
        dataset=tf.data.TextLineDataset(lang1_data_file),
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_sentence,
        vocab_table=tf.contrib.lookup.index_table_from_file(lang1_vocab_file, default_value=UNK_ID),
        num_repeat=lang1_num_repeat,
        dtype=tf.int64,
        datasets_num_private_threads=datasets_num_private_threads,
        num_parallel_batches=1,
        bucket_width=bucket_width)

    lang2_dataset = process_record_dataset(
        dataset=tf.data.TextLineDataset(lang2_data_file),
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=_SHUFFLE_BUFFER,
        parse_record_fn=parse_sentence,
        vocab_table=tf.contrib.lookup.index_table_from_file(lang2_vocab_file, default_value=UNK_ID),
        num_repeat=lang2_num_repeat,
        dtype=tf.int64,
        datasets_num_private_threads=datasets_num_private_threads,
        num_parallel_batches=1,
        bucket_width=bucket_width)

    lang1_iterator = lang1_dataset.make_initializable_iterator()
    lang2_iterator = lang2_dataset.make_initializable_iterator()

    return BatchedInput(
        initializer=(lang1_iterator.initializer, lang2_iterator.initializer),
        lang1_iterator=lang1_iterator,
        lang2_iterator=lang2_iterator)


