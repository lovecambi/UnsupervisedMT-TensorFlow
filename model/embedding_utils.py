# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from model import misc_utils


VOCAB_SIZE_THRESHOLD_CPU = 50001


def _get_embed_device(vocab_size):
    """Decide on which device to place an embed matrix given its vocab size."""
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    else:
        return "/gpu:0"


def _create_pretrained_emb_from_txt(vocab_file, embed_file, num_notpretrained_tokens=4, dtype=tf.float32):
    """Load pretrain embeding from embed_file, and return an embedding matrix.
      Args:
        embed_file: Path to a Glove formated embedding txt file.
        num_notpretrained_tokens: Make the first n tokens in the vocab file as not
          pretrained variables. Default is 4, which is "</s>, <s>, <unk>, <mask>".
    """
    vocab, vocab_size = misc_utils.load_vocab(vocab_file)
    # notpretrained_tokens = vocab[:num_notpretrained_tokens]
    # TODO: hparam to control
    notpretrained_tokens = vocab[:1] + vocab[2:num_notpretrained_tokens]  # id=1 of </s> has been pretrained.

    misc_utils.print_out("# Using pre-trained embedding: %s." % embed_file)
    misc_utils.print_out("  Analyzing not pre-trained tokens: ")

    emb_dict, emb_size = misc_utils.load_embed_txt(embed_file)
    assert len(emb_dict) == vocab_size - num_notpretrained_tokens + 1
    for token in notpretrained_tokens:
        misc_utils.print_out("    %s" % token)
        if token == notpretrained_tokens[0]:
            emb_dict[token] = [0.0] * emb_size
        elif token not in emb_dict:
            emb_dict[token] = emb_size ** -0.5 * np.random.randn(emb_size)

    emb_np = np.array(
        [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
    return emb_np


def _create_embed(embed_name, vocab_size, embed_size, dtype, emb_np=None):
    """Create a new embedding matrix or load from pretrained numpy matrix."""
    with tf.device(_get_embed_device(vocab_size)):
        if isinstance(emb_np, np.ndarray):
            assert emb_np.shape == (vocab_size, embed_size)
            embedding = tf.get_variable(
                embed_name, [vocab_size, embed_size], dtype,
                initializer=tf.constant_initializer(emb_np),
                trainable=True)
        else:
            embedding = tf.get_variable(
                embed_name, [vocab_size, embed_size], dtype)
    return embedding


def get_all_embeddings(params, dtype=tf.float32, scope=None):

    if params["lang1_partitions"] <= 1:
        lang1_partitioner = None
    else:
        lang1_partitioner = tf.fixed_size_partitioner(params["lang1_partitions"])

    if params["lang2_partitions"] <= 1:
        lang2_partitioner = None
    else:
        lang2_partitioner = tf.fixed_size_partitioner(params["lang2_partitions"])

    encoder_embeddings = {}
    decoder_embeddings = {}

    lang1_emb_np, lang2_emb_np = None, None
    if params["lang1_embed_file"] and params["lang2_embed_file"]:
        lang1_emb_np = _create_pretrained_emb_from_txt(params["lang1_vocab_file"], params["lang1_embed_file"])
        if params["lang1_embed_file"] == params["lang2_embed_file"]:
            lang2_emb_np = lang1_emb_np
        else:
            lang2_emb_np = _create_pretrained_emb_from_txt(params["lang2_vocab_file"], params["lang2_embed_file"])

    if params["share_decpro_emb"]:
        if params["share_lang_emb"]:
            assert params["share_output_emb"]
            share_bias = tf.get_variable('share_projection/bias',
                                         [params["lang1_vocab_size"], ],
                                         initializer=tf.zeros_initializer())
            pro_embs = {params["lang1"]: share_bias,
                        params["lang2"]: share_bias}
        else:
            pro_embs = {params["lang1"]: tf.get_variable('bias',
                                                         [params["lang1_vocab_size"], ],
                                                         initializer=tf.zeros_initializer()),
                        params["lang2"]: tf.get_variable('bias',
                                                         [params["lang2_vocab_size"], ],
                                                         initializer=tf.zeros_initializer())}
    else:
        if params["share_output_emb"]:
            assert params["share_lang_emb"]
            if params["pretrained_out"]:
                assert params["lang1_embed_file"] == params["lang2_embed_file"]
                misc_utils.print_out("# Using pre-trained embedding to initialize shared projection kernel.")
                share_proj_layer = tf.layers.Dense(params["lang1_vocab_size"],
                                                   use_bias=True,
                                                   kernel_initializer=tf.constant_initializer(lang1_emb_np.transpose()),
                                                   name="share_projection")
            else:
                share_proj_layer = tf.layers.Dense(params["lang1_vocab_size"],
                                                   use_bias=True,
                                                   name="share_projection")
            pro_embs = {params["lang1"]: share_proj_layer,
                        params["lang2"]: share_proj_layer}
        else:
            if params["pretrained_out"]:
                misc_utils.print_out("# Using pre-trained embedding to initialize two projection kernels.")
                pro_embs = {params["lang1"]: tf.layers.Dense(params["lang1_vocab_size"],
                                                             use_bias=True,
                                                             kernel_initializer=tf.constant_initializer(
                                                                 lang1_emb_np.transpose()),
                                                             name="%s_projection" % params["lang1"]),
                            params["lang2"]: tf.layers.Dense(params["lang2_vocab_size"],
                                                             use_bias=True,
                                                             kernel_initializer=tf.constant_initializer(
                                                                 lang2_emb_np.transpose()),
                                                             name="%s_projection" % params["lang2"])}
            else:
                pro_embs = {params["lang1"]: tf.layers.Dense(params["lang1_vocab_size"],
                                                             use_bias=True,
                                                             name="%s_projection" % params["lang1"]),
                            params["lang2"]: tf.layers.Dense(params["lang2_vocab_size"],
                                                             use_bias=True,
                                                             name="%s_projection" % params["lang2"])}

    with tf.variable_scope(scope or "all_embeddings", dtype=dtype) as scope:

        # encoder embeddings
        with tf.variable_scope("encoder", partitioner=lang1_partitioner):
            lang = "share" if params["share_lang_emb"] else params["lang1"]
            lang1_enc_embedding = _create_embed(
                "%s_embedding" % lang, params["lang1_vocab_size"], params["hidden_size"], dtype,
                lang1_emb_np)

        if params["share_lang_emb"]:
            if params["lang1_vocab_size"] != params["lang2_vocab_size"]:
                raise ValueError("Share embedding but different vocab sizes"
                                 " %d vs. %d" % (params["lang1_vocab_size"], params["lang2_vocab_size"]))
            assert params["lang1_vocab_size"] == params["lang2_vocab_size"]

            misc_utils.print_out("# Use the same encoder embedding for both languages.")
            lang2_enc_embedding = lang1_enc_embedding

        else:
            with tf.variable_scope("encoder", partitioner=lang2_partitioner):
                lang2_enc_embedding = _create_embed(
                    "%s_embedding" % params["lang2"], params["lang2_vocab_size"], params["hidden_size"], dtype,
                    lang2_emb_np)

        encoder_embeddings[params["lang1"]] = lang1_enc_embedding
        encoder_embeddings[params["lang2"]] = lang2_enc_embedding

        # decoder embeddings
        if params["share_encdec_emb"]:
            misc_utils.print_out("# Use the same embedding for encoder and decoder of each language.")
            decoder_embeddings = encoder_embeddings

        else:
            with tf.variable_scope("decoder", partitioner=lang1_partitioner):
                lang = "share" if params["share_lang_emb"] else params["lang1"]
                lang1_dec_embedding = _create_embed(
                    "%s_embedding" % lang, params["lang1_vocab_size"], params["hidden_size"], dtype,
                    lang1_emb_np)

                if params["share_lang_emb"]:
                    misc_utils.print_out("# Use the same decoder embedding for both languages.")
                    lang2_dec_embedding = lang1_dec_embedding

                else:
                    lang2_dec_embedding = _create_embed(
                        "%s_embedding" % params["lang2"], params["lang2_vocab_size"], params["hidden_size"], dtype,
                        lang2_emb_np)

                decoder_embeddings[params["lang1"]] = lang1_dec_embedding
                decoder_embeddings[params["lang2"]] = lang2_dec_embedding

    return encoder_embeddings, decoder_embeddings, pro_embs


