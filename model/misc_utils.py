# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import codecs
import tensorflow as tf


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""
    print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()


def print_out(s, f=None, new_line=True):
    """Similar to print but with support to flush and output to a file."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")

    if f:
        f.write(s.encode("utf-8"))
        if new_line:
            f.write(b"\n")

    # stdout
    out_s = s.encode("utf-8")
    if not isinstance(out_s, str):
        out_s = out_s.decode("utf-8")
    print(out_s, end="", file=sys.stdout)

    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size


def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.
      Note: the embed_file should be a Glove/word2vec formatted txt file. Assuming
      Here is an exampe assuming embed_size=5:
      the -0.071549 0.093459 0.023738 -0.090339 0.056123
      to 0.57346 0.5417 -0.23477 -0.3624 0.4037
      and 0.20327 0.47348 0.050877 0.002103 0.060547
      For word2vec format, the first line will be: <num_words> <emb_size>.
      Args:
        embed_file: file path to the embedding file.
      Returns:
        a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None

    is_first_line = True
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, "rb")) as f:
        for line in f:
            tokens = line.rstrip().split(" ")
            if is_first_line:
                is_first_line = False
                if len(tokens) == 2:  # header line
                    emb_size = int(tokens[1])
                    continue
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                if emb_size != len(vec):
                    print_out(
                        "Ignoring %s since embeding size is inconsistent." % word)
                    del emb_dict[word]
            else:
                emb_size = len(vec)
    return emb_dict, emb_size
