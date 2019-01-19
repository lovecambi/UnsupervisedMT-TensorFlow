# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

import os
import codecs
import time
import subprocess
import collections
from multiprocessing import Pool

from model import misc_utils as utils
import unsupervised_iterator
import supervised_iterator

import tensorflow as tf


def get_learning_rate(learning_rate, step, hidden_size, learning_rate_warmup_steps, noam_decay=True):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        learning_rate = tf.constant(learning_rate)

        if noam_decay:
            warmup_steps = tf.to_float(learning_rate_warmup_steps)
            step = tf.to_float(step) + 1.0
            # noam decay
            inv_decay = (hidden_size ** -0.5) * tf.minimum(step * warmup_steps ** -1.5,
                                                           step ** -0.5)
            learning_rate *= inv_decay

        tf.identity(learning_rate, "learning_rate")

        return learning_rate


def decode_and_evaluate(tags,
                        model,
                        sess,
                        pred_files,
                        ref_files,
                        tgt_eos="</s>",
                        bleu_script_path="~/mosesdecoder/scripts/generic/multi-bleu.perl"):
    start_time = time.time()
    num_sentences = 0
    if tgt_eos:
        tgt_eos = tgt_eos.encode("utf-8")

    pred_file_1to2, pred_file_2to1 = pred_files
    with codecs.getwriter("utf-8")(tf.gfile.GFile(pred_file_1to2, mode="w")) as pred_f_1to2:
        with codecs.getwriter("utf-8")(tf.gfile.GFile(pred_file_2to1, mode="w")) as pred_f_2to1:
            pred_f_1to2.write("")
            pred_f_2to1.write("")
            while True:
                try:
                    sample_results = model.infer(sess)
                    batch_size = sample_results[0].shape[0]
                    for sample_words, pred_f in zip(sample_results, (pred_f_1to2, pred_f_2to1)):
                        for sent_id in range(batch_size):
                            output = sample_words[sent_id].tolist()
                            if tgt_eos and tgt_eos in output:
                                output = output[:output.index(tgt_eos)]
                            # pred_f.write((b" ".join(output) + b"\n").decode("utf-8"))
                            pred_f.write(
                                (b" ".join(output).replace(b"@@ ", b"").replace(b"@@", b"") + b"\n").decode("utf-8"))
                    num_sentences += batch_size
                except tf.errors.OutOfRangeError:
                    utils.print_out("  done, num sentences 2 * %d, time %ds" % (num_sentences, time.time() - start_time))
                    break

    # Evaluation
    scores = {}
    if len(ref_files) == len(pred_files):
        for ref_file, pred_file, tag in zip(ref_files, pred_files, tags):
            bleu = eval_moses_bleu(ref_file, pred_file, bleu_script_path)
            scores[tag] = bleu
            utils.print_out(" %s BLEU: %.2f" % (tag, bleu))
    return scores


def eval_moses_bleu(ref, hyp, bleu_script_path):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(ref) and os.path.isfile(hyp)
    command = bleu_script_path + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        utils.print_out('Impossible to parse BLEU score! "%s"' % result)
        return -1


def shuffle_single_train(data_file, suffix):
    assert os.path.isfile(data_file)
    new_data_file = "%s.%s" % (data_file, suffix)
    if os.path.isfile(new_data_file):
        p = subprocess.Popen("rm -f %s" % new_data_file,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True,
                             universal_newlines=True)
        p.communicate()

    command = "shuf %s > %s" % (data_file, new_data_file)
    p = subprocess.Popen(command,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True,
                         universal_newlines=True)
    p.communicate()


def separate_shuffle(data_files, suffix):
    assert isinstance(data_files, list)
    start_time = time.time()
    with Pool(2) as p:
        p.starmap(shuffle_single_train, [(data_file, suffix) for data_file in data_files])
    utils.print_out(
        "  Shuffled monolingual training datasets separately, time %.2fs" % (time.time() - start_time))


class CreatedModel(
        collections.namedtuple("CreatedModel", ("graph", "model", "iterator"))):
    pass


def create_train_model(model_creator, params):
    graph = tf.Graph()

    with graph.as_default(), tf.container("train"):
        iterator = unsupervised_iterator.get_iterator_txt(
            is_training=True,
            lang1_data_file="%s.%s" % (params["lang1_train_data"], params["train_data_suffix"]),
            lang2_data_file="%s.%s" % (params["lang2_train_data"], params["train_data_suffix"]),
            lang1_vocab_file=params["lang1_vocab_file"],
            lang2_vocab_file=params["lang2_vocab_file"],
            batch_size=params["batch_size"],
            bucket_width=params["bucket_width"],
            lang1_num_repeat=1,
            lang2_num_repeat=1)

        model = model_creator(
            params,
            mode=tf.estimator.ModeKeys.TRAIN,
            iterator=iterator)

    return CreatedModel(
        graph=graph,
        model=model,
        iterator=iterator)


def create_infer_model(model_creator, params):
    graph = tf.Graph()

    with graph.as_default(), tf.container("infer"):
        lang1_reverse_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
            params["lang1_vocab_file"], default_value="<unk>")
        if params["share_lang_emb"]:
            lang2_reverse_vocab_table = lang1_reverse_vocab_table
        else:
            lang2_reverse_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
                params["lang2_vocab_file"], default_value="<unk>")
        reverse_vocab_tables = {params["lang1"]: lang1_reverse_vocab_table,
                                params["lang2"]: lang2_reverse_vocab_table}

        iterator = supervised_iterator.get_iterator_txt(
            is_training=False,
            lang1_data_file=params["lang1_valid_data"],
            lang2_data_file=params["lang2_valid_data"],
            lang1_vocab_file=params["lang1_vocab_file"],
            lang2_vocab_file=params["lang2_vocab_file"],
            batch_size=params["infer_batch_size"],
            bucket_width=0,
            num_repeat=1)

        model = model_creator(
            params,
            mode=tf.estimator.ModeKeys.PREDICT,
            iterator=iterator,
            reverse_vocab_tables=reverse_vocab_tables)

    return CreatedModel(
        graph=graph,
        model=model,
        iterator=iterator)


class EvalModel(
        collections.namedtuple("CreatedModel", ("graph", "model"))):
    pass


def create_eval_model(model_creator, params):
    graph = tf.Graph()

    with graph.as_default(), tf.container("eval"):
        model = model_creator(
            params,
            mode=tf.estimator.ModeKeys.EVAL,
            iterator=None,
            reverse_vocab_tables=None)

    return EvalModel(
        graph=graph,
        model=model)


def load_model(model, ckpt_path, session, name):
    """Load model from a checkpoint."""
    start_time = time.time()
    try:
        model.saver.restore(session, ckpt_path)
    except tf.errors.NotFoundError as e:
        utils.print_out("Can't load checkpoint")
        utils.print_out("%s" % str(e))

    session.run(tf.tables_initializer())
    utils.print_out(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, ckpt_path, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                        (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


def get_lambda_xe_mono(config, step):
    ranges = [i for i in range(len(config) - 1) if config[i][0] <= step < config[i + 1][0]]
    if len(ranges) == 0:
        assert step >= config[-1][0]
        return config[-1][1]
    assert len(ranges) == 1
    i = ranges[0]
    x_a, y_a = config[i]
    x_b, y_b = config[i + 1]
    return y_a + (step - x_a) * float(y_b - y_a) / float(x_b - x_a)


def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000 iterations, then will linearly increase to 1 until iteration 2000
    """
    assert isinstance(x, str)
    split = x.split(',')
    assert len(split) > 1
    split = [s.split(':') for s in split]
    assert all(len(s) == 2 for s in split)
    assert all(k.isdigit() for k, _ in split)
    assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
    return [(int(k), float(v)) for k, v in split]


def count_lines(filename):
    """Returns the number of lines of the file :obj:`filename`."""
    with tf.gfile.Open(filename, mode="rb") as f:
        i = 0
        for i, _ in enumerate(f):
            pass
        return i + 1

