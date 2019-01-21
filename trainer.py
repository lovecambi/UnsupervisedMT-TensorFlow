# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import tensorflow as tf
import numpy as np

import trainer_utils
from model import unmt, misc_utils, common_layers


class TrainerMT(unmt.UNMT):
    """
    TODO: FB PyTorch code uses aysnchronous back translaton generation.
    e.g., a minimal description
    thread 0: continously generate back translation, synchronize the model every 1K training steps.
    thread 1: continously training for ae and bt losses, don't take care of generation.
    Need to figure it out in TensorFlow.
    """

    def __init__(self,
                 params,
                 mode,
                 iterator,
                 reverse_vocab_tables=None,
                 eos_id=1):
        """
        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            mode: tf.estimator.ModeKeys. Used to determine if dropout layers should be added.
            iterator: data loader
            reverse_vocab_tables: for decoding
            eos_id: for decoding
        """
        self.iterator = iterator
        self.mode = mode

        if mode == tf.estimator.ModeKeys.TRAIN:
            self.lambda_xe = tf.placeholder(tf.float32, shape=[])
            # prepare data for auto-encoder loss
            input_lang1, noise_input_lang1 = self.iterator.lang1_iterator.get_next()
            input_lang2, noise_input_lang2 = self.iterator.lang2_iterator.get_next()

            self.noise_inputs = {params["lang1"]: noise_input_lang1,
                                 params["lang2"]: noise_input_lang2}
            self.clean_inputs = {params["lang1"]: input_lang1,
                                 params["lang2"]: input_lang2}

            # prepare data for back translation inference
            new_input_lang1, _ = self.iterator.lang1_iterator.get_next()
            new_input_lang2, _ = self.iterator.lang2_iterator.get_next()

            self.new_clean_inputs = {params["lang1"]: new_input_lang1,
                                     params["lang2"]: new_input_lang2}

            # prepare data for back translation loss
            self.bt_noise_input1 = tf.placeholder(tf.int64, shape=[None, None])  # 2to1 infer
            self.bt_noise_input2 = tf.placeholder(tf.int64, shape=[None, None])  # 1to2 infer
            self.bt_clean_input1 = tf.placeholder(tf.int64, shape=[None, None])
            self.bt_clean_input2 = tf.placeholder(tf.int64, shape=[None, None])

            self.bt_noise_inputs = {params["lang1"]: self.bt_noise_input1,
                                    params["lang2"]: self.bt_noise_input2}
            self.bt_clean_inputs = {params["lang1"]: self.bt_clean_input1,
                                    params["lang2"]: self.bt_clean_input2}

        elif mode == tf.estimator.ModeKeys.EVAL:
            # for back translation inference
            self.infer_input1 = tf.placeholder(tf.int64, shape=[None, None])
            self.infer_input2 = tf.placeholder(tf.int64, shape=[None, None])
            self.infer_inputs = {params["lang1"]: self.infer_input1,
                                 params["lang2"]: self.infer_input2}

        elif mode == tf.estimator.ModeKeys.PREDICT:
            infer_input1, infer_input2 = self.iterator.iterator.get_next()
            self.infer_inputs = {params["lang1"]: infer_input1,
                                 params["lang2"]: infer_input2}
            
        super(TrainerMT, self).__init__(params=params, mode=mode, eos_id=eos_id)

        # start to build model, train_op and saver
        self.global_step = tf.Variable(0, trainable=False)
        res = self.build_model(params)
        self._set_train_or_infer(res, reverse_vocab_tables, params)
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=params["num_keep_ckpts"])

    def build_model(self, params):
        with tf.variable_scope(
            "model", initializer=tf.variance_scaling_initializer(
                params["initializer_gain"], mode="fan_avg", distribution="uniform")):

            if self.mode == tf.estimator.ModeKeys.TRAIN:
                # auto-encoding loss
                ae_loss = self.dual_unsupervised_loss(self.noise_inputs, self.clean_inputs, params)

                # supervised loss via back translation results
                bt_loss = self.dual_supervised_loss(self.bt_noise_inputs, self.bt_clean_inputs, params)

                sample_ids = None

            else:
                # if mode is tf.estimator.ModeKeys.EVAL, infer from placehoder
                # if mode is tf.estimator.ModeKeys.PREDICT, infer from iterator
                sample_ids = self.dual_infer(self.infer_inputs, params)

                ae_loss, bt_loss = None, None

            return ae_loss, bt_loss, sample_ids
    
    def dual_unsupervised_loss(self, src_inputs, tgt_inputs, params):
        """unparallel inputs
        for auto-encoding training
        Args:
            src_inputs: dict, {l1:, l2:}
            tgt_inputs: dict, {l1:, l2:}
            params:
        Return:
        """
        usv_loss = 0.
        for lang in [params["lang1"], params["lang2"]]:
            logits = self.enc_dec_step(
                src_inputs[lang],
                tgt_inputs[lang],
                lang,
                lang)
            xentropy, weights = common_layers.padded_cross_entropy_loss(
                logits,
                tgt_inputs[lang],
                params["label_smoothing"])
            usv_loss += tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
        return usv_loss

    def dual_supervised_loss(self, src_inputs, tgt_inputs, params):
        """cross parallel input, i.e., src[l1] ~ tgt[l2], src[l2] ~ tgt[l1]
        for supervised training or back-translation
        Args:
            src_inputs: dict, {l1:, l2:}
            tgt_inputs: dict, {l1:, l2:}
            params:
        Return:
        """
        sv_loss = 0.
        for lang1, lang2 in [(params["lang1"], params["lang2"]), (params["lang2"], params["lang1"])]:
            logits = self.enc_dec_step(
                src_inputs[lang1],
                tgt_inputs[lang2],
                lang1,
                lang2)
            xentropy, weights = common_layers.padded_cross_entropy_loss(
                logits,
                tgt_inputs[lang2],
                params["label_smoothing"])
            sv_loss += tf.reduce_sum(xentropy) / tf.reduce_sum(weights)
        return sv_loss

    def dual_infer(self, infer_inputs, params):
        """
        Args:
            infer_inputs: dict, {l1:, l2:}
            params:
        Return:
        """
        sample_ids = []
        for lang1, lang2 in [(params["lang1"], params["lang2"]), (params["lang2"], params["lang1"])]:
            sample_ids_x2x, _ = self.greedy_predict(infer_inputs[lang1], lang1, lang2)
            sample_ids.append(sample_ids_x2x)
        return sample_ids

    def _set_train_or_infer(self, res, reverse_vocab_tables, params):
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.ae_loss, self.bt_loss, _ = res
        else:
            _, _, sample_ids = res
            self.sample_ids_1to2, self.sample_ids_2to1 = sample_ids

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            self.sample_words_1to2 = reverse_vocab_tables[params["lang2"]].lookup(tf.to_int64(self.sample_ids_1to2))
            self.sample_words_2to1 = reverse_vocab_tables[params["lang1"]].lookup(tf.to_int64(self.sample_ids_2to1))

        # start to optimize
        tvars = tf.trainable_variables()

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.learning_rate = trainer_utils.get_learning_rate(
                learning_rate=params["learning_rate"],
                step=self.global_step,
                hidden_size=params["hidden_size"],
                learning_rate_warmup_steps=params["learning_rate_warmup_steps"],
                noam_decay=params["noam_decay"])

            optimizer = tf.contrib.opt.LazyAdamOptimizer(
                self.learning_rate,
                beta1=params["optimizer_adam_beta1"],
                beta2=params["optimizer_adam_beta2"],
                epsilon=params["optimizer_adam_epsilon"])

            self.ae_train_op = tf.contrib.layers.optimize_loss(
                self.lambda_xe * self.ae_loss,
                self.global_step,
                learning_rate=None,
                optimizer=optimizer,
                variables=tvars,
                clip_gradients=params["clip_grad_norm"],
                colocate_gradients_with_ops=True,
                increment_global_step=False)
            
            self.bt_train_op = tf.contrib.layers.optimize_loss(
                self.lambda_xe * self.bt_loss,
                self.global_step,
                learning_rate=None,
                optimizer=optimizer,
                variables=tvars,
                clip_gradients=params["clip_grad_norm"],
                colocate_gradients_with_ops=True,
                increment_global_step=True)
 
            self.train_ae_summary = tf.summary.merge([tf.summary.scalar("lr", self.learning_rate),
                                                      tf.summary.scalar("ae_loss", self.ae_loss)])
            self.train_bt_summary = tf.summary.merge([tf.summary.scalar("lr", self.learning_rate),
                                                      tf.summary.scalar("bt_loss", self.bt_loss)])

            misc_utils.print_out("# Trainable variables")
            misc_utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
            for tvar in tvars:
                misc_utils.print_out("  %s, %s, %s" % (tvar.name, str(tvar.get_shape()),
                                                       tvar.op.device))

    def ae_updates(self, sess, lambda_xe_mono):
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        return sess.run([self.ae_train_op,
                         self.ae_loss,
                         self.train_ae_summary,
                         self.global_step,
                         self.learning_rate,
                         self.new_clean_inputs],
                        feed_dict={self.lambda_xe: lambda_xe_mono})

    def bt_updates(self, sess, lambda_xe_otfb, bt_ni1, bt_ni2, bt_ci1, bt_ci2):
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        return sess.run([self.bt_train_op,
                         self.bt_loss,
                         self.train_bt_summary,
                         self.global_step,
                         self.learning_rate],
                        feed_dict={self.lambda_xe: lambda_xe_otfb,
                                   self.bt_noise_input1: bt_ni1,
                                   self.bt_noise_input2: bt_ni2,
                                   self.bt_clean_input1: bt_ci1,
                                   self.bt_clean_input2: bt_ci2})

    def infer(self, sess):
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        return sess.run([self.sample_words_1to2,
                         self.sample_words_2to1])

    def otfb(self, sess, ii1, ii2):
        assert self.mode == tf.estimator.ModeKeys.EVAL
        return sess.run([self.sample_ids_1to2,
                         self.sample_ids_2to1],
                        feed_dict={self.infer_input1: ii1,
                                   self.infer_input2: ii2})


def add_summary(summary_writer, global_step, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def init_stats():
    """Initialize statistics that we want to accumulate."""
    return {"step_time": 0.0, "ae_loss": 0.0, "bt_loss": 0.0, }


def update_stats(stats, start_time, step_result):
    """Update stats: write summary and accumulate statistics."""
    if isinstance(step_result, list):
        ae_step_result, bt_step_result = step_result
        _, ae_step_loss, _, _, _ = ae_step_result
        _, bt_step_loss, step_summary, global_step, learning_rate = bt_step_result
    else:
        _, _, ae_step_loss, bt_step_loss, step_summary, global_step, learning_rate = step_result

    # Update statistics
    stats["step_time"] += (time.time() - start_time)
    stats["ae_loss"] += ae_step_loss
    stats["bt_loss"] += bt_step_loss
    return global_step, learning_rate, step_summary


def process_stats(stats, info, global_step, steps_per_stats, log_f):
    """Update info and check for overflow."""
    # Update info
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_train_ae_loss"] = stats["ae_loss"] / steps_per_stats
    info["avg_train_bt_loss"] = stats["bt_loss"] / steps_per_stats

    is_overflow = False
    for avg_loss in [info["avg_train_ae_loss"], info["avg_train_bt_loss"]]:
        if math.isnan(avg_loss) or math.isinf(avg_loss) or avg_loss > 1e20:
            misc_utils.print_out("  step %d overflow loss, stop early" % global_step, log_f)
            is_overflow = True
            break
    return is_overflow


def print_step_info(prefix, global_step, info, log_f):
    """Print all info at the current global step."""
    misc_utils.print_out("%sstep %d lr %g step-time %.2fs ae_loss %.4f bt_loss %.4f, %s" %
                         (prefix, global_step, info["learning_rate"], info["avg_step_time"],
                          info["avg_train_ae_loss"], info["avg_train_bt_loss"], time.ctime()), log_f)


def before_train(loaded_train_model, train_model, train_sess, global_step, log_f):
    """Misc tasks to do before training."""
    stats = init_stats()
    info = {"avg_step_time": 0.0,
            "avg_train_ae_loss": 0.0,
            "avg_train_bt_loss": 0.0,
            "learning_rate": loaded_train_model.learning_rate.eval(
                session=train_sess)}
    start_train_time = time.time()
    misc_utils.print_out("# Start step %d, lr %g, %s" %
                         (global_step, info["learning_rate"], time.ctime()), log_f)

    # Initialize all of the iterators
    train_sess.run(train_model.iterator.initializer)

    return stats, info, start_train_time


def sync_eval_model(eval_model, eval_sess, model_dir):
    with eval_model.graph.as_default():
        loaded_eval_model, _ = trainer_utils.create_or_load_model(
            eval_model.model, model_dir, eval_sess, "eval")
    return loaded_eval_model


def train_and_eval(params, target_session=""):
    out_dir = params["model_dir"]
    steps_per_stats = params["steps_per_stats"]
    steps_per_eval = 10 * steps_per_stats

    # Log and output files
    log_file = os.path.join(out_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    misc_utils.print_out("# log_file=%s" % log_file, log_f)

    # create models
    model_creator = TrainerMT
    train_model = trainer_utils.create_train_model(model_creator, params)
    eval_model = trainer_utils.create_eval_model(model_creator, params)
    infer_model = trainer_utils.create_infer_model(model_creator, params)

    # TensorFlow models
    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True

    train_sess = tf.Session(target=target_session, config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(target=target_session, config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(target=target_session, config=config_proto, graph=infer_model.graph)

    with train_model.graph.as_default():
        loaded_train_model, global_step = trainer_utils.create_or_load_model(
            train_model.model, params["model_dir"], train_sess, "train")

    # Summary writer
    summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, "train_log"), train_model.graph)

    # First evaluation without training yet
    # run_external_eval(infer_model, infer_sess, params["model_dir"], params, summary_writer)

    last_stats_step = global_step
    last_eval_step = global_step

    # This is the train loop.
    trainer_utils.separate_shuffle(
        [params["lang1_train_data"], params["lang2_train_data"]], params["train_data_suffix"])
    stats, info, start_train_time = before_train(
        loaded_train_model, train_model, train_sess, global_step, log_f)
    lambda_xe_mono_config = trainer_utils.parse_lambda_config(params["lambda_xe_mono"])

    loaded_eval_model = sync_eval_model(eval_model, eval_sess, params["model_dir"])

    while global_step < params["num_train_steps"]:
        # Run a step
        start_time = time.time()
        lambda_xe_mono = trainer_utils.get_lambda_xe_mono(lambda_xe_mono_config, global_step)
        try:
            ae_step_result = loaded_train_model.ae_updates(train_sess, lambda_xe_mono)

            new_clean_inputs = ae_step_result[-1]
            ii1, ii2 = new_clean_inputs[params["lang1"]], new_clean_inputs[params["lang2"]]
            ids1to2, ids2to1 = loaded_eval_model.otfb(eval_sess, ii1, ii2)

            bt_step_result = loaded_train_model.bt_updates(
                train_sess, params["lambda_xe_otfb"], ids2to1, ids1to2, ii1, ii2)

            step_result = [ae_step_result[:-1], bt_step_result]
        except tf.errors.OutOfRangeError:
            misc_utils.print_out("# Finished Training of One Epochs.")

            trainer_utils.separate_shuffle(
                [params["lang1_train_data"], params["lang2_train_data"]], params["train_data_suffix"])
            train_sess.run(train_model.iterator.initializer)
            continue

        global_step, info["learning_rate"], step_summary = update_stats(stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)

        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step
            is_overflow = process_stats(stats, info, global_step, steps_per_stats, log_f)
            print_step_info("  ", global_step, info, log_f)
            if is_overflow:
                break
            # Reset statistics
            stats = init_stats()

        if global_step - last_eval_step >= steps_per_eval:
            last_eval_step = global_step

            misc_utils.print_out("# Save eval, global step %d" % global_step)
            loaded_train_model.saver.save(
                train_sess,
                os.path.join(params["model_dir"], "model.ckpt"),
                global_step=global_step)

            loaded_eval_model = sync_eval_model(eval_model, eval_sess, params["model_dir"])

            run_external_eval(infer_model, infer_sess, params["model_dir"], params, summary_writer)

    # Done training
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(params["model_dir"], "model.ckpt"),
        global_step=global_step)

    misc_utils.print_out("# Done training, time %ds!" % (time.time() - start_train_time))

    summary_writer.close()
    return global_step


def run_external_eval(infer_model, infer_sess, model_dir, params, summary_writer):
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = trainer_utils.create_or_load_model(
            infer_model.model, model_dir, infer_sess, "infer")

    out_dir = params["model_dir"]
    misc_utils.print_out("# External BLEU evaluation, global step %d" % global_step)

    infer_sess.run(infer_model.iterator.initializer)

    output = os.path.join(out_dir, "output_eval")
    tags = ["%s2%s" % (params["lang1"], params["lang2"]),
            "%s2%s" % (params["lang2"], params["lang1"])]
    pred_files = ["%s_%s" % (output, tag) for tag in tags]
    ref_files = [params["lang1to2_ref"], params["lang2to1_ref"]]

    scores = trainer_utils.decode_and_evaluate(
        tags,
        loaded_infer_model,
        infer_sess,
        pred_files,
        ref_files,
        bleu_script_path=params["moses_bleu_script"])

    for tag in scores:
        add_summary(summary_writer, global_step, "%s_BLEU" % tag, scores[tag])

    return scores, global_step


def dual_inference(params):
    misc_utils.print_out("# lang1_valid_data and lang2_valid_data are used for inference.")

    infer_model = trainer_utils.create_infer_model(TrainerMT, params)

    config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True

    ckpt_path = tf.train.latest_checkpoint(params["model_dir"])
    with tf.Session(graph=infer_model.graph, config=config_proto) as sess:
        loaded_infer_model = trainer_utils.load_model(
            infer_model.model, ckpt_path, sess, "infer")

        with infer_model.graph.as_default():
            sess.run(infer_model.iterator.initializer)

            output = os.path.join(params["model_dir"], "output_pred")
            tags = ["%s2%s" % (params["lang1"], params["lang2"]),
                    "%s2%s" % (params["lang2"], params["lang1"])]
            pred_files = ["%s_%s" % (output, tag) for tag in tags]
            ref_files = []

            trainer_utils.decode_and_evaluate(
                tags,
                loaded_infer_model,
                sess,
                pred_files,
                ref_files)  # unused since it is empty.

