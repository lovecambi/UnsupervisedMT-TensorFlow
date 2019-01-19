# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

import os
import argparse
import tensorflow as tf

from trainer_utils import count_lines
from model import misc_utils
from trainer import train_and_eval, dual_inference
#from trainer_dev import train_and_eval, dual_inference


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("--batch_size", type=int, default=2048,
                        help="If >= 1024, it means token level batch size, else sentence level.")
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="sentence level batch size when mode == PREDICT.")
    parser.add_argument("--bucket_width", type=int, default=5,
                        help="Put data into similar-length buckets, 5, 10, 15,...")
    parser.add_argument("--num_train_steps", type=int, default=1000000,
                        help="Total number of global steps of training.")
    parser.add_argument("--lambda_xe_mono", type=str, default="0:1,100000:0.1,300000:0",
                        help=("""
                        To scale the loss for auto-encoding loss, decay schedule see trainer_utils,.\
                        get_lambda_xe_mono() and parse_lambda_config(). 
                        """))
    parser.add_argument("--lambda_xe_otfb", type=float, default=1.0,
                        help="To scale the loss for back translation loss.")

    parser.add_argument("--lang1", type=str, default="en",
                        help="Language 1 identifier.")
    parser.add_argument("--lang2", type=str, default="fr",
                        help="See lang1.")
    parser.add_argument("--lang1_partitions", type=int, default=1,
                        help="Number of partitions for embedding matrix, set <= 1 if using pre-trained.")
    parser.add_argument("--lang2_partitions", type=int, default=1,
                        help="See lang1_partitions.")

    parser.add_argument("--share_lang_emb", type="bool", nargs="?", const=True, default=False,
                        help="Whether sharing the embeddings of two languages, for both encoder and decoder.")
    parser.add_argument("--share_encdec_emb", type="bool", nargs="?", const=True, default=False,
                        help="Whether sharing the embedding of encoder / decoder for each language.")
    parser.add_argument("--share_decpro_emb", type="bool", nargs="?", const=True, default=False,
                        help="Whether sharing the decoder embedding and projection matrix for each language.")
    parser.add_argument("--share_output_emb", type="bool", nargs="?", const=True, default=False,
                        help="Whether share projection matrix and biases of two languages.")

    parser.add_argument("--lang1_embed_file", type=str, default="./data/wmt/enfr/all.en-fr.60000.vec",
                        help="The pretrained embedding matrix for language 1, from fasttext.")
    parser.add_argument("--lang2_embed_file", type=str, default="./data/wmt/enfr/all.en-fr.60000.vec",
                        help="See lang1_embed_file.")
    parser.add_argument("--pretrained_out", type="bool", nargs="?", const=True, default=False,
                        help="Whether using pretrained embedding to initialize projection matrix.")

    parser.add_argument("--lang1_vocab_file", type=str, default="./data/wmt/enfr/vocab.en-fr.60000",
                        help=("""
                        The vocabulary file of language 1, first four should be: \
                        <s>, </s>, <unk>, <mask>
                        """))
    parser.add_argument("--lang2_vocab_file", type=str, default="./data/wmt/enfr/vocab.en-fr.60000",
                        help="See lang1_vocab_file.")
    parser.add_argument("--lang1_train_data", type=str, default="./data/wmt/enfr/all.en.tok.60000",
                        help="Monolingual training data for language 1.")
    parser.add_argument("--lang2_train_data", type=str, default="./data/wmt/enfr/all.fr.tok.60000",
                        help="See lang1_train_data.")
    parser.add_argument("--lang1_valid_data", type=str, default="./data/wmt/enfr/newstest2014-fren-src.en.60000",
                        help="Parallel valid data for language 1 after BPE.")
    parser.add_argument("--lang2_valid_data", type=str, default="./data/wmt/enfr/newstest2014-fren-src.fr.60000",
                        help="See lang1_valid_data.")
    parser.add_argument("--lang1to2_ref", type=str, default="./data/wmt/enfr/newstest2014-fren-src.fr",
                        help="Parallel valid data for language 2 before BPE.")
    parser.add_argument("--lang2to1_ref", type=str, default="./data/wmt/enfr/newstest2014-fren-src.en",
                        help="See lang1to2_ref.")
    parser.add_argument("--train_data_suffix", type=str, default="shuffled",
                        help="The suffix added to training data after cmd shuf.")

    parser.add_argument("--initializer_gain", type=float, default=1.0,
                        help="Multiplicative factor to apply to the matrix.")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="The number of hidden units of transformer.")
    parser.add_argument("--filter_size", type=int, default=2048,
                        help="The number of hidden units of feed forward layer in transformer.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="The number of self-attention heads.")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Total number of layers for transformer encoder.")
    parser.add_argument("--share_enc", type=int, default=3,
                        help="The number of shared encoder layers for two languages.")
    parser.add_argument("--n_dec_layers", type=int, default=4,
                        help="Total number of layers for transformer decoder.")
    parser.add_argument("--share_dec", type=int, default=3,
                        help="The number of shared decoder layers for two languages.")
    parser.add_argument("--allow_ffn_pad", type="bool", nargs="?", const=True, default=False,
                        help="Whether to remove padding when feeding to FFN layers. Enable may increase speed.")

    parser.add_argument("--prepostprocess_dropout", type=float, default=0.1,
                        help="Dropout rate for embedding.")
    parser.add_argument("--attention_dropout", type=float, default=0.1,
                        help="Dropout rate for self-attention layer.")
    parser.add_argument("--relu_dropout", type=float, default=0.1,
                        help="Dropout rate for feed forward layer.")

    parser.add_argument("--argmax", type="bool", nargs="?", const=True, default=True,
                        help="Sampling strategy for greedy decoding.")
    parser.add_argument("--sampling_temp", type=float, default=0.0,
                        help="Sampling temperature when argmax = False.")

    parser.add_argument("--label_smoothing", type=float, default=0.1, 
                        help="Soft label for computing xe loss when training.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,  # 2.0
                        help="Initial learning rate.")
    parser.add_argument("--noam_decay", type="bool", nargs="?", const=True, default=False,  # True
                        help="Inverse sqrt learning rate decay with warmup.")
    parser.add_argument("--learning_rate_warmup_steps", type=int, default=16000,
                        help="Warm up steps for noam decay.")

    parser.add_argument("--optimizer_adam_beta1", type=float, default=0.5,  # 0.9
                        help="beta1 for adam opt.")
    parser.add_argument("--optimizer_adam_beta2", type=float, default=0.999,  # 0.997
                        help="beta2 for adam opt.")
    parser.add_argument("--optimizer_adam_epsilon", type=float, default=1e-8,  # 1e-9
                        help="Initial learning rate.")
    parser.add_argument("--clip_grad_norm", type=float, default=5.0,
                        help="Clipping gradient norm.")

    parser.add_argument("--model_dir", type=str, default="./saved_ckpt",
                        help="Where to save the models.")
    parser.add_argument("--num_keep_ckpts", type=int, default=5,
                        help="How many models to save.")
    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help="Print out training stats every several steps.")
    parser.add_argument("--moses_bleu_script", type=str, default="../../mosesdecoder/scripts/generic/multi-bleu.perl",
                        help="Used for compute BLEU during training.")

    # only inference
    parser.add_argument("--only_infer", type="bool", nargs="?", const=True, default=False,
                        help="Only inference from saved model dir.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    params = vars(parser.parse_args())

    params["lang1_vocab_size"] = count_lines(params["lang1_vocab_file"])
    if params["lang1_vocab_file"] == params["lang2_vocab_file"]:
        params["lang2_vocab_size"] = params["lang1_vocab_size"]
    else:
        params["lang2_vocab_size"] = count_lines(params["lang2_vocab_file"])

    misc_utils.print_out("# All hyperparameters:")
    for key in params:
        misc_utils.print_out("%s=%s" % (key, str(params[key])))

    if params["batch_size"] >= 1024:
        misc_utils.print_out("# batch_size >= 1024 indicates token level batch size for training.")

    if params["only_infer"]:
        if not tf.gfile.Exists(params["model_dir"]):
            raise ValueError("No checkpoint saved in %s" % params["model_dir"])
        dual_inference(params)
    else:
        if params["model_dir"] and not tf.gfile.Exists(params["model_dir"]):
            misc_utils.print_out("# Creating saved model directory %s ..." % params["model_dir"])
            tf.gfile.MakeDirs(params["model_dir"])
        assert os.path.isfile(params["moses_bleu_script"]), "%s not found. Please be sure you downloaded Moses" % \
                                                            params["moses_bleu_script"]
        train_and_eval(params)
