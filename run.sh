#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

CODES=60000

SRC=en
TGT=fr

UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para

MOSES_BLEU=$TOOLS_PATH/mosesdecoder/scripts/generic/multi-bleu.perl

TF_FULL_VOCAB=$MONO_PATH/vocab.$SRC-$TGT.$CODES.tf

PRETRAINED_EMB=$MONO_PATH/all.$SRC-$TGT.$CODES.vec

SRC_TRAIN_BPETOK=$MONO_PATH/all.$SRC.tok.$CODES
TGT_TRAIN_BPETOK=$MONO_PATH/all.$TGT.tok.$CODES

SRC_TEST_BPETOK=$PARA_PATH/dev/newstest2014-fren-src.fr.$CODES
TGT_TEST_BPETOK=$PARA_PATH/dev/newstest2014-fren-src.en.$CODES
SRC_TEST_REF=$PARA_PATH/dev/newstest2014-fren-src.en
TGT_TEST_REF=$PARA_PATH/dev/newstest2014-fren-src.fr

SAVED_MODEL_DIR=$PWD/saved_ckpt

python main.py \
  --batch_size=2048 \
  --bucket_width=5 \
  --infer_batch_size=64 \
  --lang1=$SRC \
  --lang2=$TGT \
  --share_lang_emb=True \
  --share_output_emb=True \
  --lang1_embed_file=$PRETRAINED_EMB \
  --lang2_embed_file=$PRETRAINED_EMB \
  --pretrained_out=True \
  --lang1_vocab_file=$TF_FULL_VOCAB \
  --lang2_vocab_file=$TF_FULL_VOCAB \
  --lang1_train_data=$SRC_TRAIN_BPETOK \
  --lang2_train_data=$TGT_TRAIN_BPETOK \
  --lang1_valid_data=$SRC_TEST_BPETOK \
  --lang2_valid_data=$TGT_TEST_BPETOK \
  --lang1to2_ref=$TGT_TEST_REF \
  --lang2to1_ref=$SRC_TEST_REF \
  --prepostprocess_dropout=0.0 \
  --attention_dropout=0.0 \
  --relu_dropout=0.0 \
  --n_enc_layers=4 \
  --share_enc=3 \
  --n_dec_layers=4 \
  --share_dec=3 \
  --allow_ffn_pad=False \
  --label_smoothing=0.0 \
  --learning_rate=0.0001 \
  --noam_decay=False \
  --model_dir=$SAVED_MODEL_DIR \
  --moses_bleu_script=$MOSES_BLEU \
  --only_infer=False

