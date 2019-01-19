# Unsupervised Machine Translation (Transformer Based UNMT)

This repository provides the TensorFlow implementation of the transformer based unsupervised NMT model presented in  
[Phrase-Based & Neural Unsupervised Machine Translation](https://arxiv.org/abs/1804.07755) (EMNLP 2018).

## Requirements
* Python 3
* TensorFlow 1.12
* [Moses](http://www.statmt.org/moses/) (clean and tokenize text)
* [fastBPE](https://github.com/glample/fastBPE) (generate and apply BPE codes)
* [fastText](https://github.com/facebookresearch/fastText) (generate embeddings)
* (optional) [MUSE](https://github.com/facebookresearch/MUSE) (generate cross-lingual embeddings)

The data preprocessing script `get_enfr_data.sh` (copied from [UnsupervisedMT-Pytorch](https://github.com/facebookresearch/UnsupervisedMT), 
but remove the cmd of binarizing dataset by torch and add the special tokens to the vocabulary files) will take care of installing everything (except Python, TensorFlow).

### Download / preprocess data

The first thing to do to download and preprocess data. To do so, just run:

```
cd UnsupervisedMT-TensorFlow
./get_enfr_data.sh
```

Note that there are several ways to train cross-lingual embeddings:
- Train monolingual embeddings separately for each language, and align them with [MUSE](https://github.com/facebookresearch/MUSE) (please refer to [the original paper](https://openreview.net/pdf?id=H196sainb) for more details).
- Concatenate the source and target monolingual corpora in a single file, and train embeddings with fastText on that generated file (this is what is implemented in the `get_enfr_data.sh` script).

The second method works better when the source and target languages are similar and share a lot of common words (such as French and English in `get_enfr_data.sh`). However, when the overlap between the source and target vocabulary is too small, the alignment will be very poor and you should opt for the first method using MUSE to generate your cross-lingual embeddings.


You can skip the script for preprocessing since it takes long time including downloading, learning/applying BPE, and fasttext training. 
Just download the prepared datasets, after running `get_enfr_data.sh`.
```
cd UnsupervisedMT-TensorFlow
./download_enfr_data.sh
```

### Train the NMT model

```
./run.sh
```

The hyperparameter in `run.sh` are almost identical to the [UnsupervisedMT-Pytorch](https://github.com/facebookresearch/UnsupervisedMT) except the `batch_size=2048` which is a token level batch size.

On newstest2014 en-fr, the above command should give about 23.0 BLEU after 100K steps training on a P100.

## TODO
* Supervised Training
* Multi-GPUs Training
* Beam Search

## References Github
* [facebookresearch/UnsupervisedMT](https://github.com/facebookresearch/UnsupervisedMT)
* [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)
* [tensorflow/models/official/transformer](https://github.com/tensorflow/models/tree/master/official/transformer)
