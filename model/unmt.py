# coding=utf-8
# author: Kai Fan
# email: interfk@gmail.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import embedding_utils
from model import common_attention
from model import common_layers

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.client import device_lib


class UNMT(object):
    """Unsupervised Neural Machine Translation Model"""

    def __init__(self,
                 params,
                 mode,
                 eos_id=1):
        """
        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            mode: tf.estimator.ModeKeys. Used to determine if dropout layers should be added.
            eos_id: for decoding
        """
        self.is_training = mode == tf.estimator.ModeKeys.TRAIN
        self.params = params
        self.eos_id = eos_id

        # prepare basic modules
        self.encoder_embeddings, self.decoder_embeddings, self.pro_stack = embedding_utils.get_all_embeddings(params)

        self.share_enc_stack = SharedEncoderStack(params, self.is_training)
        self.enc_stack = {params["lang1"]: EncoderStack(params, self.is_training),
                          params["lang2"]: EncoderStack(params, self.is_training)}

        self.share_dec_stack = SharedDecoderStack(params, self.is_training)
        self.dec_stack = {params["lang1"]: DecoderStack(params, self.is_training),
                          params["lang2"]: DecoderStack(params, self.is_training)}

    def enc_dec_step(self, ed_input, ed_target, lang1, lang2):
        """
        Args:
            ed_input:
            ed_target:
            lang1:
            lang2:
        Return:
        """
        # training from lang1 -> lang2
        # if lang1 == lang2, it is auto encoding step.
        # else, it is parallel training.
        enc_output, enc_attn_bias = self._encode_one(ed_input, lang1)
        dec_output = self._decode_one(ed_target, enc_output, enc_attn_bias, lang2)
        logits = self._projection_one(dec_output, lang2)
        return logits

    def otfb_step(self, otfb_input, lang1, lang2):
        # on-the-fly back-translation
        # lang1 -> lang2 -> lang1
        # inputs -> predictions -> inputs

        # Inference from translation model: lang1 -> lang2
        # It is not recommend to use this function 
        # 1. The predict step within the training graph results in slow training.
        # 2. The mode will still be TRAIN for greedy_predict
        devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
        assert len(devices) > 1
        with tf.device(devices[-1]):
            otfb_pred, _ = self.greedy_predict(otfb_input, lang1, lang2)
            otfb_pred = tf.stop_gradient(otfb_pred)

        # training the translation model: lang2 -> lang1
        return self.enc_dec_step(otfb_pred, otfb_input, lang2, lang1)

    def _encode_one(self, e_input, lang):
        """
        Args
            e_input: int tensor, [batch_size, input_length].
            lang: string, identifier of language, e.g., "en".
        Return:
        """
        e_emb, e_attn_bias, e_pad = common_layers.transformer_prepare_encoder(
            e_input, self.encoder_embeddings[lang], "%s_prepare_encoder" % lang)
       
        e_emb *= self.params["hidden_size"] ** 0.5

        e_emb = common_layers.add_timing_signal_1d(e_emb)

        if self.is_training:
            e_emb = tf.nn.dropout(e_emb, 1.0 - self.params["prepostprocess_dropout"])

        with tf.variable_scope("shared_encode", reuse=tf.AUTO_REUSE):
            e_emb = self.share_enc_stack(e_emb, e_attn_bias, e_pad)

        with tf.variable_scope("%s_encode" % lang, reuse=tf.AUTO_REUSE):
            e_output = self.enc_stack[lang](e_emb, e_attn_bias, e_pad)

        return e_output, e_attn_bias

    def _decode_one(self, d_input, e_output, e_attn_bias, lang):
        """
        Args:
            d_input:
            e_output:
            e_attn_bias:
            lang:
        Return:
        """
        d_emb, d_attn_bias = common_layers.transformer_prepare_decoder(
            d_input, self.decoder_embeddings[lang], "%s_prepare_decoder" % lang)

        d_emb *= self.params["hidden_size"] ** 0.5

        d_emb = common_layers.add_timing_signal_1d(d_emb)

        if self.is_training:
            d_emb = tf.nn.dropout(d_emb, 1.0 - self.params["prepostprocess_dropout"])

        with tf.variable_scope("shared_decode", reuse=tf.AUTO_REUSE):
            d_emb = self.share_dec_stack(d_emb, e_output, d_attn_bias, e_attn_bias)

        with tf.variable_scope("%s_decode" % lang, reuse=tf.AUTO_REUSE):
            d_output = self.dec_stack[lang](d_emb, e_output, d_attn_bias, e_attn_bias)

        return d_output

    def _shared_project_one(self, d_output, lang):
        outputs = tf.tensordot(d_output, self.decoder_embeddings[lang], axes=[[2], [1]])
        outputs = tf.nn.bias_add(outputs, self.pro_stack[lang])
        return outputs

    def _projection_one(self, d_output, lang):
        """
        Args:
            d_output:
            lang:
        Return:
        """
        with tf.variable_scope("output_projection", reuse=tf.AUTO_REUSE):
            if self.params["share_decpro_emb"]:
                return self._shared_project_one(d_output, lang)
            else:
                return self.pro_stack[lang](d_output)

    def _get_symbols_to_logits_fn(self, max_decode_length, lang):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = common_layers.get_timing_signal_1d(
            max_decode_length + 1, self.params["hidden_size"])
        decoder_self_attention_bias = common_layers.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
            Args:
              ids: Current decoded sequences.
                int tensor with shape [batch_size * beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.
            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = tf.gather(self.decoder_embeddings[lang], decoder_input)
            decoder_input *= self.params["hidden_size"] ** 0.5
            decoder_input += timing_signal[:, i:i + 1]

            if self.is_training:
                decoder_input = tf.nn.dropout(decoder_input, 1.0 - self.params["prepostprocess_dropout"])

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            with tf.variable_scope("shared_decode", reuse=tf.AUTO_REUSE):
                decoder_outputs = self.share_dec_stack(decoder_input,
                                                       cache.get("e_output"),
                                                       self_attention_bias,
                                                       cache.get("e_attn_bias"),
                                                       cache)
            with tf.variable_scope("%s_decode" % lang, reuse=tf.AUTO_REUSE):
                decoder_outputs = self.dec_stack[lang](decoder_outputs,
                                                       cache.get("e_output"),
                                                       self_attention_bias,
                                                       cache.get("e_attn_bias"),
                                                       cache)
            logits = self._projection_one(decoder_outputs, lang)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def greedy_predict(self, e_input, lang1, lang2):
        """greedy decoding for translation model lang1 -> lang2
        Args:
            e_input: int tensor, [batch_size, input_length].
            lang1:
            lang2:
        Return:
        """
        e_output, e_attn_bias = self._encode_one(e_input, lang1)
        e_output_shape = common_layers.shape_list(e_output)
        batch_size = e_output_shape[0]
        input_length = e_output_shape[1]
        max_decode_length = tf.cast(1.5 * tf.to_float(input_length), tf.int32)

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length, lang2)

        # Create cache storing decoder attention values for each layer.
        cache = {}
        for n in range(self.params["share_dec"]):
            cache["share_layer_%d" % n] = {
                "k": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, self.params["hidden_size"]])}

        for n in range(self.params["n_dec_layers"] - self.params["share_dec"]):
            cache["layer_%d" % n] = {
                "k": tf.zeros([batch_size, 0, self.params["hidden_size"]]),
                "v": tf.zeros([batch_size, 0, self.params["hidden_size"]])}

        # Add encoder output and attention bias to the cache.
        cache["e_output"] = e_output
        cache["e_attn_bias"] = e_attn_bias

        def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
            """One step of greedy decoding."""
            logits, cache = symbols_to_logits_fn(next_id, i, cache)
            log_probs = common_layers.log_prob_from_logits(logits)
            temperature = 0.0 if self.params["argmax"] else self.params["sampling_temp"]
            next_id = common_layers.sample_with_temperature(logits, temperature)
            hit_eos |= tf.equal(next_id, self.eos_id)

            log_prob_indices = tf.stack(
                [tf.range(tf.to_int64(batch_size)), next_id], axis=1)
            log_prob += tf.gather_nd(log_probs, log_prob_indices)

            next_id = tf.expand_dims(next_id, axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

        def is_not_finished(i, hit_eos, *_):
            finished = i >= max_decode_length
            finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
        hit_eos = tf.fill([batch_size], False)
        # initialize the next_id
        next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
        initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)

        def get_state_shape_invariants(tensor):
            """Returns the shape of the tensor but sets middle dims to None."""
            shape = tensor.shape.as_list()
            for i in range(1, len(shape) - 1):
                shape[i] = None
            return tf.TensorShape(shape)

        _, _, _, decoded_ids, _, log_prob = tf.while_loop(
            is_not_finished,
            inner_loop, [
                tf.constant(0), hit_eos, next_id, decoded_ids, cache,
                initial_log_prob
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                nest.map_structure(get_state_shape_invariants, cache),
                tf.TensorShape([None]),
            ])
        scores = log_prob

        return decoded_ids, scores


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params, is_training):
        self.layer = layer
        self.postprocess_dropout = params["prepostprocess_dropout"]
        self.is_training = is_training

        # Create normalization layer
        self.layer_norm = LayerNormalization(params["hidden_size"])

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.is_training:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


class TransformerEncoderLayer(tf.layers.Layer):

    def __init__(self, params, is_training):
        super(TransformerEncoderLayer, self).__init__()

        _self_attention_layer = common_attention.SelfAttention(
            params["hidden_size"], params["num_heads"],
            params["attention_dropout"], is_training)
        self.self_attention_layer = PrePostProcessingWrapper(
            _self_attention_layer, params, is_training)

        _feed_forward_network = common_attention.FeedFowardNetwork(
            params["hidden_size"], params["filter_size"],
            params["relu_dropout"], is_training, params["allow_ffn_pad"])
        self.feed_forward_network = PrePostProcessingWrapper(
            _feed_forward_network, params, is_training)

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        with tf.variable_scope("self_attention"):
            encoder_inputs = self.self_attention_layer(encoder_inputs, attention_bias)
        with tf.variable_scope("ffn"):
            encoder_inputs = self.feed_forward_network(encoder_inputs, inputs_padding)

        return encoder_inputs


class SharedEncoderStack(tf.layers.Layer):

    def __init__(self, params, is_training):
        super(SharedEncoderStack, self).__init__()
        self.layers = []
        for _ in range(params["share_enc"]):
            self.layers.append(TransformerEncoderLayer(params, is_training))

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        for n, layer in enumerate(self.layers):
            with tf.variable_scope("share_layer_%d" % n):
                encoder_inputs = layer(encoder_inputs, attention_bias, inputs_padding)

        return encoder_inputs


class EncoderStack(tf.layers.Layer):

    def __init__(self, params, is_training):
        super(EncoderStack, self).__init__()
        self.layers = []

        for _ in range(params["n_enc_layers"] - params["share_enc"]):
            self.layers.append(TransformerEncoderLayer(params, is_training))

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        for n, layer in enumerate(self.layers):
            with tf.variable_scope("layer_%d" % n):
                encoder_inputs = layer(encoder_inputs, attention_bias, inputs_padding)

        return self.output_normalization(encoder_inputs)


class TransformerDecoderLayer(tf.layers.Layer):

    def __init__(self, params, is_training):
        super(TransformerDecoderLayer, self).__init__()

        _self_attention_layer = common_attention.SelfAttention(
            params["hidden_size"], params["num_heads"],
            params["attention_dropout"], is_training)
        self.self_attention_layer = PrePostProcessingWrapper(
            _self_attention_layer, params, is_training)

        _enc_dec_attention_layer = common_attention.Attention(
            params["hidden_size"], params["num_heads"],
            params["attention_dropout"], is_training)
        self.enc_dec_attention_layer = PrePostProcessingWrapper(
            _enc_dec_attention_layer, params, is_training)

        _feed_forward_network = common_attention.FeedFowardNetwork(
            params["hidden_size"], params["filter_size"],
            params["relu_dropout"], is_training, params["allow_ffn_pad"])
        self.feed_forward_network = PrePostProcessingWrapper(
            _feed_forward_network, params, is_training)

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, cache=None):
        with tf.variable_scope("self_attention"):
            decoder_inputs = self.self_attention_layer(
                decoder_inputs, decoder_self_attention_bias, cache=cache)
        with tf.variable_scope("encdec_attention"):
            decoder_inputs = self.enc_dec_attention_layer(
                decoder_inputs, encoder_outputs, attention_bias)
        with tf.variable_scope("ffn"):
            decoder_inputs = self.feed_forward_network(decoder_inputs)

        return decoder_inputs


class SharedDecoderStack(tf.layers.Layer):

    def __init__(self, params, is_training):
        super(SharedDecoderStack, self).__init__()
        self.layers = []

        for _ in range(params["share_dec"]):
            self.layers.append(TransformerDecoderLayer(params, is_training))

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, cache=None):
        for n, layer in enumerate(self.layers):
            layer_name = "share_layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                decoder_inputs = layer(
                    decoder_inputs,
                    encoder_outputs,
                    decoder_self_attention_bias,
                    attention_bias,
                    cache=layer_cache)

        return decoder_inputs


class DecoderStack(tf.layers.Layer):

    def __init__(self, params, is_training):
        super(DecoderStack, self).__init__()
        self.layers = []

        for _ in range(params["n_dec_layers"] - params["share_dec"]):
            self.layers.append(TransformerDecoderLayer(params, is_training))

        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, cache=None):
        for n, layer in enumerate(self.layers):
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                decoder_inputs = layer(
                    decoder_inputs,
                    encoder_outputs,
                    decoder_self_attention_bias,
                    attention_bias,
                    cache=layer_cache)

        return self.output_normalization(decoder_inputs)



