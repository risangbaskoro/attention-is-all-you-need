import tensorflow as tf
import position_embedding
import utils


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_k, d_v, d_model, d_ff, dropout_rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, d_k, d_v, d_model)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.addnorm1 = utils.AddNormalization()
        self.ff = utils.FeedForward(d_ff, d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.addnorm2 = utils.AddNormalization()

    def call(self, inputs, *args, **kwargs):
        padding_mask = kwargs.pop('padding_mask')

        mha_out = self.mha(inputs, inputs, inputs, padding_mask)
        mha_out = self.dropout1(mha_out)

        addnorm_out = self.addnorm1([inputs, mha_out])

        ff_out = self.ff(addnorm_out)
        ff_out = self.dropout2(ff_out)

        return self.addnorm2([mha_out, ff_out])


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, sequence_length, num_heads, d_k, d_v, d_model, d_ff, n=6, dropout_rate=0.2, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = position_embedding.PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_layer = [EncoderLayer(num_heads, d_k, d_v, d_model, d_ff, dropout_rate) for _ in range(n)]

    def call(self, inputs, **kwargs):
        padding_mask = kwargs.pop('padding_mask')
        training = kwargs.pop('training')

        pos_encoding_out = self.pos_encoding(inputs)
        x = self.dropout(pos_encoding_out, training)

        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x