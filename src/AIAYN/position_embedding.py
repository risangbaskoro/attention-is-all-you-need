import numpy as np
import tensorflow as tf


class PositionEmbeddingFixedWeights(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)

        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)
        self.word_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False,
        )
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False,
        )

    @staticmethod
    def get_position_encoding(sequence_length, d, n=10000):
        pos = np.zeros((sequence_length, d))
        for k in range(sequence_length):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                pos[k, 2 * i] = np.sin(k / denominator)
                pos[k, 2 * i + 1] = np.cos(k / denominator)
        return pos

    def call(self, inputs, *args, **kwargs):
        pos_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding(inputs)
        embedded_indices = self.position_embedding(pos_indices)
        return embedded_words + embedded_indices
