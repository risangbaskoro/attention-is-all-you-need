import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(d_ff)
        self.fc2 = tf.keras.layers.Dense(d_model)
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs, *args, **kwargs):
        x = self.fc1(inputs)
        x = self.activation(x)
        return self.fc2(x)


class AddNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, inputs: list, *args, **kwargs):
        return self.layernorm(inputs[0], inputs[1])
