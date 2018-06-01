# ! -*- coding: utf-8 -*-
from keras.engine.topology import Layer
import tensorflow as tf
import keras.backend as K

class LayerDropout(Layer):
    def __init__(self, dropout = 0.0, residual_add = False, **kwargs):
        self.dropout = dropout
        self.residual_add = residual_add
        super(LayerDropout, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LayerDropout, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        x, residual = x
        x_train = tf.nn.dropout(x, 1.0 - self.dropout)
        x_test = x
        if self.residual_add:
            x_train += residual
            x_test += residual
        pred = tf.random_uniform([]) < self.dropout
        x_train = tf.cond(pred, lambda: residual, lambda: x_train)
        return K.in_train_phase(x_train, x_test, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape