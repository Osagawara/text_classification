# coding=utf-8

from keras.layers import Input
from keras.models import Model
from keras.activations import softmax
from keras import backend as K
from keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D, GlobalMaxPooling1D


class RCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=1,
                 last_activation='sigmoid'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

        self.input_current = None
        self.input_left = None
        self.input_right = None

        self.last_layer = None
        self.prediction = None

        self.construct_rcnn()

    def construct_rcnn(self):
        self.input_current = Input((self.maxlen,))
        self.input_left = Input((self.maxlen,))
        self.input_right = Input((self.maxlen,))


        embedder = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        embedding_current = embedder(self.input_current)
        embedding_left = embedder(self.input_left)
        embedding_right = embedder(self.input_right)

        x_left = SimpleRNN(128, return_sequences=True)(embedding_left)
        x_right = SimpleRNN(128, return_sequences=True, go_backwards=True)(embedding_right)
        x_right = Lambda(lambda x: K.reverse(x, axes=1))(x_right)
        x = Concatenate(axis=2)([x_left, embedding_current, x_right])

        x = Conv1D(64, kernel_size=1, activation='tanh')(x)
        x = GlobalMaxPooling1D()(x)

        self.last_layer = Dense(self.class_num)(x)

        self.prediction = Lambda(lambda t: softmax(t))(self.last_layer)

    def get_model(self):
        model = Model(inputs=[self.input_current, self.input_left, self.input_right], outputs=self.prediction)
        return model

    def get_extractor(self):
        model = Model(inputs=[self.input_current, self.input_left, self.input_right], outputs=self.last_layer)
        return model
