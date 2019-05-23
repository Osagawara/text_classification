'''
将抽取出的最后一层特征用来训练autoencoder
可以用来降维, 得到二维特征
'''

import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense


class auto_encoder(object):

    def __init__(self, feature_len, inner):
        '''
        初始化参数
        :param feature_len: 输入层feature的长度
        :param inner: 一个list, 隐藏层每一层的节点数
        '''
        self.feature_len = feature_len
        self.inner = inner
        self.input = None
        self.two_dim = None
        self.output = None
        self.construct()

    def construct(self):
        '''
        构建自动编码机的每一层
        :return: None
        '''
        self.input = Input(shape=(self.feature_len, ))
        x = self.input
        for i in self.inner:
            x = Dense(units=i)(x)
            if i == 2:
                self.two_dim = x

        self.output = Dense(units=self.feature_len)(x)

    def get_auto(self):
        model = Model(inputs=[self.input], outputs=[self.output])
        return model

    def get_dim_reductor(self):
        model = Model(inputs=[self.input], outputs=[self.two_dim])
        return model


if __name__ == '__main__':
    feature_len = 46
    inner = [16, 8, 2, 8, 16]
    learning_rate = 0.001
    decay = 0.001
    batch_size = 32
    epochs = 10

    auto = auto_encoder(feature_len, inner)
    trainer = auto.get_auto()
    optimizer = Adam(lr=learning_rate, decay=decay)
    trainer.compile(optimizer, loss='binary_crossentropy')

