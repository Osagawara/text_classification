import numpy as np
from keras.datasets import reuters

num_words = 5000
maxlen = 400
batch_size = 32


(x_train, y_train), (x_test, y_test) = reuters.load_data('reuters.npz', num_words=num_words, maxlen=maxlen)
print('{} train sequences'.format(len(x_train)))
print('{} test sequences'.format(len(x_test)))
