import numpy as np
import tensorflow as tf
import keras
from keras.datasets import reuters

num_words = 5000
maxlen = 400
batch_size = 32

print(np.__version__)
print(keras.__version__)

(x_train, y_train), (x_test, y_test) = reuters.load_data('reuters.npz', num_words=num_words, maxlen=maxlen)
print('{} train sequences'.format(len(x_train)))
print('{} test sequences'.format(len(x_test)))

print()