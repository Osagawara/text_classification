import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.preprocessing import sequence
from rcnn import RCNN

num_words = 5000
num_class = 46
maxlen = 400
embedding_dims = 50

(x_train, y_train), (x_test, y_test) = reuters.load_data('reuters.npz',
                                                         num_words=num_words,
                                                         maxlen=maxlen)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

data = np.concatenate([x_train, x_test])
labels = np.concatenate([y_train, y_test])
data_current = data
data_left = np.hstack([np.expand_dims(data[:, 0], axis=1), data[:, 0:-1]])
data_right = np.hstack([data[:, 1:], np.expand_dims(data[:, -1], axis=1)])

rcnn = RCNN(maxlen, num_words, embedding_dims, class_num=num_class, last_activation='softmax')
extractor = rcnn.get_extractor()
extractor.load_weights('model/rcnn/rcnn.h5')

features = extractor.predict([data_current, data_left, data_right])

np.savez('data/features.npz', features=features, labels=labels)
