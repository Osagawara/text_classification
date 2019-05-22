import numpy as np
from keras.datasets import reuters
from keras.preprocessing import sequence
from rcnn import RCNN


num_words = 5000
num_class = 46
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

(x_train, y_train), (x_test, y_test) = reuters.load_data('reuters.npz', num_words=num_words, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_test_current = x_test
x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_test[:, 0:-1]])
x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])

model = RCNN(maxlen, num_words, embedding_dims, class_num=num_class, last_activation='softmax').get_model()
model.load_weights('model/rcnn/rcnn.h5')
results = model.predict([x_test_current, x_test_left, x_test_right])

results = np.argmax(results, axis=1)
accuracy = np.sum(results == y_test) / len(y_test)
print('accuracy: {}'.format(accuracy))