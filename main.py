# coding=utf-8

import numpy as np
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import reuters
from keras.preprocessing import sequence

from rcnn import RCNN

num_words = 5000
num_class = 46
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data('reuters.npz', num_words=num_words, maxlen=maxlen)
(x_train, x_valid) = np.split(x_train, [8000])
(y_train, y_valid) = np.split(y_train, [8000])
print(len(x_train), 'train sequences')
print(len(x_valid), 'valid sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)...')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_valid = sequence.pad_sequences(x_valid, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_valid shape:', x_valid.shape)
print('x_test shape:', x_test.shape)

print('Prepare input for model...')
x_train_current = x_train
x_train_left = np.hstack([np.expand_dims(x_train[:, 0], axis=1), x_train[:, 0:-1]])
x_train_right = np.hstack([x_train[:, 1:], np.expand_dims(x_train[:, -1], axis=1)])

x_valid_current = x_valid
x_valid_left = np.hstack([np.expand_dims(x_valid[:, 0], axis=1), x_valid[:, 0:-1]])
x_valid_right = np.hstack([x_valid[:, 1:], np.expand_dims(x_valid[:, -1], axis=1)])

x_test_current = x_test
x_test_left = np.hstack([np.expand_dims(x_test[:, 0], axis=1), x_test[:, 0:-1]])
x_test_right = np.hstack([x_test[:, 1:], np.expand_dims(x_test[:, -1], axis=1)])

y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=num_class)
y_valid_one_hot = keras.utils.to_categorical(y_valid, num_classes=num_class)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=num_class)
print('x_train_current shape:', x_train_current.shape)
print('x_train_left shape:', x_train_left.shape)
print('x_train_right shape:', x_train_right.shape)
print('x_test_current shape:', x_test_current.shape)
print('x_test_left shape:', x_test_left.shape)
print('x_test_right shape:', x_test_right.shape)

print('Build model...')
model = RCNN(maxlen, num_words, embedding_dims, class_num=num_class, last_activation='softmax').get_model()
optimizer = Adam(lr=0.01, decay=0.009)
model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
model.load_weights('model/rcnn/rcnn.h5')

print('Train...')
# early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
tensor_board = TensorBoard(update_freq='batch')
model.fit([x_train_current, x_train_left, x_train_right], y_train_one_hot,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[tensor_board],
          validation_data=([x_valid_current, x_valid_left, x_valid_right], y_valid_one_hot))

print('Test...')
result = model.predict([x_test_current, x_test_left, x_test_right])
print(result[:100])

model.save_weights('model/rcnn/rcnn.h5')
