# 将autoencoder抽取的二维数据, 展示出来

import numpy as np
from autoencoder import auto_encoder

feature_len = 46
inner = [16, 8, 2, 8, 16]
batch_size = 32

d = np.load('data/features.npz')
features = d['features']
labels = d['labels']

auto = auto_encoder(feature_len, inner)
dim_reducer = auto.get_dim_reducer()

dim_reducer.load_weights('model/autoencoder/auto.h5', by_name=True)
points = dim_reducer.predict(features, batch_size)
np.savez('data/points.npz', points=points, labels=labels)