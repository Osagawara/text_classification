import os
import sys
import numpy as np
from tqdm import trange
from keras.datasets import reuters
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from square_ovlp_detection import Square
from overlap_area import area

class TF_IDF():

    def __init__(self, sentences: np.ndarray):
        self.sentences = sentences

    def vectorize(self):
        word_num = np.max([np.amax(l) for l in self.sentences]) + 1
        self.word_freq = np.zeros((len(self.sentences), word_num))

        for i in trange(len(self.sentences)):
            for word_index in self.sentences[i]:
                self.word_freq[i][word_index] += 1

    def transform(self):
        if os.path.exists('data/tf-idf.npy'):
            self.tf_idf = np.load('data/tf-idf.npy')
        else:
            transformer = TfidfTransformer()
            self.tf_idf = transformer.fit_transform(self.word_freq)
            self.tf_idf = csr_matrix(self.tf_idf).toarray()
            np.save('data/tf-idf.npy', self.tf_idf)

    def average_dissimilarity(self, subset=np.array([])):
        if os.path.exists('data/cos_dist.npy'):
            dist = np.load('data/cos_dist.npy')
        else:
            dist = pdist(self.tf_idf, 'cosine')
            np.save('data/cos_dist.npy', dist)

        if len(subset) == 0:
            aver_dist = np.mean(dist)
        else:
            dist = squareform(dist)
            foo = dist[subset, :][:, subset]
            aver_dist = np.sum(foo) / (len(subset) * (len(subset) - 1))

        return aver_dist

    def min_dissimilarity(self, subset=np.array([])):
        if os.path.exists('data/cos_dist.npy'):
            dist = np.load('data/cos_dist.npy')
        else:
            dist = pdist(self.tf_idf, 'cosine')
            np.save('data/cos_dist.npy', dist)


        if len(subset) == 0:
            min_dist = np.min(dist)
        else:
            dist = squareform(dist)
            foo = dist[subset, :][:, subset]
            foo[foo == 0] = 2
            min_dist = np.mean(np.min(foo, axis=1))

        return min_dist




if __name__ == '__main__':
    num_words = 5000
    maxlen = 400

    (x_train, y_train), (x_test, y_test) = reuters.load_data('reuters.npz', num_words=num_words, maxlen=maxlen,
                                                             seed=np.random.randint(10000))
    sentences = np.concatenate([x_train, x_test])
    tfidf_object = TF_IDF(sentences)

    tfidf_object.transform()


    points = np.load('data/points.npz')['points']
    num = len(points)
    r = 0.4
    dist_data = []
    area_data = []

    iteration = 300
    for i in trange(iteration):
        up, down = 3500, 4500
        foo = np.random.randint(up, down)
        subset = np.random.permutation(len(sentences))[:foo]
        min_dist = tfidf_object.min_dissimilarity(subset)
        s = [Square(y+r, y-r, x-r, x+r) for x, y in points[subset]]
        aver_area = area(s) / foo
        dist_data.append(min_dist)
        area_data.append(aver_area)

    np.savez('data/dist_area_{ }_{}.npz'.format(up, down), dist_data=dist_data, area_data=area_data)





