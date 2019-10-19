import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

d = np.load('data/points.npz')
points = d['points']
labels = d['labels']
num = len(labels)
num_category = np.max(labels).astype(np.int) + 1

markers = ['o', '+', 'v', '^']
colors = ['w', 'r', 'w', 'w']
edgecolors = ['b', 'w', 'g', 'k']


# draw the diagram of the average distance of different categories
sorted_label = np.sort(labels)
sorted_label_index = np.argsort(labels)

split = [0] + [i for i in range(1, num) if sorted_label[i-1] != sorted_label[i]] + [num]

sample_time = 100
D = np.zeros( num_category * (num_category - 1) // 2 )

for i in range(sample_time):
    sample_sorted_label_index = [np.random.randint(split[_], split[_+1]) for _ in range(len(split)-1)]
    sample_data_index = sorted_label_index[np.array(sample_sorted_label_index)]
    D += pdist(points[sample_data_index])

D /= sample_time

hist, bins = np.histogram(D, 100)
hist = np.cumsum(hist) / np.sum(hist)
plt.plot(bins[:-1], hist, color='r', lw=2.5)
plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')
plt.xlabel('$x$: Average distance of categories', fontsize='xx-large')
plt.ylabel('the percentage of $\{Aver\ dist \leq x\}$', fontsize='xx-large')
plt.show()

# draw the distribution of every data to the centroid in different categories
# percentage is 60%, 70%, 80%, 90%
radius = np.zeros((5, num_category))
percentage = [60, 70, 80, 90, 100]

for i in range(num_category):
    points_this_cat = points[labels == i]
    center = np.mean(points_this_cat, axis=0)
    dist_sort = np.sort(np.linalg.norm(points_this_cat - center, axis=1))
    for j in range(5):
        index = math.ceil(len(points_this_cat) * percentage[j] / 100) - 1
        radius[j][i] = dist_sort[index]

colors = ['r', 'g', 'dodgerblue', 'b', 'k']
indice = np.arange(46) + 1
for i in range(5):
    plt.plot(indice, radius[i], c=colors[i], linewidth=1.5, label='{}%'.format(percentage[i]))

plt.xticks(fontsize='x-large')
plt.yticks(fontsize='x-large')
plt.xlabel('category number', fontsize='xx-large')
plt.ylabel('the radius containing the percentage of points', fontsize='x-large')
plt.legend(fontsize='x-large')
plt.show()











