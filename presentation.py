import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

d = np.load('data/points.npz')
points = d['points']
labels = d['labels']

markers = ['o', '+', 'v', '^']
colors = ['w', 'r', 'w', 'w']
edgecolors = ['b', 'w', 'g', 'k']

label = [3, 9, 20, 21]

# points = points.T
# for i in range(4):
#     plt.scatter(points[0][labels == label[i]],
#                 points[1][labels == label[i]],
#                 marker=markers[i],
#                 c=colors[i],
#                 edgecolors=edgecolors[i])
#
# plt.show()

centers = []
amount = []
for i in range(46):
    centers.append(np.mean(points[labels == i], axis=0))
    amount.append(np.sum(labels == i))
amount = np.array(amount)

centers = np.array(centers)
D = pdist(centers)
D = squareform(D)

# delete_num = 26
for i in range(len(D)):
    D[i, i] = 100

foo = np.arange(46)
# for i in range(delete_num):
#     a = np.unravel_index(np.argmin(D, axis=None), D.shape)
#     index = a[amount[a[0]] > amount[a[1]]]
#     D = np.delete(D, index, axis=0)
#     D = np.delete(D, index, axis=1)
#     foo = np.delete(foo, index)

while True:
    less_5 = np.sum(D < 5)
    if less_5 <= 20:
        break

    indices = np.where(D < 5)
    a = [indices[0][0], indices[1][0]]
    index = a[amount[a[0]] > amount[a[1]]]
    D = np.delete(D, index, axis=0)
    D = np.delete(D, index, axis=1)
    foo = np.delete(foo, index)

for i in range(len(D)):
    D[i, i] = 0

D = squareform(D)
hist, bins = np.histogram(D, 100)
cdf = np.cumsum(hist) / len(D)

x = bins[:100] + (bins[1] - bins[0])/2
plt.plot(x, cdf)
plt.show()

# 统计一个category中有60%, 70%, 80%, 90%, 100%的点落进的半径
radius = np.zeros((5, len(foo)))
percentage = [0.6, 0.7, 0.8, 0.9, 1]
for i in range(len(foo)):
    points_in_cate = points[labels == foo[i]]
    center = np.mean(points_in_cate, axis=0)
    dist_sort = np.sort(np.linalg.norm(points_in_cate - center, axis=1))

    for j in range(5):
        index = math.ceil(len(points_in_cate) * percentage[j]) - 1
        radius[j][i] = dist_sort[index]

colors = ['r', 'g', 'dodgerblue', 'b', 'k']
indice = np.arange(46) + 1
for i in range(5):
    plt.plot(foo, radius[i], c=colors[i])

plt.show()










