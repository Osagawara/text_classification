import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from overlap_area import area
from square_ovlp_detection import Square

d = np.load('data/points.npz')
points = d['points']
labels = d['labels']

# test the proper value of little r

radius = np.arange(0.01, 1, 0.005)
num = len(points)

selected_points = np.random.permutation(points)[:num // 10]
areas = []

pbar = tqdm(total=len(radius))
for r in radius:
    squares = [ Square(y+r, y-r, x-r, x+r) for x, y in selected_points ]
    areas.append(area(squares))
    pbar.update()
pbar.close()

plt.plot(radius, areas, 'bo')
plt.show()

# num = len(points)
