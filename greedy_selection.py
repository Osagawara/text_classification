import sys
import pickle
import numpy as np
from tqdm import tqdm, trange
from evolutionary import match_degree
from square_ovlp_detection import Square
from overlap_area import area

class Greedy:
    def __init__(self, points: np.ndarray, centers: np.ndarray, r: float, R: float, budget: float, bid: np.ndarray):
        self.points = points
        self.centers = centers
        self.r = r
        self.R = R
        self.budget = budget
        self.bid = bid

        self.squares = [Square(y + r, y - r, x - r, x + r) for x, y in points]

        self.num = len(points)
        self.matching_degree_all = match_degree(points, centers, R)
        self.matching_degree_proportion = self.matching_degree_all / np.sum(self.matching_degree_all)
        self.matching_degree_proportion[self.matching_degree_proportion < 0] = 0
        self.selected = []
        self.unselected = [i for i in range(len(points))]

        self.covered_area = 0
        self.total_area = area(self.squares)
        # marginal contribution per cost
        self.mcpc = np.zeros(self.num)

    def one_round(self, w):
        for i in self.unselected:
            this_area = area([self.squares[j] for j in self.selected + [i]])
            foo = w[0] * 1 / self.num + w[1] * self.matching_degree_proportion[i] + \
                  w[2] * (this_area - self.covered_area) / self.total_area
            self.mcpc[i] = foo / self.bid[i]

        max_i = np.argmax(self.mcpc)

        self.total_value = w[0] * ( len(self.selected) + 1 ) / num + \
                           w[1] * np.sum(self.matching_degree_proportion[np.array(self.selected + [max_i])]) + \
                           w[2] * area([self.squares[j] for j in self.selected + [max_i]]) / self.total_area

        return max_i

    def greedy_selection(self, w):
        '''

        :param w: w is a vector of three elements, representing the weight of three measurement
        :return:
        '''
        assert len(w) == 3, "w's length is not 3"
        max_mcpc_index = np.argmin(self.bid)
        self.mcpc[max_mcpc_index] = 4 * self.r**2 / self.bid[max_mcpc_index]
        self.covered_area = 4 * self.r**2
        self.total_value = w[0] * 1 / self.num + w[1] * self.matching_degree_proportion[max_mcpc_index] + \
                           w[2] * self.covered_area / self.total_area

        while len(self.unselected) > 0 and self.mcpc[max_mcpc_index] >= 2 * self.total_value / self.budget:
            self.selected.append(max_mcpc_index)
            self.unselected.remove(max_mcpc_index)
            self.mcpc[max_mcpc_index] = -1
            self.covered_area = area([self.squares[_] for _ in self.selected])
            max_mcpc_index = self.one_round(w)

    def clear(self):
        self.selected = []
        self.unselected = [_ for _ in range(self.num)]
        self.covered_area = 0
        self.mcpc = np.zeros(self.num)



if __name__ == '__main__':
    d = np.load('data/points.npz')
    points = d['points']
    labels = d['labels']
    centers = np.load('data/centers.npy')
    r = 0.4
    R = 5
    budget = 2000
    num = 1000
    bid = np.random.uniform(0, 1, num) * 2 + 2
    population = 100

    X = np.random.permutation(np.arange(len(points)))
    X = X[:num]
    greedy_object = Greedy(points[X], centers[labels][X], r, R, budget, bid)

    result = []
    for i in trange(population):
        w = np.random.rand(3)
        w = w / np.sum(w)
        greedy_object.greedy_selection(w)

        m = np.sum(greedy_object.matching_degree_all[np.array(greedy_object.selected)])
        a = area([greedy_object.squares[i] for i in greedy_object.selected])
        result.append(np.array([len(greedy_object.selected), m, a]))
        greedy_object.clear()

    result = np.vstack(tuple(result))
    np.save('data/greedy_result.npy', result)

