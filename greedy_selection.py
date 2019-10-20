import sys
import pickle
import numpy as np
from tqdm import tqdm, trange
from evolutionary import match_degree
from square_ovlp_detection import Square
from overlap_area import area

class Greedy:
    def __init__(self, points: np.ndarray, centers: np.ndarray, r: float, R: float, budget: float, bid: np.ndarray,
                 weight: np.ndarray, kind: str):
        self.points = points
        self.centers = centers
        self.r = r
        self.R = R
        self.budget = budget
        self.bid = bid
        self.w = weight
        self.kind = kind

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

    def set_weight(self, w: np.ndarray):
        assert len(w) == 3, "w's length is not 3"
        self.w = w

    def set_bid(self, bid: np.ndarray):
        self.bid = bid

    def marginal_contribution(self, i):
        if self.kind == 'matching_degree':
            return self.matching_degree_proportion[i]
        else:
            this_area = area([self.squares[j] for j in self.selected + [i]])
            if self.kind == 'diversity':
                return this_area - self.covered_area
            else :
                return self.w[0] * 1 / self.num + self.w[1] * self.matching_degree_proportion[i] + \
                       self.w[2] * (this_area - self.covered_area) / self.total_area

    def total(self, max_i):
        temp_selected = self.selected + [max_i]
        if self.kind == 'matching_degree':
            return np.sum(self.matching_degree_proportion[np.array(temp_selected)])
        else:
            this_area = area([self.squares[_] for _ in temp_selected])
            if self.kind == 'diversity':
                return this_area
            else:
                return  self.w[0] * ( len(self.selected) + 1 ) / num + \
                        self.w[1] * np.sum(self.matching_degree_proportion[np.array(temp_selected)]) + \
                        self.w[2] * this_area / self.total_area

    def quantity_driven(self):
        sorted_bid = np.sort(self.bid)
        sorted_bid_index = np.argsort(self.bid)
        foo = self.budget / np.arange(1, len(sorted_bid)+1) >= sorted_bid
        self.selected = [sorted_bid_index[_] for _ in range(len(sorted_bid)) if foo[_]]
        print(np.sum(self.bid[np.array(self.selected)]))


    def one_round(self):
        for i in self.unselected:
            foo = self.marginal_contribution(i)
            self.mcpc[i] = foo / self.bid[i]

        max_i = np.argmax(self.mcpc)
        self.total_value = self.total(max_i)

        return max_i

    def greedy_selection(self):
        '''

        :param w: w is a vector of three elements, representing the weight of three measurement
        :return:
        '''
        max_mcpc_index = np.argmin(self.bid)
        self.mcpc[max_mcpc_index] = self.marginal_contribution(max_mcpc_index)
        self.covered_area = 4 * self.r**2
        self.total_value = self.total(max_mcpc_index)

        pbar = tqdm(total=self.num)
        while len(self.unselected) > 0 and self.mcpc[max_mcpc_index] >= 2 * self.total_value / self.budget:
            pbar.update()
            self.selected.append(max_mcpc_index)
            self.unselected.remove(max_mcpc_index)
            self.mcpc[max_mcpc_index] = -1
            self.covered_area = area([self.squares[_] for _ in self.selected])
            max_mcpc_index = self.one_round()
        pbar.close()

    def clear(self):
        self.selected = []
        self.unselected = [_ for _ in range(self.num)]
        self.covered_area = 0
        self.mcpc = np.zeros(self.num)



if __name__ == '__main__':
    kind = 'diversity'

    d = np.load('data/points.npz')
    points = d['points']
    labels = d['labels']
    centers = np.load('data/centers.npy')
    r = 0.4
    R = 5
    budget = 20000
    num = 3000
    bid = np.random.uniform(0, 1, num) * 3 + 3
    weight = np.random.rand(3)
    weight = weight / np.sum(weight)
    population = 1

    X = np.random.permutation(np.arange(len(points)))
    X = X[:num]
    greedy_object = Greedy(points[X], centers[labels][X], r, R, budget, bid, weight, kind)


    # print information
    print("kind = {}, budget = {}, U[3, 6]".format(kind, budget))
    result = []
    for i in range(population):
        greedy_object.quantity_driven()

        m = np.sum(greedy_object.matching_degree_all[np.array(greedy_object.selected)])
        a = area([greedy_object.squares[i] for i in greedy_object.selected])
        result.append(np.array([len(greedy_object.selected), m, a]))

        greedy_object.clear()
        bid = np.random.uniform(0, 1, num) * 3 + 3
        greedy_object.set_bid(bid)
        weight = np.random.rand(3)
        weight = weight / np.sum(weight)
        greedy_object.set_weight(weight)


    result = np.vstack(tuple(result))
    np.save('data/{}_{}_uniform_result.npy'.format(kind, budget), result)


