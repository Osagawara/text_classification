import numpy as np
from tqdm import tqdm
from evolutionary import match_degree
from square_ovlp_detection import Square
from overlap_area import area

class Greedy:
    def __init__(self, points: np.ndarray, centers: np.ndarray, r: float, budget: float, bid: np.ndarray):
        self.points = points
        self.centers = centers
        self.r = r
        self.budget = budget
        self.bid = bid

        self.squares = [Square(y + r, y - r, x - r, x + r) for x, y in points]

        self.num = len(points)
        matching_degree_all = match_degree(points, centers, r)
        self.matching_degree_proportion = matching_degree_all / np.sum(matching_degree_all)
        self.matching_degree_proportion[self.matching_degree_proportion < 0] = 0
        self.selected = []
        self.unselected = [i for i in range(len(points))]

        self.covered_area = 0
        self.total_area = area(self.squares)
        # marginal contribution per cost
        self.mcpc = np.zeros(self.num)

    def one_round(self):
        for i in self.unselected:
            this_area = area([self.squares[j] for j in self.selected + [i]])
            foo = 1 / self.num + self.matching_degree_proportion[i] + (this_area - self.covered_area) / self.total_area
            self.mcpc[i] = foo / self.bid[i]

        max_i = np.argmax(self.mcpc)

        self.total_value = ( len(self.selected) + 1 ) / num + \
                           np.sum(self.matching_degree_proportion[np.array(self.selected + [max_i])]) + \
                           area([self.squares[j] for j in self.selected + [max_i]]) / self.total_area

        return max_i

    def greedy_selection(self):
        max_mcpc_index = np.argmin(self.bid)
        self.mcpc[max_mcpc_index] = 4 * self.r**2 / self.bid[max_mcpc_index]
        self.covered_area = 4 * self.r**2
        self.total_value = 1 / self.num + self.matching_degree_proportion[max_mcpc_index] + \
                           self.covered_area / self.total_area

        pbar = tqdm(total=self.num)
        while len(self.unselected) > 0 and self.mcpc[max_mcpc_index] >= 2 * self.total_value / self.budget:
            pbar.update()
            self.selected.append(max_mcpc_index)
            self.unselected.remove(max_mcpc_index)

            self.mcpc[max_mcpc_index] = -1
            self.covered_area = area([self.squares[_] for _ in self.selected])
            max_mcpc_index = self.one_round()
        pbar.close()





if __name__ == '__main__':
    d = np.load('data/points.npz')
    points = d['points']
    labels = d['labels']
    centers = np.load('data/centers.npy')
    r = 0.4
    budget = 1500
    num = 1400
    bid = np.random.uniform(0, 1, num) * 2 + 2

    X = np.random.permutation(np.arange(len(points)))
    X = X[:num]
    greedy_object = Greedy(points[X], centers[labels][X], r, budget, bid)
    greedy_object.greedy_selection()
    print(len(greedy_object.selected))

