import random
import numpy as np
from tqdm import tqdm
from typing import List
from square_ovlp_detection import Square, overlap
from overlap_area import area
from concurrent.futures import ProcessPoolExecutor

# M is the amount of the objectives
RADIUS = 0
M = 3

def match_degree(points, centers, R):
    return 1 - np.linalg.norm(points - centers, axis=1) / R


def dominate(pair):
    # a, b 都是numpy三元组, 如果a支配b 返回1, b支配a返回-1, 其他返回0
    a = obj_matrix[pair[0]]
    b = obj_matrix[pair[1]]

    if np.sum(a >= b) == 3:
        return 1
    if np.sum(a <= b) == 3:
            return -1
    return 0

obj_matrix = np.zeros(10000, 3)


def pair(popu_amount):
    # i, j are the indices of populations
    # this generator produces the index pairs
    for i in range(popu_amount):
        for j in range(i + 1, popu_amount):
            yield (i, j)

def ranking(individuals):
    # 将所有解决方法, 也即individuals进行排序, 得到一个个non-dominant front
    # num_domed是一个方法被支配的数量, doms是此方法支配的方法集合

    num_domed = [0 for _ in range(len(individuals))]
    doms = [[] for _ in range(len(individuals))]

    with ProcessPoolExecutor(4) as pool:
        dom_result = pool.map(dominate, pair(len(individuals)))

    for i in range(len(individuals)):
        for j in range(i+1, len(individuals)):
            tag = next(dom_result)
            if tag == 1:
                doms[i].append(j)
                num_domed[j] += 1
            elif tag == -1:
                doms[j].append(i)
                num_domed[i] += 1

    rankings = []
    ranking_map = [0 for _ in range(len(individuals))]
    P = [i for i in range(len(individuals)) if num_domed[i] == 0]
    # for i in range(len(individuals)):
    #     if num_domed[i] == 0:
    #         P.append(i)

    while len(P) > 0:
        Q = []
        for i in P:
            for j in doms[i]:
                num_domed[j] -= 1
            Q += [j for j in doms[i] if num_domed[j] == 0]
                # if not num_domed[j]:
                #     Q.append(j)
        rankings.append(P)
        P = Q

    for i in range(len(rankings)):
        for j in rankings[i]:
            ranking_map[j] = i
    return rankings, ranking_map

# class Individual:
#     def __init__(self, selection, prob):
#         self.selection = selection
#         self.num = np.sum(selection)
#         self.match_degree = match_degree(prob.points[selection], prob.centers[prob.labels[selection]], RADIUS)
#         self.area = overlap([s for i, s in enumerate(prob.squares) if selection[i]])
#         self.ranking = -1

# multi-objective optimization
class MO:
    def __init__(self, points: np.ndarray, centers: np.ndarray, r: float, budget: float, bid: np.ndarray):
        self.points = points
        self.centers = centers
        self.r = r
        self.budget = budget
        self.bid = bid

        self.squares = [ Square(y+r, y-r, x-r, x+r) for  x, y in points]
        self.matching_degree_all = match_degree(points, centers, r)

    def metric_calculation(self, selections: np.ndarray, triple: np.ndarray):
        assert len(selections) == len(triple), 'matrix selection and triple have different length'

        i = 0
        for s in selections:
            triple[i][0] = np.sum(s)
            triple[i][1] = np.sum(s * self.matching_degree_all)
            triple[i][2] = area([self.squares[j] for j in s if j == 1])
            i += 1

    def NSGA(self, N: int, steps: int, probability: float):
        # large_triple: the metrics of the New_P
        # triple: the metrics of P
        self.large_triple = np.zeros((2 * N, 3), np.float)
        self.triple = np.zeros((N, 3), np.float)

        large_bid = np.concatenate([ self.bid for _ in range(N)], axis=0)
        num = len(self.points)
        P = []
        while len(P) < N:
            foo = np.random.randint(0, 2, (N, num), dtype=np.bool)
            cost = np.sum(foo * large_bid, axis=1)
            P += [ foo[i] for i in range(N) if cost[i] <= self.budget]
        P = P[:N]
        P = np.concatenate(P, axis=0)
        self.metric_calculation(P, self.triple)
        # 这里的ranking_map只是序列的排序
        for _ in tqdm(steps):
            rankings, ranking_map = ranking(P)
            Q = []
            len_Q = 0
            while len_Q < N:
                # 选择
                a = np.random.randint(low=len(P), size=N)
                b = np.random.randint(low=len(P), size=N)
                index = [a[i] if ranking_map[a[i]] >= ranking_map[b[i]] else b[i] for i in range(N)]
                mate = P[index]

                # 杂交
                mate_a = mate[0::2]
                mate_b = mate[1::2]
                # assume that N is even
                assert N % 2 == 0, 'N is not even'
                split = np.random.randint(1, num, N // 2)
                foo = [
                    np.concatenate([
                        np.hstack(mate_a[i][:split[i]], mate_b[i][split[i]:]),
                        np.hstack(mate_a[i][split[i]:], mate_b[i][:split[i]])
                    ],
                    axis=0)
                    for i in range(N // 2)
                ]

                foo = np.concatenate(foo, axis=0)

                # for i in range(0, N, 2):
                #     split = random.randint(1, num-1)
                #     new_a = Individual(np.hstack(mate[i].selection[:split], mate[i+1].selection[split:]), self)
                #     new_b = Individual(np.hstack(mate[i].selection[split:], mate[i+1].selection[:split]), self)
                #     foo.extend([new_a, new_b])

                # 变异

                flip = np.random.uniform(0, 1, (N, num))
                mask = np.zeros((N, num))
                mask[flip < probability] = 1
                foo = np.logical_xor(foo, mask)
                cost = large_bid * foo
                cost = np.sum(cost, axis=1)

                selected = np.concatenate([foo[i] for i in range(N) if cost[i] <= self.budget], axis=0)
                len_Q += len(selected)
                Q.append(selected)

            Q = np.concatenate(Q, axis=0)
            new_P = P + Q[:N]
            self.metric_calculation(new_P, self.large_triple)
            rankings, ranking_map = ranking(new_P)
            temp = [j for r in rankings for j in r][:N]
            P = new_P[np.array(temp)]
            self.triple = self.large_triple[np.array(temp)]

        return P


if __name__ == '__main__':
    a = [1, 2, 3, 4]
    b = np.array(1)
    print(a[b])