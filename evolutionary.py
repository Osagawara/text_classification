import time
import numpy as np
from tqdm import tqdm
from typing import List
from square_ovlp_detection import Square, overlap
from overlap_area import area

# M is the amount of the objectives
RADIUS = 0
M = 3

def match_degree(points, centers, R):
    return 1 - np.linalg.norm(points - centers, axis=1) / R

def ranking(individuals):
    # 将所有解决方法, 也即individuals进行排序, 得到一个个non-dominant front
    # num_domed是一个方法被支配的数量, doms是此方法支配的方法集合

    num_domed = [0 for _ in range(len(individuals))]
    doms = [[] for _ in range(len(individuals))]

    for i in range(len(individuals)):
        for j in range(i+1, len(individuals)):
            a = individuals[i]
            b = individuals[j]

            if all(a >= b):
                doms[i].append(j)
                num_domed[j] += 1
            if all(a <= b):
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

# multi-objective optimization
class MO:
    def __init__(self, points: np.ndarray, centers: np.ndarray, r: float, budget: float, bid: np.ndarray):
        self.points = points
        self.centers = centers
        self.r = r
        self.budget = budget
        self.bid = bid

        self.squares = [ Square(y+r, y-r, x-r, x+r) for  x, y in points ]
        self.matching_degree_all = match_degree(points, centers, r)

    def metric_calculation(self, selections: np.ndarray, triple: np.ndarray):
        assert len(selections) == len(triple), 'matrix selection and triple have different length'

        i = 0
        for s in selections:
            triple[i][0] = np.sum(s)
            triple[i][1] = np.sum(s * self.matching_degree_all)
            foo = [self.squares[j] for j in range(len(s)) if s[j]]
            triple[i][2] = area(foo)
            i += 1

    def NSGA(self, N: int, steps: int, probability: float):
        # large_triple: the metrics of the New_P
        # triple: the metrics of P
        self.large_triple = np.zeros((2 * N, 3), np.float)
        self.triple = np.zeros((N, 3), np.float)

        num = len(self.points)
        foo = self.bid.reshape((1, num))
        large_bid = np.vstack( ( self.bid for _ in range(N) ) )
        P = []
        len_P = 0
        while len_P < N:
            foo = np.random.randint(0, 2, (N, num), dtype=np.bool)
            cost = np.sum(foo * large_bid, axis=1)
            foo = foo[cost <= self.budget]
            if len(foo.shape) == 1:
                foo = foo[np.newaxis, :]
            P.append(foo)
            len_P += len(foo)

        P = np.concatenate(P, axis=0)[:N]
        self.metric_calculation(P, self.triple)
        # 这里的ranking_map只是序列的排序
        for _ in range(steps):
            rankings, ranking_map = ranking(self.triple)
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
                foo1 = np.vstack( ( np.hstack( (mate_a[i][:split[i]], mate_b[i][split[i]:]) ) for i in range( N // 2 ) ) )
                foo2 = np.vstack( ( np.hstack( (mate_a[i][split[i]:], mate_b[i][:split[i]]) ) for i in range( N // 2 ) ) )

                foo = np.concatenate([foo1, foo2], axis=0)

                # 变异

                flip = np.random.uniform(0, 1, (N, num))
                mask = np.zeros((N, num))
                mask[flip < probability] = 1
                foo = np.logical_xor(foo, mask)
                cost = large_bid * foo
                cost = np.sum(cost, axis=1)

                selected = foo[cost <= self.budget]
                if len(selected.shape) == 1:
                    selected = selected[np.newaxis, :]
                len_Q += len(selected)
                Q.append(selected)

            Q = np.vstack( tuple(Q) )[:N]
            new_P = np.vstack((P, Q))
            self.metric_calculation(new_P, self.large_triple)
            rankings, ranking_map = ranking(self.large_triple)
            temp = [j for r in rankings for j in r][:N]
            P = new_P[np.array(temp)]
            self.triple = self.large_triple[np.array(temp)]

        return P


if __name__ == '__main__':
    d = np.load('data/points.npz')
    points = d['points']
    labels = d['labels']
    centers = np.load('data/centers.npy')
    X = np.random.permutation(np.arange(len(points)))

    num = 1000
    X = X[:num]
    r = 1.2
    budget = 1500
    bid = np.random.uniform(0, 1, num) * 2 + 2
    MO_object = MO(points[X], centers[labels][X], r, budget, bid)


    population = 100
    step = 1
    probability = 0.05

    P = MO_object.NSGA(population, step, probability)



