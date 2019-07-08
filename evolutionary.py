import random
import numpy as np
from tqdm import tqdm
from square_ovlp_detection import Square, overlap

RADIUS = 0


def match_degree(points, centers, R):
    return 1 - np.linalg.norm(points - centers, axis=1) / R


def dominate(a, b):
    # a, b 都是三元组, 如果a支配b 返回1, b支配a返回-1, 其他返回0
    if not (a[0] == b[0] and a[1] == b[1] and a[2] == b[2]):
        if a[0] >= b[0] and a[1] >= b[1] and a[2] >= b[2]:
            return 1
        if a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]:
            return -1

    return 0


def ranking(individuals):
    # 将所有解决方法, 也即individuals进行排序, 得到一个个non-dominant front
    # num_domed是一个方法被支配的数量, doms是此方法支配的方法集合
    num_domed = [0 for _ in range(len(individuals))]
    doms = [[] for _ in range(len(individuals))]
    for i in range(len(individuals)):
        for j in range(i+1, len(individuals)):
            a = [individuals[i].num, individuals[i].match_degree, individuals[i].area]
            b = [individuals[j].num, individuals[j].match_degree, individuals[j].area]
            r = dominate(a, b)
            if r == 1:
                num_domed[j] += 1
                doms[i].append(j)
            if r == -1:
                num_domed[i] += 1
                doms[j].append(i)

    rankings = []
    P = []
    for i in range(len(individuals)):
        if num_domed[i] == 0:
            P.append(i)

    while len(P) > 0:
        Q = []
        for i in P:
            for j in doms[i]:
                num_domed[j] -= 1
                if not num_domed[j]:
                    Q.append(j)
        rankings.append(P)
        P = Q

    return rankings

class Individual:
    def __init__(self, selection, prob):
        self.selection = selection
        self.num = np.sum(selection)
        self.match_degree = match_degree(prob.points[selection], prob.centers[prob.labels[selection]], RADIUS)
        self.area = overlap([s for i, s in enumerate(prob.squares) if selection[i]])
        self.ranking = -1

# multi-objective optimization
class MO:
    def __init__(self, r, points, centers):
        self.r = r
        self.points = points
        self.center = centers

        self.squares = []
        for p in points:
            self.squares.append(Square(p[1]+r, p[1]-r, p[0]-r, p[0]+r))

    def NSGA(self, N, steps, probability):
        num = len(self.points)
        P = np.random.randint(0, 2, (N, num))
        P = [Individual(s, self) for s in P]
        # 这里的ranking_P只是序列的排序
        for _ in tqdm(steps):
            Q = []
            mate = []
            # 选择
            for _ in range(N):
                i = random.randrange(0, N)
                j = random.randrange(0, N)
                Q.append(P[[j, i][i >= j]])

            # 杂交
            for _ in range(N):
                a, b = random.sample(P)
                if dominate(a, b) >= 0:
                    mate.append(a)
                else:
                    mate.append(b)

            for i in range(0, N, 2):
                split = random.randint(1, num-1)
                new_a = Individual(np.hstack(mate[i].selection[:split], mate[i+1].selection[split:]), self)
                new_b = Individual(np.hstack(mate[i].selection[split:], mate[i+1].selection[:split]), self)
                Q.extend([new_a, new_b])

            # 变异
            for i in Q:
                flip = np.random.uniform(0, 1, num)
                mask = np.zeros(num)
                mask[flip < probability] = 1
                i.selection = np.logical_xor(i.selection, mask)

            new_P = P + Q
            rankings_P = ranking(new_P)
            P = []
            collected = 0
            for front in rankings_P:
                if collected + len(front) > N:
                    P.extend([new_P[i] for i in random.sample(front, N - collected)])
                    break

                P.extend([new_P[i] for i in front])

        return P







if __name__ == '__main__':
    a = [1, 2, 3, 4]
    b = np.array(1)
    print(a[b])