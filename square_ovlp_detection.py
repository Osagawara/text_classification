'''
检查一些正方形中,是否有重叠部分, 报告所有重叠的正方形
'''
import math
import bisect
import time
import random
import numpy as np
from functools import cmp_to_key
from bintrees import FastRBTree

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

class Point:
    def __init__(self, x, y, side, belong=None):
        self.x = x
        self.y = y
        self.side = side
        self.belong = belong
        self.key = self.x

    def __cmp__(self, other):
        if self.key == other.key:
            if isinstance(other, Point):
                return self.side - other.side
            else:
                if self.side == LEFT:
                    return -1
                else:
                    return 1
        elif self.key < other.key:
            return -1
        else:
            return 1


class Square:
    def __init__(self, up, down, left, right):
        self.up = up
        self.down = down
        self.left = left
        self.right = right


class Horizontal:
    def __init__(self, left, right, y, belong=None):
        '''
        构建水平边
        :param left: 左顶点的横坐标
        :param right: 右顶点的横坐标
        :param y: 边的纵坐标, 同时也是比较用的键值
        :param belong: 属于哪个正方形
        '''
        self.left = left
        self.right = right
        self.y = y
        self.belong = belong
        self.key = y

    def __cmp__(self, other):
        if self.key == other.key:
            return 0
        elif self.key < other.key:
            return -1
        else:
            return 1

class Vertical:
    def __init__(self, up, down, x, belong=None):
        self.up = up
        self.down = down
        self.x = x
        self.belong = belong
        self.key = x

    def __cmp__(self, other):
        if self.key == other.key:
            if isinstance(other, Vertical):
                return 0
            else:
                if other.side == LEFT:
                    return 1
                else:
                    return -1
        elif self.key < other.key:
            return -1
        else:
            return 1


def overlap(squares):
    '''
    输入正方形, 返回它们重叠的关系图
    :param squares: Square对象的list
    :return: 一个邻接表,表示正方形的重叠关系
    '''

    # 构建横边和竖边
    horizontals = []
    verticals = []

    for s in squares:
        up_edge = Horizontal(s.left, s.right, s.up, belong=s)
        down_edge = Horizontal(s.left, s.right, s.down, belong=s)
        horizontals.extend([up_edge, down_edge])

        left_edge = Vertical(s.up, s.down, s.left, belong=s)
        right_edge = Vertical(s.up, s.down, s.right, belong=s)
        verticals.extend([left_edge, right_edge])

    Q = []
    for e in horizontals:
        lp = Point(e.left, e.y, LEFT, belong=e)
        rp = Point(e.right, e.y, RIGHT, belong=e)
        Q.extend([lp, rp])
    Q.extend(verticals)

    # 由于横坐标有可能相同, 故而按照算法需要左边点在前, 垂直边在中间, 右边点在后面
    # 所以使用内建的比较函数
    Q.sort()
    cross_relation = []    # 每一个元素为相交的两条边
    R = []  # 水平边按照纵坐标的排序
    for p in Q:
        if isinstance(p, Point):
            if p.x == p.belong.left:
                index = bisect.bisect_right(R, p.belong)
                R.insert(index, p.belong.copy())
            else:
                index = bisect.bisect_left(R, p.belong)
                R = R[:index] + R[index+1:]

        else:
            bottom = Point(p.x, p.down, DOWN, belong=p)
            top = Point(p.x, p.up, UP, belong=p)

            down_index = bisect.bisect_left(R, bottom)
            up_index = bisect.bisect_right(R, top)
            for h in R[down_index:up_index]:
                cross_relation.append([h, p])

    overlaps = dict()   # 用字典表示邻接表
    for cr in cross_relation:
        h, v = cr
        if h.belong not in overlaps:
            overlaps[h] = set()
        overlaps[h].add(v)

    return overlaps

# d = np.load('data/points.npz')
# points = d['points']
# labels = d['labels']
#
# i = np.argsort(points[:, 0])
# points = points[i]
#
# count = 0
# for i in range(len(points) - 1):
#     if points[i][0] == points[i+1][0] and not points[i][1] == points[i+1][1]:
#         print(points[i], points[i+1])
#
# print(count)






