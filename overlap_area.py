'''
计算互相重叠的正方形的面积
'''
import time
import numpy as np
from square_ovlp_detection import Horizontal, Square, overlap


def merge_width(squares: list):
    n = len(squares)
    if n == 0:
        return 0

    squares.sort(key=lambda s: s.left)
    merge_w = 0
    interval = [squares[0].left, squares[0].right]
    for i in range(1, n):
        if interval[1] >= squares[i].left:
            interval[1] = squares[i].right
        else:
            merge_w += interval[1] - interval[0]
            interval = [squares[i].left, squares[i].right]

    merge_w += interval[1] - interval[0]
    return merge_w


def area(squares: list):
    horizontals = []

    for s in squares:
        up_edge = Horizontal(s.left, s.right, s.up, belong=s)
        down_edge = Horizontal(s.left, s.right, s.down, belong=s)
        horizontals.extend([up_edge, down_edge])

    horizontals.sort(reverse=True)

    # 在两条水平边之间有重叠部分的正方形
    a = 0
    exist_squares = [horizontals[0].belong]

    for i in range(1, len(horizontals)):
        u = horizontals[i-1].y
        d = horizontals[i].y
        if u != d:
            w = merge_width(exist_squares)
            if w != 0:
                a += w * (u - d)

        if d == horizontals[i].belong.down:
            exist_squares.remove(horizontals[i].belong)
        else:
            exist_squares.append(horizontals[i].belong)

    return a


if __name__ == '__main__':
    # foo = [
    #     [2, 3],
    #     [3.3, 4.3],
    #     [5.5, 6.5],
    #     [10, 11],
    #     [1.2, 2.2],
    #     [4.8, 5.8],
    #     [2.7, 3.7],
    #     [4.6, 5.6],
    #     [9.2, 10.2],
    #     [11.3, 12.3]
    # ]

    foo = [
        [2, 0, 0, 2],
        [3, 1, 1, 3],
        [4, 2, 2, 4],
        [2.5, 0.5, 2.5, 4.5],
        [1.5, -0.5, 3.5, 5.5]
    ]
    s = []
    for f in foo:
        s.append(Square(f[0], f[1], f[2], f[3]))

    print(area(s))

    d = np.load('data/points.npz')
    points = d['points']
    print(points.shape)
    labels = d['labels']

    r = 0.1
    s = [Square(y + r, y - r, x - r, x + r) for x, y in points]
    # for x, y in points:
    #     s.append(Square(y + r, y - r, x - r, x + r))

    start = time.time()
    a = area(s)
    end = time.time()
    print(a, end - start)

    foo = [
        [9, -1, -1, 9],
        [8, -2, -2, 8],
        [7, -3, -3, 7]
    ]

    s = [Square(u, b, l ,r) for u, b, l ,r in foo]
    a = area(s)
    print(a)

