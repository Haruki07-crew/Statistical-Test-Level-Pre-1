"""準1級 教科書 第24章「クラスター分析」の検算（scipy の linkage・k-means の自前実装・総当たり）。"""
import itertools

import numpy as np
from bookcheck import Book
from scipy.cluster.hierarchy import linkage

b = Book("pre1", 24)
pts = np.array([[1, 1], [2, 1], [1, 2], [5, 5], [6, 5], [5, 7], [10, 1]], float)
c = b.chart(1)
b.check("図と表の座標", c["x"] == list(pts[:, 0]) and c["y"] == list(pts[:, 1]) and b.has("| 変数 1 | 1 | 2 | 1 | 5 | 6 | 5 | 10 |"), "")
d = pts[0] - pts[3]
b.num("ユークリッド", np.linalg.norm(d), "5.66")
b.num("マンハッタン", abs(d).sum(), "8")
b.num("チェビシェフ", abs(d).max(), "4")
names = "ABCDEFG"


def members(L, idx, n=7):
    if idx < n:
        return {names[int(idx)]}
    r = L[int(idx) - n]
    return members(L, r[0]) | members(L, r[1])


first4 = None
for m, five, h5, hl in [("single", ({"A", "B", "C"}, {"D", "E", "F"}), "5.00", "5.66"), ("complete", ({"A", "B", "C"}, {"D", "E", "F"}), "7.21", "9.06"),
                        ("average", ({"A", "B", "C"}, {"D", "E", "F"}), "5.99", "7.65"), ("ward", ({"D", "E", "F"}, {"G"}), "8.08", "11.22")]:
    L = linkage(pts, m)
    sets = [frozenset(members(L, 7 + i)) for i in range(4)]
    if first4 is None:
        first4 = set(sets)
    b.check(f"{m}：最初の 4 回は同じ", set(sets) == first4, sets)
    got = {frozenset(members(L, L[4][0])), frozenset(members(L, L[4][1]))}
    b.check(f"{m}：5 回目の併合", got == {frozenset(five[0]), frozenset(five[1])}, got)
    b.num(f"{m}：5 回目の高さ", L[4][2], h5)
    b.num(f"{m}：最後の高さ", L[5][2], hl)


def km(cc):
    for _ in range(50):
        lab = np.argmin(((pts[:, None] - cc[None]) ** 2).sum(2), 1)
        cc = np.array([pts[lab == k].mean(0) for k in range(len(cc))])
    return lab, ((pts - cc[lab]) ** 2).sum()


lab, W = km(np.array([[1, 1], [10, 1.]]))
b.check("A・G から：{A,B,C,D,F} と {E,G}", [names[i] for i in range(7) if lab[i] == lab[4]] == ["E", "G"], lab)
b.num("W", W, "61.6")
lab, W = km(np.array([[1, 1], [5, 5.]]))
b.num("A・D から W", W, "37.3")
best = min(sum(((pts[np.array(l) == k] - pts[np.array(l) == k].mean(0)) ** 2).sum() for k in range(2)) for l in itertools.product([0, 1], repeat=7) if len(set(l)) == 2)
b.check("2 分割の総当たりの最小", abs(best - W) < 1e-9, best)
lab, W3 = km(np.array([[1, 1], [5, 5], [10, 1.]]))
b.num("k=3 の W", W3, "4.67")
b.num("全体の平方和", ((pts - pts.mean(0)) ** 2).sum(), "100.3")
# 章末問題
dd = np.array([1, 4, 2.]) - np.array([3, 1, 2.])
b.num("問1 ユークリッド", np.linalg.norm(dd), "3.61")
b.num("問1 マンハッタン", abs(dd).sum(), "5")
b.num("問1 チェビシェフ", abs(dd).max(), "3")
b.num("問2 最短", 0.5 * 4 + 0.5 * 7 - 0.5 * 3, "4")
b.num("問2 最長", 0.5 * 4 + 0.5 * 7 + 0.5 * 3, "7")
x = np.array([1, 2, 4, 7, 8, 10.])
lab1 = (abs(x - 8) < abs(x - 2)).astype(int)
b.num("問3 中心 1", x[lab1 == 0].mean(), "2.33")
b.num("問3 中心 2", x[lab1 == 1].mean(), "8.33")
b.done()
