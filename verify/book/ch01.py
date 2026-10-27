"""準1級 教科書 第1章「事象と確率」の数値の検算（総当たり・正確な分数で確かめる）。"""
import itertools
import math
from fractions import Fraction as F

from bookcheck import Book

b = Book("pre1", 1)


def nofix(n):
    return sum(F((-1) ** k, math.factorial(k)) for k in range(n + 1))


b.check("n=4 の一致しない確率 3/8", nofix(4) == F(3, 8), nofix(4))
c = sum(1 for p in itertools.permutations(range(4)) if all(p[i] != i for i in range(4)))
b.check("総当たりで 9 通り", c == 9, c)
b.num("0.375", float(nofix(4)), "0.375")
b.num("1/e", math.exp(-1), "0.3679")
b.num("n=10", float(nofix(10)), "0.36788")
b.check("包除の k 番目の項は 1/k!", all(F(math.comb(6, k) * math.factorial(6 - k), math.factorial(6)) == F(1, math.factorial(k)) for k in range(7)), "")


def bday(n):
    p = 1.0
    for k in range(n):
        p *= (365 - k) / 365
    return p


b.num("23 人ですべて異なる", bday(23), "0.493")
b.num("23 人で一致あり", 1 - bday(23), "0.507")
b.num("30 人", 1 - bday(30), "0.706")
b.num("50 人", 1 - bday(50), "0.970")
b.num("23C2", math.comb(23, 2), "253")
b.num("ブールの上界", 253 / 365, "0.693")
S = list(itertools.product("HT", repeat=2))
P = lambda f: F(sum(1 for s in S if f(s)), 4)
A = lambda s: s[0] == "H"
B = lambda s: s[1] == "H"
C = lambda s: s[0] == s[1]
b.check("対ごとに独立", P(lambda s: A(s) and C(s)) == P(A) * P(C) and P(lambda s: A(s) and B(s)) == P(A) * P(B) and P(lambda s: B(s) and C(s)) == P(B) * P(C), "")
b.check("相互独立でない（1/4 ≠ 1/8）", P(lambda s: A(s) and B(s) and C(s)) == F(1, 4) != P(A) * P(B) * P(C), "")
p, se, fp = 0.01, 0.9, 0.05
pt = p * se + (1 - p) * fp
ptt = p * se ** 2 + (1 - p) * fp ** 2
b.num("P(T1)", pt, "0.0585")
b.num("P(T1∩T2)", ptt, "0.010575")
b.num("積", pt ** 2, "0.00342")
b.num("P(T2|T1)", ptt / pt, "0.181")
b.check("約 3 倍", 2.9 < (ptt / pt) / pt < 3.2, (ptt / pt) / pt)
# モンティ・ホール（総当たり）
tot = F(0)
win2 = F(0)
for car in (1, 2, 3):
    opts = [d for d in (2, 3) if d != car]
    for d in opts:
        pr = F(1, 3) * F(1, len(opts))
        if d == 3:
            tot += pr
            win2 += pr if car == 2 else 0
b.check("モンティ・ホール P(扉2|扉3を開けた) = 2/3", win2 / tot == F(2, 3), win2 / tot)
# オッズ
lr = 0.95 / 0.10
po = 0.02 / 0.98
b.num("尤度比", lr, "9.5")
b.num("事前オッズ", po, "0.0204")
b.num("事後オッズ", po * lr, "0.194")
b.num("事後確率", po * lr / (1 + po * lr), "0.162")
b.check("ベイズの定理の直接計算と一致", abs(0.02 * 0.95 / (0.02 * 0.95 + 0.98 * 0.10) - po * lr / (1 + po * lr)) < 1e-12, "")
b.num("2 回目のオッズ", po * lr * lr, "1.84")
b.num("2 回目の確率", po * lr * lr / (1 + po * lr * lr), "0.648")
# 章末問題
n = sum(1 for k in range(1, 101) if k % 2 == 0 or k % 3 == 0 or k % 5 == 0)
b.check("問1 74 個", n == 74 == 50 + 33 + 20 - 16 - 10 - 6 + 3, n)
b.num("問2", 13 * 12 * 11 / (52 * 51 * 50), "0.0129")
b.check("問2 分数", F(1716, 132600) == F(math.comb(13, 3), math.comb(52, 3)) and math.comb(13, 3) == 286 and math.comb(52, 3) == 22100, "")
pr = F(1, 2) * F(3, 4) + F(1, 2) * F(1, 4)
prr = F(1, 2) * F(3, 4) ** 2 + F(1, 2) * F(1, 4) ** 2
b.check("問3 P(R1)=1/2・P(R1∩R2)=5/16", pr == F(1, 2) and prr == F(5, 16) == F(10, 32), (pr, prr))
d = 0.2 * 0.05 + 0.3 * 0.03 + 0.5 * 0.01
b.num("問4 不良品の確率", d, "0.024")
b.num("問4 事後確率", 0.01 / d, "0.417")
b.done()
