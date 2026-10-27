"""2級 教科書 第7章「確率」の数値の検算（数え上げは総当たりでも確かめる）。"""
import itertools
from fractions import Fraction as F
from math import comb, perm

from bookcheck import Book

b = Book("g2", 7)
b.num("8P3", perm(8, 3), "336")
b.num("8C3", comb(8, 3), "56")
b.num("10C4", comb(10, 4), "210")
b.num("6C2", comb(6, 2), "15")
b.num("4C2", comb(4, 2), "6")
b.num("男2女2の数", comb(6, 2) * comb(4, 2), "90")
ppl = ["M"] * 6 + ["W"] * 4
cnt = sum(1 for s in itertools.combinations(range(10), 4) if sum(ppl[i] == "M" for i in s) == 2)
b.check("総当たりでも 90 通り", cnt == 90, cnt)
b.check("3/7", F(90, 210) == F(3, 7), F(90, 210))
b.num("3/7", 3 / 7, "0.429")
S = list(itertools.product(range(1, 7), repeat=2))
P = lambda ev: F(sum(1 for s in S if ev(s)), 36)
b.check("和が 7 は 6 通り", P(lambda s: sum(s) == 7) == F(6, 36), "")
b.check("少なくとも 1 個が 6 は 11/36", P(lambda s: 6 in s) == F(11, 36), "")
b.check("どちらも 6 でないは 25/36", P(lambda s: 6 not in s) == F(25, 36), "")
A = lambda s: s[0] % 2 == 0
b.check("1 個目が偶数は 18 通り", sum(1 for s in S if A(s)) == 18, "")
b.check("偶数かつ和10以上は 4 通り", sum(1 for s in S if A(s) and sum(s) >= 10) == 4, "")
b.check("P(和≥10 | 1個目偶数) = 2/9", P(lambda s: A(s) and sum(s) >= 10) / P(A) == F(2, 9), "")
b.check("P(和≥10) = 1/6", P(lambda s: sum(s) >= 10) == F(1, 6), "")
b.check("偶数と和7は独立", P(lambda s: A(s) and sum(s) == 7) == P(A) * P(lambda s: sum(s) == 7) == F(1, 12), "")
b.check("偶数と和≥10は独立でない（1/9 ≠ 1/12）", P(lambda s: A(s) and sum(s) >= 10) == F(1, 9) and P(A) * P(lambda s: sum(s) >= 10) == F(1, 12), "")
# くじ
b.check("2 本とも当たり 2/15", F(4, 10) * F(3, 9) == F(2, 15), "")
b.check("ちょうど 1 本 8/15", F(4, 10) * F(6, 9) + F(6, 10) * F(4, 9) == F(8, 15), "")
b.check("24/90", F(4, 10) * F(6, 9) == F(24, 90), "")
lots = [1] * 4 + [0] * 6
pairs = list(itertools.permutations(range(10), 2))
b.check("総当たり：2 本とも当たり", F(sum(1 for i, j in pairs if lots[i] and lots[j]), len(pairs)) == F(2, 15), "")
b.num("2/15", 2 / 15, "0.133")
b.num("8/15", 8 / 15, "0.533")
b.num("(5/6)^4", (5 / 6) ** 4, "0.482")
b.num("1−(5/6)^4", 1 - (5 / 6) ** 4, "0.518")
# 工場
w, d = [0.5, 0.3, 0.2], [0.01, 0.02, 0.04]
pd = sum(x * y for x, y in zip(w, d))
b.num("全体の不良率", pd, "0.019")
b.num("ライン3の事後確率", w[2] * d[2] / pd, "0.421")
# 検査
p, se, fp = 0.01, 0.9, 0.05
pp = p * se + (1 - p) * fp
b.num("陽性の確率", pp, "0.0585")
b.num("0.99×0.05", 0.99 * 0.05, "0.0495")
post = p * se / pp
b.num("陽性的中率", post, "0.154")
b.check("1 万人の表", 100 * 0.9 == 90 and abs(9900 * 0.05 - 495) < 1e-9 and 90 + 495 == 585 and 10 + 9405 == 9415 and 9900 - 495 == 9405, "")
b.num("90/585", 90 / 585, "0.154")
b.num("2 回目の事後確率（0.154 から）", 0.154 * 0.9 / (0.154 * 0.9 + 0.846 * 0.05), "0.766")
b.num("2 回目の事後確率（正確な値から）", post * 0.9 / (post * 0.9 + (1 - post) * 0.05), "0.766")
# 章末問題
b.num("9C3", comb(9, 3), "84")
b.num("5C3", comb(5, 3), "10")
b.check("5/42", F(10, 84) == F(5, 42), "")
b.num("5/42", 5 / 42, "0.119")
b.num("37/42", 37 / 42, "0.881")
cards = [(s, r) for s in range(4) for r in range(1, 14)]
pa = F(sum(1 for c in cards if c[0] == 0), 52)
pb = F(sum(1 for c in cards if c[1] >= 11), 52)
pab = F(sum(1 for c in cards if c[0] == 0 and c[1] >= 11), 52)
b.check("トランプ：独立", pa == F(1, 4) and pb == F(3, 13) and pab == F(3, 52) == pa * pb, (pa, pb, pab))
b.num("迷惑メール判定の割合", 0.2 * 0.98 + 0.8 * 0.01, "0.204")
b.num("迷惑メールの事後確率", 0.196 / 0.204, "0.961")
b.num("独立のときの和事象", 0.4 + 0.5 - 0.2, "0.7")
b.done()
