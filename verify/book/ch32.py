"""準1級 教科書 第32章「シミュレーション」の検算（同じ乱数で再計算・理論値と照合）。"""
import math

import numpy as np
from bookcheck import Book
from scipy import integrate, stats as st

b = Book("pre1", 32)
x, seq = 1, []
for _ in range(17):
    x = (5 * x + 3) % 16
    seq.append(x)
b.check("線形合同法の列", b.has("8、11、10、5、12、15、14、9、0、3、2、13、4、7、6、1、8") and seq == [8, 11, 10, 5, 12, 15, 14, 9, 0, 3, 2, 13, 4, 7, 6, 1, 8], seq)
b.check("周期 16", len(set(seq[:16])) == 16, "")
b.check("Beta(2,2) の最大値 1.5", abs(st.beta(2, 2).pdf(0.5) - 1.5) < 1e-12, "")
true = integrate.quad(lambda v: math.exp(-v * v), 0, 1)[0]
b.num("真の値", true, "0.7468")
b.num("真の値（3 桁）", true, "0.747")
rng = np.random.default_rng(32)
u = rng.uniform(size=10000)
f = np.exp(-u ** 2)
b.num("MC 推定値", f.mean(), "0.7471")
b.num("MC 標準誤差", f.std() / 100, "0.0020")
f2 = (np.exp(-u[:5000] ** 2) + np.exp(-(1 - u[:5000]) ** 2)) / 2
b.num("対照変量の標準誤差", f2.std() / math.sqrt(5000), "0.0004")
b.num("P(Z>4)", st.norm.sf(4) * 1e5, "3.17")
b.num("1 万個中の期待個数", 10000 * st.norm.sf(4), "0.3")
z = rng.normal(4, 1, 10000)
w = np.exp(-4 * z + 8)
b.check("重みの式 f/g = e^{-4x+8}", np.allclose(st.norm.pdf(z) / st.norm(4, 1).pdf(z), w), "")
est = np.mean((z > 4) * w)
b.num("IS 推定値 ×1e5", est * 1e5, "3.17")
b.num("IS 標準誤差 ×1e7", ((z > 4) * w).std() / 100 * 1e7, "6.7")
b.num("相対誤差 %", ((z > 4) * w).std() / 100 / est * 100, "2", tol=0.5)
# ボックス・ミュラー
u1, u2 = rng.uniform(size=200000), rng.uniform(size=200000)
z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
b.check("ボックス・ミュラー：正規で無相関", abs(z1.std() - 1) < 0.01 and abs(np.corrcoef(z1, z2)[0, 1]) < 0.01 and st.kstest(z1, "norm").pvalue > 0.01, "")
# MH（標準正規、提案 sd 2.4）
xx, out, acc = 0.0, [], 0
for _ in range(100000):
    y = xx + rng.normal(0, 2.4)
    if rng.uniform() < min(1, math.exp(-(y * y - xx * xx) / 2)):
        xx, acc = y, acc + 1
    out.append(xx)
out = np.array(out[1000:])
b.check("MH の標本は N(0,1)", abs(out.mean()) < 0.03 and abs(out.var() - 1) < 0.04, (out.mean(), out.var()))
# ギブス
x1 = x2 = 0.0
G = []
for _ in range(100000):
    x1 = rng.normal(0.8 * x2, 0.6)
    x2 = rng.normal(0.8 * x1, 0.6)
    G.append((x1, x2))
G = np.array(G[1000:])
b.check("ギブス：相関 0.8", abs(np.corrcoef(G.T)[0, 1] - 0.8) < 0.01, np.corrcoef(G.T)[0, 1])
b.num("条件付き分散", 1 - 0.64, "0.36")
# ブートストラップ
d = np.array([2.1, 3.4, 1.9, 5.6, 2.8, 4.1, 3.0, 7.2, 2.5, 3.7])
b.num("中央値", np.median(d), "3.2")
bs = np.array([np.median(rng.choice(d, 10)) for _ in range(20000)])
b.num("ブートストラップ SE", bs.std(), "0.53", tol=0.02)
lo, hi = np.percentile(bs, [2.5, 97.5])
b.num("2.5% 点", lo, "2.45", tol=0.051)
b.num("97.5% 点", hi, "4.65", tol=0.051)
jk = np.array([np.delete(d, i).mean() for i in range(10)])
sej = math.sqrt(9 / 10 * ((jk - jk.mean()) ** 2).sum())
b.check("ジャックナイフ = s/√n", abs(sej - d.std(ddof=1) / math.sqrt(10)) < 1e-12, "")
b.num("s/√n", d.std(ddof=1) / math.sqrt(10), "0.523")
# 章末問題
b.num("問1", 0.512 ** (1 / 3), "0.8")
b.num("問2 M", 4 / math.pi, "1.27")
b.num("問2 採用率", math.pi / 4, "0.785")
b.num("問3", (2 / 0.01) ** 2, "40000")
b.done()
