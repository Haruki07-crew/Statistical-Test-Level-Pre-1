"""2級 第2章（1変数データの要約）の検算。本文の具体例の数値を計算し直す。"""
import numpy as np
from scipy import stats

results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-3):
    return abs(float(x) - claim) <= tol


def quart(v):
    """中央値で上下に分け（奇数なら中央値を除く）、それぞれの中央値を Q1・Q3 とする（本文の定義）。"""
    s = sorted(v); n = len(s); h = n // 2
    return np.median(s[:h]), np.median(s), np.median(s[h + 1:] if n % 2 else s[h:])


A = np.array([2, 4, 5, 5, 6, 8, 9, 11, 22]); B = np.array([3, 5, 6, 7, 7, 8, 9, 10, 17])
T = "mean-median-mode"
check(T, "平均 8・中央値 6・最頻値 5", A.mean() == 8 and np.median(A) == 6 and stats.mode(A).mode == 5, (A.mean(), np.median(A), stats.mode(A).mode))
check(T, "22を除くと平均 6.25・中央値 5.5", A[:-1].mean() == 6.25 and np.median(A[:-1]) == 5.5, (A[:-1].mean(), np.median(A[:-1])))
check(T, "偏差の合計 0", (A - A.mean()).sum() == 0, (A - A.mean()).sum())
T = "sample-variance-sd"
check(T, "偏差平方和 280・うち 196 は 22 から", ((A - 8) ** 2).sum() == 280 and (22 - 8) ** 2 == 196, ((A - 8) ** 2).sum())
check(T, "分散 35・標準偏差 5.92・n で割ると 31.1", A.var(ddof=1) == 35 and close(A.std(ddof=1), 5.92) and close(A.var(), 31.1, 0.05), (A.var(ddof=1), A.std(ddof=1), A.var()))
check(T, "Σx² − n x̄² = 280", (A ** 2).sum() - 9 * 64 == 280, (A ** 2).sum())
check(T, "範囲 20", A.max() - A.min() == 20, A.max() - A.min())
check(T, "b 倍で分散 b² 倍", close((3 * A + 5).var(ddof=1), 9 * 35), (3 * A + 5).var(ddof=1))
T = "quartiles-iqr"
q1, q2, q3 = quart(A)
check(T, "Q1 4.5・中央値 6・Q3 10・IQR 5.5", (q1, q2, q3) == (4.5, 6, 10) and q3 - q1 == 5.5, (q1, q2, q3))
A2 = A.copy(); A2[-1] = 12
check(T, "22 を 12 にしても四分位数は同じ", quart(A2) == (q1, q2, q3), quart(A2))
T = "box-plot"
b1, b2, b3 = quart(B)
check(T, "B: Q1 5.5・中央値 7・Q3 9.5・IQR 4・平均 8", (b1, b2, b3) == (5.5, 7, 9.5) and B.mean() == 8, (b1, b2, b3, B.mean()))
check(T, "A の上側の基準 18.25 で 22 が外れ値・ひげは 11", q3 + 1.5 * (q3 - q1) == 18.25 and A[A <= 18.25].max() == 11, q3 + 1.5 * (q3 - q1))
check(T, "B の上側の基準 15.5 で 17 が外れ値・ひげは 10、下側 −0.5", b3 + 6 == 15.5 and B[B <= 15.5].max() == 10 and b1 - 6 == -0.5, (b3 + 6, b1 - 6))
T = "standard-score"
check(T, "国語 z=1.5 偏差値 65、数学 z=1.3 偏差値 63", (78 - 60) / 12 == 1.5 and (76 - 50) / 20 == 1.3 and 50 + 10 * 1.5 == 65 and close(50 + 10 * 1.3, 63), None)
check(T, "z=2 は上位約 2.3%、z=1 は上位約 16%", close(stats.norm.sf(2), 0.023, 5e-4) and close(stats.norm.sf(1), 0.16, 0.005), (stats.norm.sf(2), stats.norm.sf(1)))
check(T, "|z|≤1 に約 68%、|z|≤2 に約 95%", close(1 - 2 * stats.norm.sf(1), 0.68, 0.005) and close(1 - 2 * stats.norm.sf(2), 0.95, 0.005), None)
z = (A - A.mean()) / A.std(ddof=1)
check(T, "標準化すると平均 0・標準偏差 1", close(z.mean(), 0, 1e-12) and close(z.std(ddof=1), 1, 1e-12), (z.mean(), z.std(ddof=1)))
T = "lorenz-curve-gini"
v = np.array([100, 200, 300, 400, 1000]); cs = np.cumsum(v) / v.sum()
check(T, "累積割合 0.05,0.15,0.30,0.50,1.00", np.allclose(cs, [0.05, 0.15, 0.30, 0.50, 1.0]), cs)
area = sum(0.2 * (a + b) / 2 for a, b in zip(np.r_[0, cs[:-1]], cs))
check(T, "曲線の下の面積 0.30・G=0.40", close(area, 0.30, 1e-12) and close(1 - 2 * area, 0.40, 1e-12), area)
g2 = np.abs(v[:, None] - v[None, :]).sum() / (2 * 25 * v.mean())
check(T, "差の平均の式でも 0.40", close(g2, 0.40, 1e-12), g2)
one = np.array([0, 0, 0, 0, 1.0])
check(T, "1 人がすべてを持つと (n−1)/n = 0.8", close(np.abs(one[:, None] - one[None, :]).sum() / (2 * 25 * one.mean()), 0.8, 1e-12), None)

ng = results.count(False)
print(f"第2章（2級）の検算: OK {results.count(True)} / NG {ng}")
raise SystemExit(1 if ng else 0)
