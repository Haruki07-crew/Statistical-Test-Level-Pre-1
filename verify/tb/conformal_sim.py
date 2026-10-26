"""コンフォーマル予測のページの例：不均一分散のある回帰で、分割コンフォーマル予測区間の被覆率を調べる。verify/tb/ch56.py から使う。

x ~ U(0,1)、y = 1 + 2x + ε、ε = (0.2 + 1.5x²)·t₃（自由度 3 の t 分布）（右側ほど誤差が大きい。誤差の分布の形は予測の側に教えない）。
"""
import numpy as np
from scipy import stats


def gen(rng, n):
    x = rng.random(n)
    return x, 1 + 2 * x + rng.standard_t(3, size=n) * (0.2 + 1.5 * x ** 2)


def split_conformal(xtr, ytr, xcal, ycal, alpha=0.1):
    b = np.polyfit(xtr, ytr, 1)
    s = np.abs(ycal - np.polyval(b, xcal))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return b, np.sort(s)[k - 1]


def run(reps=2000, ntr=200, ncal=200, ntest=1000, alpha=0.1, seed=56):
    rng = np.random.default_rng(seed)
    bins = np.linspace(0, 1, 11)
    cov_c, cov_n, cnt = np.zeros(10), np.zeros(10), np.zeros(10)
    marg_c, marg_n, width = [], [], []
    for _ in range(reps):
        xtr, ytr = gen(rng, ntr)
        xc, yc = gen(rng, ncal)
        xt, yt = gen(rng, ntest)
        b, q = split_conformal(xtr, ytr, xc, yc, alpha)
        pred = np.polyval(b, xt)
        inc = np.abs(yt - pred) <= q
        # 正規分布を仮定した最小二乗の予測区間（訓練データ 200 個の残差の SD を使う。てこ比の補正は小さいので省く）
        sd = np.sqrt(((ytr - np.polyval(b, xtr)) ** 2).sum() / (ntr - 2))
        inn = np.abs(yt - pred) <= stats.t.isf(alpha / 2, ntr - 2) * sd
        idx = np.minimum((xt * 10).astype(int), 9)
        for j in range(10):
            m = idx == j
            cov_c[j] += inc[m].sum(); cov_n[j] += inn[m].sum(); cnt[j] += m.sum()
        marg_c.append(inc.mean()); marg_n.append(inn.mean()); width.append(2 * q)
    return cov_c / cnt, cov_n / cnt, np.mean(marg_c), np.mean(marg_n), np.mean(width)
