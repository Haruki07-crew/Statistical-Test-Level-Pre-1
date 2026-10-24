"""模擬試験 第15回（set15）の検算。generator/set15.py とは別の方法（scipy の分布関数・数値積分・シミュレーション・
全列挙・行列計算・定義からの直接計算）で正解を求め直し、問題文に示した数表の値も確かめる。"""
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examcheck import Exam  # noqa: E402

E = Exam(15)
rng = np.random.default_rng(20261024)


def rnd(x, d):
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f(x, d):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def nums(iid):
    return [float(re.findall(r"-?\d+\.?\d*", str(c))[-1]) for c in E.item(iid)["choices"]]


def nearest(iid, value, what=""):
    xs = nums(iid)
    k = int(np.argmin([abs(x - value) for x in xs]))
    E.check(iid, k == E.item(iid)["answer"], f"最も近い選択肢 {k}（値 {value:.5f}）が正解 {E.item(iid)['answer']} {what}")


def distinct(iid, d):
    xs = nums(iid)
    E.check(iid, len({f(x, d) for x in xs}) == 5, f"選択肢の数値が小数第{d}位で互いに異なる {xs}")


# ---- 大問1
# (a) 真の状態とA・Bの判定をシミュレーション（条件付き独立）
N = 1_000_000
D = rng.random(N) < 0.3
A = np.where(D, rng.random(N) < 0.80, rng.random(N) < 0.10)
B = np.where(D, rng.random(N) < 0.70, rng.random(N) < 0.05)
pos = A | B
sens, spec = pos[D].mean(), (~pos[~D]).mean()
E.check("s15p01a", abs(sens - 0.94) < 0.002 and abs(spec - 0.855) < 0.002, f"シミュレーション 感度 {sens:.4f} 特異度 {spec:.4f}")
E.choice("s15p01a", f"感度{1 - 0.2 * 0.3:.3f}、特異度{0.9 * 0.95:.3f}", "全確率から")
# (b) scipy で区間と表の e^x
tab = np.array([[24, 36], [24, 96]])
lor = math.log(tab[0, 0] * tab[1, 1] / (tab[0, 1] * tab[1, 0]))
se = math.sqrt((1 / tab).sum())
z = stats.norm.isf(0.025)
lo, hi = math.exp(lor - z * se), math.exp(lor + z * se)
E.choice("s15p01b", f"({f(lo, 2)},\\{f(hi, 2)})", f"正確な z と log で ({lo:.4f}, {hi:.4f})")
E.check("s15p01b", f(lor, 3) == "0.981", "log(8/3)=0.981")
for x, v in [(0.298, "1.347"), (0.408, "1.504"), (0.743, "2.102"), (1.219, "3.384"), (1.554, "4.730"), (1.664, "5.280")]:
    E.check("s15p01b", f(math.exp(x), 3) == v, f"e^{x}={v}")

# ---- 大問2
# (a) ハザードを数値積分して条件付き確率
h = lambda t: 0.06 if t < 1 else (0.02 if t < 4 else 0.01)
H = integrate.quad(h, 1, 12, points=[4])[0]
E.num("s15p02a", rnd(1 - math.exp(-H), 3), f"数値積分 H={H:.4f}")
for x, v in [(0.14, "0.8694"), (0.20, "0.8187"), (0.12, "0.8869")]:
    E.check("s15p02a", f(math.exp(-x), 4) == v, f"e^-{x}={v}")
# (b) カプラン・マイヤーとグリーンウッドを手続きで計算
times = [2, 3, 4, 4, 6, 7, 9, 10, 12, 12]
event = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
S, g = 1.0, 0.0
for t in sorted(set(tt for tt, e in zip(times, event) if e)):
    if t > 7:
        break
    n = sum(1 for tt in times if tt >= t)
    d = sum(1 for tt, e in zip(times, event) if tt == t and e)
    S *= 1 - d / n
    g += d / (n * (n - d))
E.num("s15p02b", rnd(math.sqrt(S * S * g), 3), f"S(7)={S:.4f}")

# ---- 大問3
# (a) シミュレーションで対数正規の変動係数
C = np.exp(rng.normal(1.0, math.sqrt(0.2), 2_000_000))
cv_sim = C.std() / C.mean()
E.check("s15p03a", abs(cv_sim - math.sqrt(math.exp(0.2) - 1)) < 0.003, f"シミュレーション CV {cv_sim:.4f}")
E.choice("s15p03a", f"${f(math.sqrt(math.exp(0.2) - 1), 3)}$", "√(e^0.2−1)")
E.check("s15p03a", f(math.exp(0.2), 4) == "1.2214" and f(math.exp(0.1), 4) == "1.1052", "e^0.2, e^0.1")
# (b) 比の標準誤差をシミュレーション
xt = rng.normal(80, 16 / math.sqrt(20), 1_000_000)
xr = rng.normal(100, 20 / math.sqrt(20), 1_000_000)
sd_sim = (xt / xr).std()
nearest("s15p03b", sd_sim, "（シミュレーション）")
distinct("s15p03b", 4)

# ---- 大問4
for (m, n, c), v in {(9, 7, 2.25): "0.1488", (9, 7, 1.5): "0.3033", (7, 9, 2.25): "0.1278", (10, 8, 2.25): "0.1315"}.items():
    E.check("s15p04a", f(stats.f.sf(c, m, n), 4) == v, f"P(F({m},{n})>{c})={v}")
Xa = rng.normal(0, 1, (300_000, 10))
Xb = rng.normal(0, 1, (300_000, 8))
pr = np.mean(Xa.std(axis=1, ddof=1) > 1.5 * Xb.std(axis=1, ddof=1))
E.check("s15p04a", abs(pr - 0.1488) < 0.003, f"シミュレーション {pr:.4f}")
E.choice("s15p04a", f"${f(stats.f.sf(2.25, 9, 7), 4)}$", "scipy f.sf(2.25, 9, 7)")
# (b) scipy の t 分布で区間
sp2 = (9 * 0.2 ** 2 + 7 * 0.16 ** 2) / 16
hw = stats.t.isf(0.025, 16) * math.sqrt(sp2 * (1 / 10 + 1 / 8))
E.check("s15p04b", f(stats.t.isf(0.025, 16), 3) == "2.120" and f(stats.t.isf(0.05, 16), 3) == "1.746", "t 表")
E.choice("s15p04b", f"({f(0.22 - hw, 3)},\\{f(0.22 + hw, 3)})", "scipy t.isf で区間")
# (c) 実データで単位変換して回帰し直す
x = rng.uniform(60, 300, 50)
y = 3.6 + 0.98 * x + rng.normal(0, 8, 50)


def fit(xx, yy):
    X = np.column_stack([np.ones(len(xx)), xx])
    b, rss = np.linalg.lstsq(X, yy, rcond=None)[:2]
    s2 = rss[0] / (len(xx) - 2)
    seb = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    r2 = 1 - rss[0] / ((yy - yy.mean()) ** 2).sum()
    return b, b[1] / seb, r2


b1, t1, r1 = fit(x, y)
b2, t2, r2 = fit(x / 18, y / 18)
E.check("s15p04c", abs(b1[1] - b2[1]) < 1e-10 and abs(b1[0] / 18 - b2[0]) < 1e-10 and abs(t1 - t2) < 1e-8 and abs(r1 - r2) < 1e-12,
        "両方 18 で割ると傾き・t・R² 不変、切片は 1/18")
E.choice("s15p04c", f"切片は{f(3.6 / 18, 2)}、傾きは0.98のまま、傾きの$t$値と決定係数も変わらない", "正解の文")

# ---- 大問5
# (a) 順序統計量をシミュレーション
L = np.sort(rng.exponential(36, (1_000_000, 5)), axis=1)
E.check("s15p05a", abs(L[:, 1].mean() - 16.2) < 0.05, f"シミュレーション E[X(2)]={L[:, 1].mean():.3f}")
E.num("s15p05a", rnd(integrate.quad(lambda t: t * 20 * (1 - math.exp(-t / 36)) * math.exp(-4 * t / 36) / 36, 0, np.inf)[0], 1),
      "X(2) の密度を数値積分")
# (b) 大きさ 5% の確認と棄却限界
c = 36 * stats.chi2.ppf(0.05, 10) / 2
E.check("s15p05b", f(stats.chi2.ppf(0.05, 10), 3) == "3.940" and f(stats.chi2.isf(0.05, 10), 2) == "18.31" and f(stats.chi2.ppf(0.05, 5), 3) == "1.145", "χ² 表")
E.choice("s15p05b", f"\\sumX_i\\le{f(c, 1)}$", "scipy の χ² 点")
size = np.mean(rng.exponential(36, (400_000, 5)).sum(axis=1) <= 70.9)
E.check("s15p05b", abs(size - 0.05) < 0.002, f"大きさのシミュレーション {size:.4f}")

# ---- 大問6：全列挙で最小の k
k = next(k for k in range(1, 200) if 1 - 0.95 ** k >= 0.5)
E.num("s15p06a", k, "1−0.95^k≥0.5 を直接探索")
E.check("s15p06a", f(math.log(0.5), 4) == "-0.6931" and f(math.log(0.95), 4) == "-0.0513", "対数の値")
# 無記憶性のシミュレーション
G = rng.geometric(0.05, 2_000_000)
cond = G[G > 20] - 20
E.check("s15p06a", abs(np.median(cond) - 14) <= 0.5, f"20 人後の追加人数の中央値 {np.median(cond)}")

# ---- 大問7
# (a) 帰無仮説のもとでの z（プール）
p = 18 / 360
zz = (14 / 200 - 4 / 160) / math.sqrt(p * (1 - p) * (1 / 200 + 1 / 160))
E.choice("s15p07a", f"$z={f(zz, 2)}$で1.96を超えないので、帰無仮説を棄却しない", "プールした z")
chi2, pv, _, _ = stats.chi2_contingency(np.array([[14, 186], [4, 156]]), correction=False)
E.check("s15p07a", abs(chi2 - zz ** 2) < 1e-9 and pv > 0.05, f"連続修正なしの χ² = z² ({chi2:.4f}, p={pv:.4f})")
# (b) 経験ベイズ
yv = np.array([12, 15, 9, 14, 10.0])
tau2 = yv.var(ddof=1) - 4
post = yv.mean() + tau2 / (tau2 + 4) * (yv[1] - yv.mean())
E.num("s15p07b", rnd(post, 2), "事後平均 = (τ²y+σ²μ)/(τ²+σ²)")

# ---- 大問8：推移確率行列の固有ベクトル
P = np.zeros((4, 4))
for kk in range(4):
    up = 0.4 if kk < 3 else 0
    dn = 0.2 * kk
    if kk < 3:
        P[kk, kk + 1] = up
    if kk > 0:
        P[kk, kk - 1] = dn
    P[kk, kk] = 1 - up - dn
w, v = np.linalg.eig(P.T)
pi = np.real(v[:, np.argmin(abs(w - 1))])
pi /= pi.sum()
E.num("s15p08a", rnd(pi[3], 3), f"固有ベクトル π={np.round(pi, 4)}")
E.check("s15p08a", np.allclose(np.linalg.matrix_power(P, 200)[0], pi), "P^200 の行が定常分布")

# ---- 大問9
# (a) データを作って二元配置の平方和から偏 η²（与えた平方和の比で定義どおり）
E.choice("s15p09a", f"${f(48 / 90, 3)}$", "48/(48+42)")
distinct("s15p09a", 2)
# (b) 全並べ替えで J の帰無分布の平均・分散を確かめ z を計算
vals = [1, 3, 4, 7, 2, 6, 8, 9, 5, 10, 11, 12]


def Jstat(arr):
    g = [arr[0:4], arr[4:8], arr[8:12]]
    return sum(1 for i in range(3) for j in range(i + 1, 3) for a in g[i] for b in g[j] if a < b)


J = Jstat(vals)
sim = np.array([Jstat(list(rng.permutation(12) + 1)) for _ in range(60_000)])
E.check("s15p09b", abs(sim.mean() - 24) < 0.1 and abs(sim.var() - 46.667) < 1.0, f"並べ替え分布 平均 {sim.mean():.3f} 分散 {sim.var():.3f}")
E.num("s15p09b", rnd((J - 24) / math.sqrt(3360 / 72), 2), f"J={J}")

# ---- 大問10
# (a) 折れ線の予測
E.choice("s15p10a", f"${f(120 - 0.5 * 80 - 0.8 * max(80 - 65, 0), 1)}$", "予測値")
distinct("s15p10a", 1)
# (b) 実際にデータで DFBETAS の定義を確認したうえで値を計算
xx = rng.uniform(40, 90, 40)
yy = 120 - 0.9 * xx + rng.normal(0, 8, 40)
X = np.column_stack([np.ones(40), xx])
b = np.linalg.lstsq(X, yy, rcond=None)[0]
i = 5
Xi, yi = np.delete(X, i, 0), np.delete(yy, i)
bi = np.linalg.lstsq(Xi, yi, rcond=None)[0]
si = math.sqrt(((yi - Xi @ bi) ** 2).sum() / 37)
c11 = np.linalg.inv(X.T @ X)[1, 1]
sxx = ((xx - xx.mean()) ** 2).sum()
E.check("s15p10b", abs(c11 - 1 / sxx) < 1e-12, "単回帰で c11 = 1/Sxx")
dfb = (-0.90 - (-0.84)) / (8.0 * math.sqrt(1 / 4000))
E.choice("s15p10b", f"${f(dfb, 3)}$", "定義式")
distinct("s15p10b", 3)
# (c)
n = 40
tbl = {"L": (300.0, 3), "S": (295.6, 4), "Q": (291.0, 6)}
aic = {m: v + 2 * kk for m, (v, kk) in tbl.items()}
aicc = {m: aic[m] + 2 * kk * (kk + 1) / (n - kk - 1) for m, (v, kk) in tbl.items()}
E.choice("s15p10c", f"AICではモデル{min(aic, key=aic.get)}、AICcではモデル{min(aicc, key=aicc.get)}を選ぶ", f"AIC {aic} AICc {aicc}")

# ---- 大問11：ガンマ分布のシミュレーションで平均と標準偏差の比
shape = 2.0
mu0 = 50.0
mu1 = mu0 * math.exp(0.40 + 2 * 0.10)
Y0 = rng.gamma(shape, mu0 / shape, 1_000_000)
Y1 = rng.gamma(shape, mu1 / shape, 1_000_000)
E.check("s15p11a", abs(Y1.mean() / Y0.mean() - 1.8221) < 0.01 and abs(Y1.std() / Y0.std() - 1.8221) < 0.01, "平均・標準偏差の比")
E.choice("s15p11a", f"期待値{f(math.exp(0.6), 2)}倍、標準偏差{f(math.exp(0.6), 2)}倍", "e^0.6")
E.check("s15p11a", f(math.exp(0.6), 4) == "1.8221" and f(math.exp(0.5), 4) == "1.6487" and f(math.sqrt(1.8221), 4) == "1.3499", "e^x の値")

# ---- 大問12：小さな母集団を作り 2 段抽出の推定量の不偏性をシミュレーションで確認し、データから推定値を計算
Mi = rng.integers(20, 100, 40)
Yi = np.array([rng.integers(0, m + 1) for m in Mi])
est = []
for _ in range(20_000):
    sel = rng.choice(40, 4, replace=False)
    tot = 0.0
    for j in sel:
        units = np.r_[np.ones(Yi[j]), np.zeros(Mi[j] - Yi[j])]
        smp = rng.choice(units, 10, replace=False)
        tot += Mi[j] * smp.mean()
    est.append(40 / 4 * tot)
E.check("s15p12a", abs(np.mean(est) - Yi.sum()) / Yi.sum() < 0.01, f"不偏性 {np.mean(est):.1f} vs {Yi.sum()}")
E.num("s15p12a", 40 / 4 * sum(m * x / 10 for m, x in zip([50, 80, 30, 60], [6, 4, 7, 5])), "推定値")

# ---- 大問13：scipy の正規密度で負担率
w1 = 0.7 * stats.norm.pdf(6.4, 5.6, 0.4)
w2 = 0.3 * stats.norm.pdf(6.4, 7.2, 0.8)
E.choice("s15p13a", f"${f(w2 / (w1 + w2), 3)}$", "scipy norm.pdf")
E.check("s15p13a", f(stats.norm.pdf(1), 4) == "0.2420" and f(stats.norm.pdf(2), 4) == "0.0540", "φ の値")
distinct("s15p13a", 3)

# ---- 大問14
# (a) 相関のある因子をシミュレーションして共通部分の分散
F = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 1_000_000)
hs = (0.5 * F[:, 0] + 0.3 * F[:, 1]).var()
E.check("s15p14a", abs(hs - 0.46) < 0.003, f"シミュレーション {hs:.4f}")
E.choice("s15p14a", f"${f(np.array([0.5, 0.3]) @ np.array([[1, 0.4], [0.4, 1]]) @ np.array([0.5, 0.3]), 3)}$", "λ'Φλ")
# (b) 単調回帰（PAVA）で d̂ を求め直してストレス
d = [1.0, 1.4, 1.2, 2.1, 3.2, 2.8]
blocks = [[v] for v in d]
changed = True
while changed:
    changed = False
    for j in range(len(blocks) - 1):
        if np.mean(blocks[j]) > np.mean(blocks[j + 1]):
            blocks[j] = blocks[j] + blocks.pop(j + 1)
            changed = True
            break
dh = [np.mean(bk) for bk in blocks for _ in bk]
E.check("s15p14b", np.allclose(dh, [1.0, 1.3, 1.3, 2.1, 3.0, 3.0]), f"PAVA の結果 {dh}")
st = math.sqrt(sum((a - b) ** 2 for a, b in zip(d, dh)) / sum(a * a for a in d))
E.choice("s15p14b", f"${f(st, 3)}$", "ストレス")
distinct("s15p14b", 3)

E.done()
