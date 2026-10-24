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
# (a) 同時確率表から周辺・条件付き確率
T = np.array([[0.20, 0.05], [0.25, 0.15], [0.15, 0.20]])
px, pr = T.sum(axis=1), T.sum(axis=0)
E.check("s15p02a", abs(T.sum() - 1) < 1e-12, "表の確率の和が 1")
E.check("s15p02a", not np.allclose(T, np.outer(px, pr)), "独立ではない（積と一致しないセルがある）")
E.choice("s15p02a", f"$P(R=1)={f(pr[1], 2)}$、$P(R=1\\midX=2)={f(T[2, 1] / px[2], 3)}$で両者が異なるので", "条件付き確率")
E.check("s15p02a", f((px * np.arange(3)).sum(), 2) != f(np.mean([0, 1, 2]), 2), "値の単純平均の選択肢は誤り")
# (b) 仕組みの分類：(イ) は欠測が観測変数（年齢）だけに依存することをシミュレーションで確認
age = rng.integers(0, 3, 300_000)
hb = rng.normal(7 + 0.0 * age, 1)
miss = rng.random(300_000) < np.array([0.2, 0.375, 0.571])[age]
gap = max(abs(hb[(age == a) & miss].mean() - hb[(age == a) & ~miss].mean()) for a in range(3))
E.check("s15p02b", gap < 0.02, f"年齢を条件にすると欠測と値が無関係（MAR）差 {gap:.4f}")
E.choice("s15p02b", "(ア)MCAR、(イ)MAR、(ウ)MNAR", "定義による分類")

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
# (c) 偏相関と t 値の関係をデータで確認
n = 30
X1 = rng.normal(0, 1, n)
X2 = 0.5 * X1 + rng.normal(0, 1, n)
Y = -0.6 * X1 + 0.3 * X2 + rng.normal(0, 1, n)
Xd = np.column_stack([np.ones(n), X1, X2])
bh, rss = np.linalg.lstsq(Xd, Y, rcond=None)[:2]
s2 = rss[0] / (n - 3)
tval = bh[1] / math.sqrt(s2 * np.linalg.inv(Xd.T @ Xd)[1, 1])
Z = np.column_stack([np.ones(n), X2])
ry = Y - Z @ np.linalg.lstsq(Z, Y, rcond=None)[0]
rx = X1 - Z @ np.linalg.lstsq(Z, X1, rcond=None)[0]
pc = np.corrcoef(ry, rx)[0, 1]
E.check("s15p04c", abs(pc - tval / math.sqrt(tval ** 2 + n - 3)) < 1e-10, "残差どうしの相関 = t/√(t²+ν)")
E.choice("s15p04c", f"${f(-3.2 / math.sqrt(3.2 ** 2 + 27), 3)}$", "t=-3.2, ν=27")
distinct("s15p04c", 3)

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

# ---- 大問6：全列挙で分散
ps = [0.05, 0.10, 0.20, 0.30, 0.35]
from itertools import product  # noqa: E402
dist = {}
for bits in product([0, 1], repeat=5):
    pr_ = np.prod([q if b else 1 - q for q, b in zip(ps, bits)])
    dist[sum(bits)] = dist.get(sum(bits), 0) + pr_
m1 = sum(k * v for k, v in dist.items())
E.num("s15p06a", rnd(sum(k * k * v for k, v in dist.items()) - m1 ** 2, 3), "32 通りの全列挙で分散")

# ---- 大問7
# (a) 帰無仮説のもとでの z（プール）
p = 24 / 300
zz = (20 / 200 - 4 / 100) / math.sqrt(p * (1 - p) * (1 / 200 + 1 / 100))
zu = (20 / 200 - 4 / 100) / math.sqrt(0.1 * 0.9 / 200 + 0.04 * 0.96 / 100)
E.check("s15p07a", 1.645 + 0.1 < zz < 1.96 - 0.1 and zu > 1.96 + 0.1, f"結論が丸めに左右されない（プール {zz:.4f}、非プール {zu:.4f}）")
E.choice("s15p07a", f"$z={f(zz, 2)}$で1.96を超えないので、帰無仮説を棄却しない", "プールした z")
chi2, pv, _, _ = stats.chi2_contingency(np.array([[20, 180], [4, 96]]), correction=False)
E.check("s15p07a", abs(chi2 - zz ** 2) < 1e-9 and pv > 0.05, f"連続修正なしの χ² = z² ({chi2:.4f}, p={pv:.4f})")
# (b) ジェフリーズ：フィッシャー情報量の平方根を事前密度として数値積分
prior = lambda q: math.sqrt(60 / (q * (1 - q)))
lik = lambda q: q ** 2 * (1 - q) ** 58
Zc = integrate.quad(lambda q: prior(q) * lik(q), 0, 1, limit=200)[0]
pm = integrate.quad(lambda q: q * prior(q) * lik(q), 0, 1, limit=200)[0] / Zc
E.num("s15p07b", rnd(pm, 4), f"数値積分の事後平均 {pm:.6f}")
E.check("s15p07b", abs(stats.beta(2.5, 58.5).mean() - pm) < 1e-8, "Be(2.5, 58.5) の平均と一致")

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
Pa = P.copy(); Pa[3] = [0, 0, 0, 1]
E.num("s15p08a", rnd(np.linalg.matrix_power(Pa, 3)[2, 3], 3), "満床を吸収状態にした推移行列の 3 乗")
Xs8 = np.full(100_000, 2); hit = np.zeros(100_000, bool)
for _ in range(3):
    u_ = rng.random(100_000)
    nxt = np.where(u_ < P[Xs8, 3], 3, np.where(u_ < P[Xs8, 3] + P[Xs8, 2], 2, np.where(u_ < P[Xs8, 3] + P[Xs8, 2] + P[Xs8, 1], 1, 0)))
    nxt = np.where(hit, Xs8, nxt); hit |= nxt == 3; Xs8 = nxt
E.check("s15p08a", abs(hit.mean() - 0.56) < 0.005, f"シミュレーション {hit.mean():.4f}")

# ---- 大問9
# (a) 定義にもとづく対応づけ（文章問題）：正解の選択肢がただ1つで、誤答はそれぞれ定義に反する対応を含む
truth = {"(ア)": "局所管理", "(イ)": "無作為化", "(ウ)": "反復"}
for k, ch in enumerate(E.item("s15p09a")["choices"]):
    pairs = dict(re.findall(r"(\([アイウ]\)) (\S+?)(?:、|$)", ch))
    E.check("s15p09a", (pairs == truth) == (k == E.item("s15p09a")["answer"]), f"選択肢 {k}: {pairs}")
import itertools  # noqa: E402
# (b) 全 64 通りの符号で正確な p 値、scipy とも比べる
d = np.array([12, 3, -6, 15, 8, 10])
rk = stats.rankdata(abs(d))
wobs = rk[d > 0].sum()
cnt = sum(1 for sg in itertools.product([0, 1], repeat=6) if sum(r_ for r_, g in zip(range(1, 7), sg) if g) >= wobs)
E.num("s15p09b", rnd(cnt / 64, 3), f"W+={wobs}, {cnt}/64")
E.check("s15p09b", abs(stats.wilcoxon(d, alternative="greater", method="exact").pvalue - cnt / 64) < 1e-12, "scipy wilcoxon exact")

# ---- 大問10
# (a) 実際にデータで DFBETAS の定義を確認したうえで値を計算
xx = rng.uniform(40, 90, 40)
X = np.column_stack([np.ones(40), xx])
c11 = np.linalg.inv(X.T @ X)[1, 1]
sxx = ((xx - xx.mean()) ** 2).sum()
E.check("s15p10a", abs(c11 - 1 / sxx) < 1e-12, "単回帰で c11 = 1/Sxx")
dfb = (-0.90 - (-0.84)) / (8.0 * math.sqrt(1 / 4000))
E.choice("s15p10a", f"${f(dfb, 3)}$", "定義式")
distinct("s15p10a", 3)
# (b) 目的関数を数値最小化してエラスティックネット推定値
from scipy.optimize import minimize_scalar  # noqa: E402
bh = [2.0, -0.6, 0.3]
est = [minimize_scalar(lambda t: 0.5 * (t - b_) ** 2 + 0.5 * abs(t) + 0.5 * 1.0 * t * t, bounds=(-5, 5), method="bounded",
                       options={"xatol": 1e-10}).x for b_ in bh]
txt = "(" + ",\\".join(f(v + 0.0, 2) if abs(v) > 1e-6 else "0.00" for v in est) + ")"
E.choice("s15p10b", txt, f"数値最小化 {np.round(est, 5)}")
obj = lambda bv: sum(0.5 * (t_ - b_) ** 2 + 0.5 * abs(t_) + 0.5 * t_ * t_ for t_, b_ in zip(bv, bh))
vals = [obj([float(x) for x in re.findall(r"-?\d+\.\d+", c)]) for c in E.item("s15p10b")["choices"]]
E.check("s15p10b", int(np.argmin(vals)) == E.item("s15p10b")["answer"] and sorted(vals)[1] - min(vals) > 1e-3,
        f"選択肢のうち目的関数が最小なのは正解だけ {np.round(vals, 4)}")
# (c) シミュレーションで期待二乗予測誤差の分解を確認
y0 = rng.normal(0, 3, 1_000_000)
predA = rng.normal(2.0, 1.0, 1_000_000)
predB = rng.normal(0.5, math.sqrt(4.5), 1_000_000)
eA, eB = ((y0 - predA) ** 2).mean(), ((y0 - predB) ** 2).mean()
E.check("s15p10c", abs(eA - 14.0) < 0.1 and abs(eB - 13.75) < 0.1, f"シミュレーション A {eA:.3f} B {eB:.3f}")
E.choice("s15p10c", f"モデルB（{f(9 + 0.25 + 4.5, 2)}）", "分解の公式")

# ---- 大問11：ガンマ分布のシミュレーションで平均と標準偏差の比
shape = 2.0
mu0 = 50.0
mu1 = mu0 * math.exp(0.40 + 2 * 0.10)
Y0 = rng.gamma(shape, mu0 / shape, 1_000_000)
Y1 = rng.gamma(shape, mu1 / shape, 1_000_000)
E.check("s15p11a", abs(Y1.mean() / Y0.mean() - 1.8221) < 0.01 and abs(Y1.std() / Y0.std() - 1.8221) < 0.01, "平均・標準偏差の比")
E.choice("s15p11a", f"期待値{f(math.exp(0.6), 2)}倍、標準偏差{f(math.exp(0.6), 2)}倍", "e^0.6")
E.check("s15p11a", f(math.exp(0.6), 4) == "1.8221" and f(math.exp(0.5), 4) == "1.6487" and f(math.sqrt(1.8221), 4) == "1.3499", "e^x の値")

# ---- 大問12：費用制約つき最小化を数値的に解いて比を確かめる
from scipy.optimize import minimize  # noqa: E402
W = np.array([0.75, 0.25])
Sg = np.array([10, 12])
cc = np.array([1, 4])
C = 400.0
res = minimize(lambda nn: (W ** 2 * Sg ** 2 / nn).sum(), x0=[100, 50], constraints=[{"type": "eq", "fun": lambda nn: cc @ nn - C}],
               bounds=[(1, 1000), (1, 1000)])
ratio = res.x[1] / res.x.sum()
E.check("s15p12a", abs(ratio - 6000 / 36000) < 1e-3, f"費用制約の最適解の割合 {ratio:.5f}")
E.num("s15p12a", rnd(200 * ratio, 0), "n=200 に換算")

# ---- 大問13：問題文の距離行列を読み取り、scipy の階層的クラスタリングで 2 クラスターに切る
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
Dm = np.array([[0, 2.0, 4.2, 6.6, 9.3], [2.0, 0, 2.2, 4.6, 7.3], [4.2, 2.2, 0, 2.4, 5.1],
               [6.6, 4.6, 2.4, 0, 2.7], [9.3, 7.3, 5.1, 2.7, 0]])


def parts(method):
    lab = fcluster(linkage(squareform(Dm), method=method), 2, criterion="maxclust")
    groups = sorted("".join("ABCDE"[i] for i in range(5) if lab[i] == g) for g in set(lab))
    return " と ".join("$\\{" + ",".join(g_) + "\\}$" for g_ in groups)


rows = [ln for ln in E.items["s15p13a"][0]["context"].splitlines() if re.match(r"\s*\| [A-E] \|", ln)]
shown = np.array([[float(x) for x in ln.strip().strip("|").split("|")[1:]] for ln in rows])
E.check("s15p13a", shown.shape == (5, 5) and np.allclose(shown, Dm), "問題文の距離行列が検算の行列と一致")
E.choice("s15p13a", f"最短距離法：{parts('single')}、最長距離法：{parts('complete')}", "scipy の linkage")
E.check("s15p13a", parts("single") != parts("complete"), "2 つの方法で分け方が異なる（連鎖効果）")

# ---- 大問14
# (a) 行列で一般化最小二乗（バートレット）を計算し、シミュレーションで不偏性を確認
lam = np.array([0.8, 0.7, 0.6])
Psi = np.diag(1 - lam ** 2)
xv = np.array([1.2, 0.4, -0.6])
Pi = np.linalg.inv(Psi)
bart = (lam @ Pi @ xv) / (lam @ Pi @ lam)
E.choice("s15p14a", f"${f(bart, 3)}$", "(λ'Ψ⁻¹λ)⁻¹λ'Ψ⁻¹x")
Fs = 1.5
xs = Fs * lam + rng.normal(0, np.sqrt(1 - lam ** 2), (500_000, 3))
bs = (xs @ Pi @ lam) / (lam @ Pi @ lam)
E.check("s15p14a", abs(bs.mean() - Fs) < 0.01, f"F=1.5 のときのバートレット得点の平均 {bs.mean():.4f}（不偏）")
distinct("s15p14a", 3)
# (b) 正準相関を行列の固有値で（一般の公式 Ruu^-1 Ruv Rvv^-1 Rvu）
Ruv = np.array([[0.5, 0.3], [0.2, 0.4]])
Ruu = Rvv = np.eye(2)
Mx = np.linalg.inv(Ruu) @ Ruv @ np.linalg.inv(Rvv) @ Ruv.T
rho = math.sqrt(max(np.linalg.eigvals(Mx).real))
E.num("s15p14b", rnd(rho, 3), "固有値の平方根")
Rfull = np.block([[Ruu, Ruv], [Ruv.T, Rvv]])
E.check("s15p14b", np.all(np.linalg.eigvalsh(Rfull) > 0), "4 変数の相関行列が正定値")
data = rng.multivariate_normal(np.zeros(4), Rfull, 400_000)
U, V = data[:, :2], data[:, 2:]
from numpy.linalg import svd  # noqa: E402
Cuu, Cvv, Cuv = np.cov(U.T), np.cov(V.T), np.cov(U.T, V.T)[:2, 2:]
iu, iv = np.linalg.inv(np.linalg.cholesky(Cuu)), np.linalg.inv(np.linalg.cholesky(Cvv))
E.check("s15p14b", abs(svd(iu @ Cuv @ iv.T)[1][0] - rho) < 0.005, "シミュレーションデータの正準相関")

E.done()
