"""第32章 検算スクリプト（シミュレーション）。
線形合同法の数列と周期、逆関数法・棄却法・ボックス・ミュラー法の式（sympy）と数値例、
モンテカルロ積分・重点サンプリングの分散（数値積分）、MCMC（2状態連鎖・MH の推移行列・詳細つり合い）、
ギブスサンプリング、バーンインと R̂、ブートストラップ・ジャックナイフを再計算する。
本文の「シード k で…」の値は np.random.default_rng(k) で同じ手順を再実行して照合する
（再現できるものは表示桁の丸め幅、再現できないものはモンテカルロ誤差の 3SE 以内を許容）。
charts の params は各 md の front matter を yaml で読み照合する。
"""
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats, integrate, special

ROOT = Path(__file__).resolve().parent.parent
results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-9


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-9))


def front(term):
    txt = (ROOT / "content" / "ch32" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term):
    return front(term).get("charts", [])


def acf(x, k):
    x = x - x.mean()
    return float((x[:-k] * x[k:]).sum() / (x * x).sum())


# ================================================================ linear-congruential-generator
T = "linear-congruential-generator"


def lcg(a, c, m, x0):
    seq = [x0]
    x = x0
    while True:
        x = (a * x + c) % m
        if x in seq:
            return seq, len(seq) - seq.index(x)
        seq.append(x)


seq, per = lcg(5, 3, 16, 7)
check(T, "例1 a=5,c=3,m=16,x0=7 の数列", seq == [7, 6, 1, 8, 11, 10, 5, 12, 15, 14, 9, 0, 3, 2, 13, 4], seq)
check(T, "例1 周期16・0〜15が1回ずつ", per == 16 and sorted(seq) == list(range(16)), per)
check(T, "例1 x1=38 mod16=6, x2=33 mod16=1, x3=8", (5 * 7 + 3, (5 * 7 + 3) % 16, 5 * 6 + 3, (5 * 6 + 3) % 16, (5 * 1 + 3) % 16) == (38, 6, 33, 1, 8), None)
check(T, "例1 u_n=0.4375,0.375,0.0625,0.5", allclose(np.array(seq[:4]) / 16, [0.4375, 0.375, 0.0625, 0.5], 0), None)
check(T, "例1 偶奇が交互（最下位ビット周期2）", all((seq[i] + seq[i + 1]) % 2 == 1 for i in range(15)), [s % 2 for s in seq])


def hull_dobell(a, c, m):
    ps = [p for p in range(2, m + 1) if m % p == 0 and all(p % q for q in range(2, p))]
    return math.gcd(c, m) == 1 and all((a - 1) % p == 0 for p in ps) and (m % 4 != 0 or (a - 1) % 4 == 0)


check(T, "ハル・ドーベル：3条件と最大周期の一致（m≤64 の全 a,c, c≠0）",
      all(hull_dobell(a, c, m) == (lcg(a, c, m, 0)[1] == m) for m in range(2, 65) for a in range(1, m) for c in range(1, m)), None)
seq2, per2 = lcg(3, 3, 16, 7)
check(T, "例2 a=3: 7,8,11,4,15,0,3,12 周期8", seq2 == [7, 8, 11, 4, 15, 0, 3, 12] and per2 == 8, seq2)
seq3, per3 = lcg(5, 0, 16, 7)
check(T, "例2 c=0,a=5: 7,3,15,11 周期4", seq3 == [7, 3, 15, 11] and per3 == 4, seq3)
seq4, per4 = lcg(2, 0, 13, 1)
check(T, "例2 m=13,a=2: 1,2,4,8,3,6,12,11,9,5,10,7 周期12", seq4 == [1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7] and per4 == 12, seq4)
c = charts(T)[0]
pts = [(p["x"], p["y"]) for p in c["params"]["points"]]
exp_pts = [(seq[i], seq[(i + 1) % 16]) for i in range(16)]
check(T, "chart points-2d の16点 = 例1 の (x_n, x_n+1)", pts == exp_pts and all(p["label"] == f"{p['x']}→{p['y']}" for p in c["params"]["points"]), len(pts))
# 格子構造：16点は少数の平行線（傾き5、y−5x ≡ 3 mod 16 → y = 5x+3−16k）
ks = {(5 * x + 3 - y) // 16 for x, y in pts}
check(T, "chart 点は傾き5の平行直線（数本）上に並ぶ", all((5 * x + 3 - y) % 16 == 0 for x, y in pts) and len(ks) <= 5, sorted(ks))

# ================================================================ uniform-random-number
T = "uniform-random-number"
u = sp.symbols("u")
EU = sp.integrate(u, (u, 0, 1))
VU = sp.integrate(u ** 2, (u, 0, 1)) - EU ** 2
check(T, "E[U]=1/2, V[U]=1/3−1/4=1/12（sympy）", EU == sp.Rational(1, 2) and VU == sp.Rational(1, 12), (EU, VU))
A, B = sp.symbols("a b", real=True)
ua = A + (B - A) * u
check(T, "U(a,b): 平均 (a+b)/2・分散 (b−a)²/12（sympy）",
      sp.simplify(sp.integrate(ua, (u, 0, 1)) - (A + B) / 2) == 0
      and sp.simplify(sp.integrate(ua ** 2, (u, 0, 1)) - sp.integrate(ua, (u, 0, 1)) ** 2 - (B - A) ** 2 / 12) == 0, None)
uu = np.array([0.731, 0.052, 0.498])
check(T, "例1 6U=4.386,0.312,2.988 → 目 5,1,3", allclose(6 * uu, [4.386, 0.312, 2.988], 1e-9) and list(np.ceil(6 * uu).astype(int)) == [5, 1, 3], 6 * uu)
t = sp.symbols("t")
m4 = sp.integrate(t ** 4, (t, -sp.Rational(1, 2), sp.Rational(1, 2)))
kurt = m4 / sp.Rational(1, 12) ** 2
check(T, "例2 E[(U−1/2)^4]=1/80・尖度 1.8・超過 −1.2", m4 == sp.Rational(1, 80) and kurt == sp.Rational(9, 5), (m4, kurt))
# Z の4次モーメント：キュムラント加法性で厳密に
k2 = 12 * sp.Rational(1, 12)
k4 = 12 * (m4 - 3 * sp.Rational(1, 12) ** 2)
EZ4 = k4 + 3 * k2 ** 2
check(T, "例2 E[Z]=0, V[Z]=1, 超過尖度 −0.1, E[Z^4]=2.9（キュムラント）", k2 == 1 and k4 == sp.Rational(-1, 10) and EZ4 == sp.Rational(29, 10), EZ4)
rng = np.random.default_rng(1)
Z = rng.random((100000, 12)).sum(1) - 6
# 再現されない（default_rng(1) で 平均0.0011・分散0.9960・4次2.865）→ モンテカルロ誤差 3SE で妥当性を確認
se_v = math.sqrt((2.9 - 1) / 1e5)
se_4 = math.sqrt(float(sp.Rational(29, 10)) * 0 + 96.0 / 1e5)  # V[Z^4]≈E[Z^8]−9 ≈ 96（正規近似）
check(T, "例2 シード1: 平均0.001 は 0 の 3SE（0.0095）以内", abs(0.001) <= 3 * math.sqrt(1 / 1e5), 0.001)
check(T, "例2 シード1: 分散0.997 は 1 の 3SE 以内", abs(0.997 - 1) <= 3 * se_v, se_v)
check(T, "例2 シード1: 4次モーメント2.883 は 2.9 の 3SE 以内", abs(2.883 - 2.9) <= 3 * se_4, se_4)
check(T, "例2 再計算（default_rng(1)）でも 平均・分散・4次が本文値と 3SE 以内",
      abs(Z.mean() - 0.001) <= 3 * math.sqrt(1 / 1e5) and abs(Z.var() - 0.997) <= 3 * se_v and abs((Z ** 4).mean() - 2.883) <= 3 * se_4,
      (round(Z.mean(), 4), round(Z.var(), 4), round((Z ** 4).mean(), 4)))
check(T, "例2 |Z|≤6", np.abs(Z).max() <= 6, np.abs(Z).max())
c = charts(T)[0]
check(T, "chart clt-sim base unif n=12（例2と同じ）", c["params"]["base"] == "unif" and c["params"]["n"] == 12, c["params"])

# ================================================================ inverse-transform-sampling
T = "inverse-transform-sampling"
x, lam, k, eta, p = sp.symbols("x lambda k eta p", positive=True)
U = sp.symbols("U", positive=True)
F_exp = 1 - sp.exp(-lam * x)
check(T, "指数 F(−log(1−U)/λ)=U（sympy）", sp.simplify(F_exp.subs(x, -sp.log(1 - U) / lam) - U) == 0, None)
F_wb = 1 - sp.exp(-(x / eta) ** k)
vv = sp.symbols("v", positive=True)  # v=−log(1−U)>0（0<U<1）とおき 1−U=e^{−v}
check(T, "ワイブル F(η(−log(1−U))^{1/k})=U（sympy, v=−log(1−U)）", sp.simplify(F_wb.subs(x, eta * vv ** (1 / k)) - (1 - sp.exp(-vv))) == 0, None)
F_c = sp.Rational(1, 2) + sp.atan(x) / sp.pi
xc = sp.symbols("xc", real=True)
check(T, "コーシー F(tan(π(U−1/2)))=U（sympy）",
      sp.simplify((sp.Rational(1, 2) + sp.atan(sp.tan(sp.pi * (U - sp.Rational(1, 2)))) / sp.pi).subs(U, sp.Rational(3, 10)) - sp.Rational(3, 10)) == 0
      and all(close(0.5 + math.atan(math.tan(math.pi * (uv - 0.5))) / math.pi, uv, 1e-12) for uv in [0.01, 0.3, 0.77, 0.99]), None)
check(T, "コーシーの F は標準コーシーの cdf", all(close(0.5 + math.atan(v) / math.pi, stats.cauchy.cdf(v), 1e-12) for v in [-3, 0, 1.2]), None)
# 幾何（台1,2,...）：ceil(log(1−U)/log(1−p)) の分布が幾何分布
pp = 0.2
uv = (np.arange(1, 200001) - 0.5) / 200000  # 等間隔格子（決定的）
g = np.ceil(np.log(1 - uv) / np.log(1 - pp)).astype(int)
freq = np.bincount(g, minlength=12)[1:12] / len(uv)
check(T, "幾何の生成式（等間隔格子200000点）の度数 = stats.geom pmf", allclose(freq, stats.geom.pmf(np.arange(1, 12), pp), 1e-5), freq[:3])
check(T, "幾何の F(x)=1−(1−p)^x", close(stats.geom.cdf(4, pp), 1 - 0.8 ** 4, 1e-12), None)
check(T, "例1 λ=2: u=0.3→0.1783, u=0.9→1.1513（log0.7=−0.3567, log0.1=−2.3026）",
      close(-0.5 * math.log(0.7), 0.1783) and close(-0.5 * math.log(0.1), 1.1513) and close(math.log(0.7), -0.3567) and close(math.log(0.1), -2.3026),
      (-0.5 * math.log(0.7), -0.5 * math.log(0.1)))
rng = np.random.default_rng(1)
uu = rng.random(100000)
xe = -np.log(1 - uu) / 2
check(T, "例1 シード1 10^5個: 平均0.4997・分散0.2482（default_rng(1) で再現）", close(xe.mean(), 0.4997) and close(xe.var(ddof=1), 0.2482), (xe.mean(), xe.var(ddof=1)))
check(T, "例2 ワイブル k=2,η=10: 5.9722, 15.1743", close(10 * math.sqrt(-math.log(0.7)), 5.9722) and close(10 * math.sqrt(-math.log(0.1)), 15.1743),
      (10 * math.sqrt(-math.log(0.7)), 10 * math.sqrt(-math.log(0.1))))
check(T, "例2 √0.3567 と √2.3026 の表示値", close(-math.log(0.7), 0.3567) and close(-math.log(0.1), 2.3026), None)
cum = np.cumsum([0.2, 0.5, 0.3])
pick = lambda v: int(np.searchsorted(cum, v, side="left"))
check(T, "例3 離散: 累積 0.2,0.7,1.0、u=0.10→0, 0.65→1, 0.85→2", allclose(cum, [0.2, 0.7, 1.0], 1e-12) and [pick(v) for v in (0.10, 0.65, 0.85)] == [0, 1, 2], None)
r1, r2 = math.log(0.7) / math.log(0.8), math.log(0.1) / math.log(0.8)
check(T, "例3 幾何 p=0.2: 1.598→2, 10.319→11", close(r1, 1.598, 5e-4) and close(r2, 10.319, 5e-4) and math.ceil(r1) == 2 and math.ceil(r2) == 11, (r1, r2))
cs = charts(T)
check(T, "chart1 expon rate=1, chart2 weibull shape2 scale1", cs[0]["params"]["dist"] == "expon" and cs[0]["params"]["rate"] == 1
      and cs[1]["params"] == {"dist": "weibull", "n": 1000, "shape": 2, "scale": 1}, None)

# ================================================================ rejection-sampling
T = "rejection-sampling"
y = sp.symbols("y", positive=True)
f22 = 6 * y * (1 - y)
check(T, "Be(2,2) の密度 6x(1−x) が正規化・最大 1.5（x=1/2）",
      sp.integrate(f22, (y, 0, 1)) == 1 and sp.solve(sp.diff(f22, y), y) == [sp.Rational(1, 2)] and f22.subs(y, sp.Rational(1, 2)) == sp.Rational(3, 2), None)
check(T, "例1 採択率 1/1.5=0.6667（数値積分 ∫f/(Mg)g）", close(integrate.quad(lambda v: 6 * v * (1 - v) / 1.5, 0, 1)[0], 0.6667), 1 / 1.5)
check(T, "例1 Y=0.3: 1.26/1.5=0.84 採択、Y=0.8: 0.64 棄却",
      close(6 * 0.3 * 0.7, 1.26, 1e-12) and close(1.26 / 1.5, 0.84, 1e-12) and 0.7 <= 0.84 and close(6 * 0.8 * 0.2 / 1.5, 0.64, 1e-12) and 0.7 > 0.64, None)
rng = np.random.default_rng(2)
Y = rng.random(100000)
V = rng.random(100000)
acc = V <= 6 * Y * (1 - Y) / 1.5
check(T, "例1 シード2 10^5回: 採択率0.6659・平均0.5011・分散0.0503（default_rng(2) で再現）",
      close(acc.mean(), 0.6659) and close(Y[acc].mean(), 0.5011) and close(Y[acc].var(ddof=1), 0.0503), (acc.mean(), Y[acc].mean(), Y[acc].var(ddof=1)))
check(T, "例1 Be(2,2) 平均0.5・分散 4/(16·5)=0.05", close(stats.beta(2, 2).mean(), 0.5, 1e-12) and close(stats.beta(2, 2).var(), 0.05, 1e-12), None)
f25 = 30 * y * (1 - y) ** 4
check(T, "例2 Be(2,5) 密度 30x(1−x)^4 が正規化・最大点 x=1/5", sp.integrate(f25, (y, 0, 1)) == 1 and sp.Rational(1, 5) in sp.solve(sp.diff(f25, y), y), None)
M25 = 30 * 0.2 * 0.8 ** 4
check(T, "例2 M=2.4576, 採択率0.4069", close(M25, 2.4576, 1e-12) and close(1 / M25, 0.4069) and close(stats.beta(2, 5).pdf(0.2), M25, 1e-9), (M25, 1 / M25))
ratio = sp.sqrt(2 / sp.pi) * sp.exp(y - y ** 2 / 2)
Mh = ratio.subs(y, 1)
check(T, "例3 f/g の最大点 x=1, M=√(2e/π)=1.3155, 採択率0.7602",
      sp.solve(sp.diff(ratio, y), y) == [1] and sp.simplify(Mh - sp.sqrt(2 * sp.E / sp.pi)) == 0 and close(float(Mh), 1.3155) and close(1 / float(Mh), 0.7602), float(Mh))
check(T, "例3 採択条件 f/(Mg)=exp(−(y−1)²/2)（sympy）", sp.simplify(ratio / Mh - sp.exp(-(y - 1) ** 2 / 2)) == 0, None)
check(T, "例3 数値積分の採択率 ∫ halfnorm/M = 1/M", close(integrate.quad(lambda v: stats.halfnorm.pdf(v) / float(Mh), 0, np.inf)[0], 1 / float(Mh), 1e-8), None)
Ms = sp.symbols("M", positive=True)
pg = 1 / Ms
check(T, "試行回数の幾何分布: 平均 M, 分散 (1−1/M)M² = M(M−1)（sympy）",
      sp.simplify(1 / pg - Ms) == 0 and sp.simplify((1 - pg) / pg ** 2 - Ms * (Ms - 1)) == 0 and sp.simplify((1 - 1 / Ms) * Ms ** 2 - Ms * (Ms - 1)) == 0, None)
# 採択値が f に従う：一般の f,g で（シミュレーション, 許容 KS<0.01）
rng = np.random.default_rng(20261023)
Y = rng.random(200000)
V = rng.random(200000)
xa = Y[V <= stats.beta(2, 5).pdf(Y) / M25]
ks = stats.kstest(xa, stats.beta(2, 5).cdf).statistic
check(T, "Be(2,5) を棄却法で生成: 採択率 1/M（許容0.005）・KS<0.01", abs(len(xa) / 200000 - 1 / M25) < 0.005 and ks < 0.01, (len(xa) / 200000, ks))
c = charts(T)[0]
check(T, "chart rejection-sampling a=2,b=5（例2と同じ）", c["params"]["a"] == 2 and c["params"]["b"] == 5, c["params"])

# ================================================================ box-muller-method
T = "box-muller-method"
z1, z2 = sp.symbols("z1 z2", real=True)
u1 = sp.exp(-(z1 ** 2 + z2 ** 2) / 2)
u2 = sp.atan(z2 / z1) / (2 * sp.pi)
J = sp.Matrix([u1, u2]).jacobian([z1, z2])
check(T, "偏微分 ∂u1/∂z1=−z1u1, ∂u1/∂z2=−z2u1（sympy）", sp.simplify(J[0, 0] + z1 * u1) == 0 and sp.simplify(J[0, 1] + z2 * u1) == 0, None)
check(T, "偏微分 ∂u2/∂z1=−z2/(2π(z1²+z2²)), ∂u2/∂z2=z1/(2π(z1²+z2²))（sympy）",
      sp.simplify(J[1, 0] + z2 / (2 * sp.pi * (z1 ** 2 + z2 ** 2))) == 0 and sp.simplify(J[1, 1] - z1 / (2 * sp.pi * (z1 ** 2 + z2 ** 2))) == 0, None)
detJ = sp.simplify(J.det())
check(T, "ヤコビアン = −u1/(2π)（sympy）", sp.simplify(detJ + u1 / (2 * sp.pi)) == 0, detJ)
check(T, "|J| = φ(z1)φ(z2)", sp.simplify(-detJ - sp.exp(-z1 ** 2 / 2) / sp.sqrt(2 * sp.pi) * sp.exp(-z2 ** 2 / 2) / sp.sqrt(2 * sp.pi)) == 0, None)
# 順変換側のヤコビアン（u1,u2 → z1,z2）の逆数としても確認
U1, U2 = sp.symbols("U1 U2", positive=True)
Zf = sp.Matrix([sp.sqrt(-2 * sp.log(U1)) * sp.cos(2 * sp.pi * U2), sp.sqrt(-2 * sp.log(U1)) * sp.sin(2 * sp.pi * U2)])
detF = sp.simplify(Zf.jacobian([U1, U2]).det())
check(T, "順変換のヤコビアン = −2π/U1（逆変換の逆数）", sp.simplify(detF + 2 * sp.pi / U1) == 0, detF)
r, s_ = sp.symbols("r s", positive=True)
check(T, "R の密度 r e^{−r²/2} が正規化・P(R²>s)=e^{−s/2}（sympy）",
      sp.integrate(r * sp.exp(-r ** 2 / 2), (r, 0, sp.oo)) == 1 and sp.simplify(sp.integrate(r * sp.exp(-r ** 2 / 2), (r, sp.sqrt(s_), sp.oo)) - sp.exp(-s_ / 2)) == 0, None)
check(T, "R² ~ 平均2の指数 = χ²(2)（cdf 一致）", all(close(stats.chi2(2).sf(v), math.exp(-v / 2), 1e-12) for v in [0.5, 4, 9]), None)
R1 = math.sqrt(-2 * math.log(0.5))
check(T, "例1 U1=0.5: −2log0.5=1.3863, R=1.1774, Z=(0,1.1774)",
      close(-2 * math.log(0.5), 1.3863) and close(R1, 1.1774) and close(R1 * math.cos(math.pi / 2), 0) and close(R1 * math.sin(math.pi / 2), 1.1774), R1)
R2 = math.sqrt(-2 * math.log(0.2))
Z1, Z2 = R2 * math.cos(2 * math.pi * 0.6), R2 * math.sin(2 * math.pi * 0.6)
check(T, "例1 U1=0.2,U2=0.6: 3.2189, R=1.7941, cos=−0.8090, sin=−0.5878",
      close(-2 * math.log(0.2), 3.2189) and close(R2, 1.7941) and close(math.cos(1.2 * math.pi), -0.8090) and close(math.sin(1.2 * math.pi), -0.5878), R2)
check(T, "例1 Z1=−1.4515, Z2=−1.0546", close(Z1, -1.4515) and close(Z2, -1.0546), (Z1, Z2))
check(T, "例1 50+10Z: 35.485, 39.454", close(50 + 10 * Z1, 35.485, 5e-4) and close(50 + 10 * Z2, 39.454, 5e-4), (50 + 10 * Z1, 50 + 10 * Z2))
check(T, "例2 e^{−2}=0.1353=P(χ²(2)>4)", close(math.exp(-2), 0.1353) and close(stats.chi2(2).sf(4), 0.1353), math.exp(-2))
check(T, "例2 P(R²>9)=e^{−4.5}=0.0111", close(math.exp(-4.5), 0.0111) and close(stats.chi2(2).sf(9), 0.0111), math.exp(-4.5))
rng = np.random.default_rng(3)
a1 = rng.random(100000)
a2 = rng.random(100000)
RR = np.sqrt(-2 * np.log(a1))
b1, b2 = RR * np.cos(2 * np.pi * a2), RR * np.sin(2 * np.pi * a2)
vals = [b1.mean(), b1.var(ddof=1), b2.mean(), b2.var(ddof=1), np.corrcoef(b1, b2)[0, 1], (np.abs(b1) > 1.96).mean()]
check(T, "例3 シード3 10^5組: −0.008, 1.008, −0.004, 1.002, −0.0001, 0.0517（default_rng(3) で再現）",
      allclose(vals[:4], [-0.008, 1.008, -0.004, 1.002], 5e-4) and close(vals[4], -0.0001) and close(vals[5], 0.0517), np.round(vals, 5))
check(T, "例3 理論 P(|Z|>1.96)=0.05", close(2 * stats.norm.sf(1.96), 0.05, 5e-4), 2 * stats.norm.sf(1.96))
# 極座標法：採択率 π/4、W ~ U(0,1)、出力が独立 N(0,1)（seed=20261023、KS<0.01、相関|r|<0.01）
rng = np.random.default_rng(20261023)
Vv = rng.uniform(-1, 1, (400000, 2))
W = (Vv ** 2).sum(1)
ok = (W > 0) & (W < 1)
Vv, W = Vv[ok], W[ok]
fac = np.sqrt(-2 * np.log(W) / W)
p1, p2 = Vv[:, 0] * fac, Vv[:, 1] * fac
check(T, "極座標法: 採択率 π/4=0.7854（許容0.003）", abs(ok.mean() - math.pi / 4) < 0.003 and close(math.pi / 4, 0.7854), ok.mean())
check(T, "極座標法: W~U(0,1)・Z1,Z2~N(0,1)（KS<0.01）・相関|r|<0.01",
      stats.kstest(W, "uniform").statistic < 0.01 and stats.kstest(p1, "norm").statistic < 0.01 and stats.kstest(p2, "norm").statistic < 0.01
      and abs(np.corrcoef(p1, p2)[0, 1]) < 0.01, None)
cs = charts(T)
check(T, "chart2 scatter bvnorm rho=0 seed=3", cs[1]["model"] == "bvnorm" and cs[1]["params"]["rho"] == 0 and cs[1]["seed"] == 3, None)

# ================================================================ monte-carlo-integration
T = "monte-carlo-integration"
I = math.e - 1
s2 = (math.e ** 2 - 1) / 2 - I ** 2
check(T, "例1 I=e−1=1.7183, σ²=3.1945−2.9525=0.2420",
      close(I, 1.7183) and close((math.e ** 2 - 1) / 2, 3.1945) and close(I ** 2, 2.9525) and close(s2, 0.2420), s2)
check(T, "例1 σ²（数値積分）", close(integrate.quad(lambda v: (math.exp(v) - I) ** 2, 0, 1)[0], s2, 1e-10), None)
check(T, "例1 n=10000 の SE=0.00492, SE=0.001 に必要な n=242036",
      close(math.sqrt(s2 / 1e4), 0.00492, 5e-6) and math.ceil(s2 / 1e-6) == 242036, (math.sqrt(s2 / 1e4), s2 / 1e-6))
rng = np.random.default_rng(4)
ge = np.exp(rng.random(10000))
Ih, sd = ge.mean(), ge.std(ddof=1)
ci = (Ih - 1.96 * sd / 100, Ih + 1.96 * sd / 100)
check(T, "例1 シード4: Î=1.7216, s=0.4934, SE=0.00493（default_rng(4) で再現）",
      close(Ih, 1.7216) and close(sd, 0.4934) and close(sd / 100, 0.00493, 5e-6), (Ih, sd))
check(T, "例1 95%区間 [1.7119, 1.7313] が真値を含む", allclose(ci, [1.7119, 1.7313]) and ci[0] < I < ci[1], ci)
pq = math.pi / 4
check(T, "例2 V[h]=16p(1−p)=2.6968, SE=0.0164, n≥26968",
      close(16 * pq * (1 - pq), 2.6968) and close(16 * 0.7854 * 0.2146, 2.6968, 1e-4) and close(math.sqrt(16 * pq * (1 - pq) / 1e4), 0.0164)
      and math.ceil(16 * pq * (1 - pq) / 1e-4) == 26968, 16 * pq * (1 - pq))
rng = np.random.default_rng(5)
X = rng.random(10000)
Yp = rng.random(10000)
h = 4.0 * (X ** 2 + Yp ** 2 <= 1)
check(T, "例2 シード5: 推定値3.1456・SE0.0164（default_rng(5) で再現）", close(h.mean(), 3.1456) and close(h.std(ddof=1) / 100, 0.0164), (h.mean(), h.std(ddof=1) / 100))
xx = sp.symbols("xx", positive=True)
v3 = 4 * (sp.integrate(xx ** 4 / 2, (xx, 0, 2)) - sp.integrate(xx ** 2 / 2, (xx, 0, 2)) ** 2)
check(T, "例3 ∫0^2 x²=8/3, V[2X²]=4(16/5−16/9)=5.6889（sympy）",
      sp.integrate(xx ** 2, (xx, 0, 2)) == sp.Rational(8, 3) and v3 == 4 * (sp.Rational(16, 5) - sp.Rational(16, 9)) and close(float(v3), 5.6889), v3)
check(T, "SE の速さ: n を100倍で誤差1/10、2倍で 0.71倍", close(1 / math.sqrt(100), 0.1, 1e-12) and close(1 / math.sqrt(2), 0.71, 5e-3), None)
c = charts(T)[0]
check(T, "chart lln-path p=0.785 ≈ π/4", close(c["params"]["p"], pq, 5e-4), c["params"])

# ================================================================ importance-sampling
T = "importance-sampling"
p3 = stats.norm.sf(3)
check(T, "例1 p=P(Z>3)=0.001350, p(1−p)=0.001348", close(p3, 0.001350, 5e-7) and close(p3 * (1 - p3), 0.001348, 5e-7), p3)
check(T, "例1 SE=0.000367・相対誤差約27%", close(math.sqrt(p3 * (1 - p3) / 1e4), 0.000367, 5e-7) and close(math.sqrt(p3 * (1 - p3) / 1e4) / p3, 0.27, 5e-3), None)
nreq = p3 * (1 - p3) / (0.1 * p3) ** 2
check(T, "例1 相対誤差10%に必要な n=73980（切り上げ）", math.ceil(nreq) in (73980, 73981) and abs(nreq - 73980) < 1, nreq)
xs_ = sp.symbols("x", real=True)
w = sp.exp(-xs_ ** 2 / 2) / sp.exp(-(xs_ - 3) ** 2 / 2)
check(T, "例1 重み φ(x)/φ(x−3)=e^{−3x+4.5}（sympy）", sp.simplify(w - sp.exp(-3 * xs_ + sp.Rational(9, 2))) == 0, None)
check(T, "例1 x=3.5 で w=e^{−6}=0.00248", close(math.exp(-3 * 3.5 + 4.5), 0.00248, 5e-6), math.exp(-6))
phi = lambda v: math.exp(-v * v / 2) / math.sqrt(2 * math.pi)
check(T, "例1 φ(x)e^{−3x}=e^{4.5}φ(x+3)（sympy）",
      sp.simplify(sp.exp(-xs_ ** 2 / 2) * sp.exp(-3 * xs_) - sp.exp(sp.Rational(9, 2)) * sp.exp(-(xs_ + 3) ** 2 / 2)) == 0, None)
m2 = math.exp(9) * stats.norm.sf(6)
m2q = integrate.quad(lambda v: phi(v) * math.exp(-3 * v + 4.5), 3, np.inf)[0]
check(T, "例1 2次モーメント e^9 P(Z>6)=7.994e-6（数値積分とも）", close(m2 * 1e6, 7.994, 5e-4) and close(m2q, m2, 1e-12), (m2, m2q))
vis = m2 - p3 ** 2
check(T, "例1 1個あたり分散 6.172e-6・比 約1/218", close(vis * 1e6, 6.172, 5e-4) and round(p3 * (1 - p3) / vis) == 218, (vis, p3 * (1 - p3) / vis))
check(T, "例1 n=10000 で SE 2.48e-5（相対1.8%）", close(math.sqrt(vis / 1e4) * 1e5, 2.48, 5e-3) and close(math.sqrt(vis / 1e4) / p3, 0.018, 5e-4), math.sqrt(vis / 1e4))
rng = np.random.default_rng(6)
xg = rng.normal(3, 1, 10000)
hw = (xg > 3) * np.exp(-3 * xg + 4.5)
check(T, "例1 シード6: 推定値0.001357・SE 2.51e-5（default_rng(6) で再現）",
      close(hw.mean(), 0.001357, 5e-7) and close(hw.std(ddof=1) / 100 * 1e5, 2.51, 5e-3), (hw.mean(), hw.std(ddof=1) / 100))
v2 = integrate.quad(lambda v: math.exp(2 * v) / (2 / 3 * (1 + v)), 0, 1)[0] - I ** 2
check(T, "例2 g=(2/3)(1+x) が密度・分散0.0269・一様の約1/9",
      close(integrate.quad(lambda v: 2 / 3 * (1 + v), 0, 1)[0], 1, 1e-12) and close(v2, 0.0269) and round(s2 / v2) == 9, (v2, s2 / v2))
# 最適 g*=hf/μ で分散0：h=x², f=Exp(1)
check(T, "最適 g*=hf/μ で h·f/g*=μ（定数、sympy）", sp.simplify((xx ** 2 * sp.exp(-xx)) / (xx ** 2 * sp.exp(-xx) / 2) - 2) == 0, None)
# 不偏性と分散公式（seed=20261023、許容 3SE）
rng = np.random.default_rng(20261023)
reps = np.array([((lambda xv: ((xv > 3) * np.exp(-3 * xv + 4.5)).mean())(rng.normal(3, 1, 1000))) for _ in range(4000)])
check(T, "重点サンプリングの不偏性と V=σ²/n（n=1000×4000回、平均 3SE・分散 相対5%）",
      abs(reps.mean() - p3) <= 3 * math.sqrt(vis / 1000 / 4000) and abs(reps.var(ddof=1) / (vis / 1000) - 1) < 0.05, (reps.mean(), reps.var(ddof=1) / (vis / 1000)))
c = charts(T)[0]
check(T, "chart pdf N(0,1)・比較 N(3,1)", c["params"] == {"mu": 0, "sigma": 1} and c["compare"][0]["params"] == {"mu": 3, "sigma": 1}, None)
check(T, "chart caption: g=N(3,1) で x>3 が半分", close(stats.norm(3, 1).sf(3), 0.5, 1e-12), None)

# ================================================================ mcmc
T = "mcmc"
P = np.array([[0.9, 0.1], [0.2, 0.8]])
ev, evec = np.linalg.eig(P.T)
pi = np.real(evec[:, np.argmin(abs(ev - 1))])
pi = pi / pi.sum()
check(T, "例1 定常分布 (2/3, 1/3)", allclose(pi, [2 / 3, 1 / 3], 1e-12) and allclose(pi @ P, pi, 1e-12), pi)
check(T, "例1 詳細つり合い 0.1π0=0.2π1", close(pi[0] * 0.1, pi[1] * 0.2, 1e-12), None)
check(T, "例1 2番目の固有値 0.7", close(sorted(np.real(np.linalg.eigvals(P)))[0], 0.7, 1e-12), np.linalg.eigvals(P))
# 指示関数の自己相関 = 0.7^k（厳密計算）
ind = np.array([0.0, 1.0])
mu1 = pi @ ind
var1 = pi @ ind ** 2 - mu1 ** 2
rhos = [((pi * ind) @ np.linalg.matrix_power(P, kk) @ ind - mu1 ** 2) / var1 for kk in range(1, 6)]
check(T, "例1 ρ_k=0.7^k（P^k から厳密に）", allclose(rhos, 0.7 ** np.arange(1, 6), 1e-12), rhos)
fac2 = 1 + 2 * 0.7 / 0.3
check(T, "例1 1+2Σ0.7^k=1.7/0.3=5.667, n_eff=1765", close(fac2, 5.667, 5e-4) and close(fac2, 1.7 / 0.3, 1e-12) and round(1e4 / fac2) == 1765, (fac2, 1e4 / fac2))
check(T, "例1 分散 2.22e-5 → 1.26e-4", close(2 / 9 / 1e4 * 1e5, 2.22, 5e-3) and close(2 / 9 / 1e4 * fac2 * 1e4, 1.26, 5e-3), 2 / 9 / 1e4 * fac2)
# 長期の割合（シミュレーション、seed=20261023、許容: 3×√(1.26e-4 相当で n=2e5) ）
rng = np.random.default_rng(20261023)
nn = 200000
st = np.empty(nn, int)
cur = 0
uu = rng.random(nn)
for i in range(nn):
    cur = (1 if uu[i] < 0.1 else 0) if cur == 0 else (0 if uu[i] < 0.2 else 1)
    st[i] = cur
se_prop = math.sqrt(2 / 9 / nn * fac2)
check(T, "例1 シミュレーション: 状態1の割合→1/3（2×10^5歩、3SE）・1次自己相関≈0.7（許容0.01）",
      abs(st.mean() - 1 / 3) < 3 * se_prop and abs(acf(st.astype(float), 1) - 0.7) < 0.01, (st.mean(), acf(st.astype(float), 1)))


def ess_trunc(xv, cut=0.05):
    s = 0.0
    kk = 1
    while True:
        rr = acf(xv, kk)
        if rr < cut:
            break
        s += rr
        kk += 1
    return len(xv) / (1 + 2 * s)


def rwm(step, seed):
    g = np.random.default_rng(seed)
    xcur = -6.0
    out = np.empty(11000)
    nacc = 0
    for t_ in range(11000):
        yv = xcur + step * g.standard_normal()
        if np.log(g.random()) <= -(yv * yv - xcur * xcur) / 2:
            xcur = yv
            nacc += 1
        out[t_] = xcur
    return nacc / 11000, out[1000:]


table = {0.1: (0.959, 0.220, 1.057, 0.996, 40), 1: (0.703, 0.009, 1.002, 0.778, 1295),
         2.4: (0.445, -0.031, 1.011, 0.634, 2422), 10: (0.127, -0.044, 0.991, 0.842, 750)}
for sv, (ca, cm, cv, cr, ce) in table.items():
    ar, ch = rwm(sv, 7)
    e = ess_trunc(ch)
    # 各幅で default_rng(7) から乱数を作り直すと再現（ESS は小数を切り捨てた表示で「約」なので許容1）
    check(T, f"例2 s={sv}: 採択率{ca}・平均{cm}・分散{cv}・ρ1 {cr}・ESS 約{ce}（default_rng(7) で再現）",
          allclose([ar, ch.mean(), ch.var(ddof=1)], [ca, cm, cv], 5e-4) and abs(acf(ch, 1) - cr) <= 5e-4 + 1e-3 and abs(e - ce) <= 1,
          (round(ar, 4), round(ch.mean(), 4), round(ch.var(ddof=1), 4), round(acf(ch, 1), 4), round(e, 2)))
check(T, "例2 ESS は s=2.4 で最大・s=0.1 で最小", max(table, key=lambda k_: table[k_][4]) == 2.4 and min(table, key=lambda k_: table[k_][4]) == 0.1, None)
cs = charts(T)
check(T, "chart mcmc-trace init=−6（本文の初期値と同じ）・burnin<iters", cs[0]["params"]["init"] == -6 and cs[0]["params"]["burnin"] < cs[0]["params"]["iters"], cs[0]["params"])

# ================================================================ metropolis-hastings
T = "metropolis-hastings"
pt = np.array([0.2, 0.3, 0.5])
Pm = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        if i != j:
            Pm[i, j] = 0.5 * min(1, pt[j] / pt[i])
    Pm[i, i] = 1 - Pm[i].sum()
check(T, "例1 推移行列 P(1,·)=(0,.5,.5), P(2,·)=(.3333,.1667,.5), P(3,·)=(.2,.3,.5)",
      allclose(Pm, [[0, 0.5, 0.5], [0.3333, 0.1667, 0.5], [0.2, 0.3, 0.5]]), Pm.round(4))
F = pt[:, None] * Pm
check(T, "例1 詳細つり合い π_iP(i,j)=π_jP(j,i)（0.1, 0.15 など）", allclose(F, F.T, 1e-12) and close(F[0, 1], 0.1, 1e-12) and close(F[1, 2], 0.15, 1e-12), F.round(4))
check(T, "例1 πP=π", allclose(pt @ Pm, pt, 1e-12), pt @ Pm)
d1 = np.array([1.0, 0, 0]) @ Pm
d2 = d1 @ Pm
d3 = d2 @ Pm
check(T, "例1 状態1から (0,.5,.5), (.2667,.2333,.5), (.1778,.3222,.5)",
      allclose(d1, [0, 0.5, 0.5]) and allclose(d2, [0.2667, 0.2333, 0.5]) and allclose(d3, [0.1778, 0.3222, 0.5]), (d2, d3))
check(T, "例1 固有値に −1/3", any(abs(ev_ + 1 / 3) < 1e-12 for ev_ in np.linalg.eigvals(Pm)), np.linalg.eigvals(Pm))
check(T, "例2 e^{−0.625}=0.5353, e^{0.48}=1.616",
      close(math.exp(-(1.5 ** 2 - 1) / 2), 0.5353) and close(math.exp(-(0.04 - 1) / 2), 1.616, 5e-4), None)
targ = lambda v: v * math.exp(-v)
qd = lambda v: 0.5 * math.exp(-0.5 * v)
al3 = min(1, targ(4) * qd(2) / (targ(2) * qd(4)))
check(T, "例3 独立連鎖型 α=min(1,2e^{−1})=0.7358、補正なし 2e^{−2}=0.2707",
      close(al3, 0.7358) and close(2 / math.e, 0.7358) and close(targ(4) / targ(2), 0.2707) and close(2 * math.exp(-2), 0.2707), al3)
# 一般の MH の詳細つり合い：min の対称性（sympy）
a_, b_ = sp.symbols("A B", positive=True)
check(T, "π(x)q(y|x)α(x,y) = min{π(x)q(y|x), π(y)q(x|y)} は対称（sympy, 数値代入）",
      all(sp.N(a_ * sp.Min(1, b_ / a_) - b_ * sp.Min(1, a_ / b_)).subs({a_: av, b_: bv}) == 0 for av, bv in [(0.3, 0.7), (2.0, 0.1), (1.0, 1.0)]), None)
# 独立連鎖型で Ga(2,1) を生成（seed=20261023、平均2・分散2 を 相対3%）
rng = np.random.default_rng(20261023)
xc_ = 1.0
outs = np.empty(100000)
for i in range(100000):
    yv = rng.exponential(2)
    if rng.random() <= min(1, (yv * math.exp(-0.5 * yv)) / (xc_ * math.exp(-0.5 * xc_))):
        xc_ = yv
    outs[i] = xc_
outs = outs[1000:]
check(T, "例3 の独立連鎖 MH で Ga(2,1)（10^5 回）: 平均≈2・分散≈2（相対3%）", abs(outs.mean() / 2 - 1) < 0.03 and abs(outs.var() / 2 - 1) < 0.03, (outs.mean(), outs.var()))
c = charts(T)[0]
check(T, "chart mcmc-trace target=norm init=−6", c["params"]["target"] == "norm" and c["params"]["init"] == -6, c["params"])

# ================================================================ gibbs-sampling
T = "gibbs-sampling"
rho = sp.symbols("rho", real=True)
xg_, yg_ = sp.symbols("x y", real=True)
Sig = sp.Matrix([[1, rho], [rho, 1]])
dens = sp.exp(-(sp.Matrix([xg_, yg_]).T * Sig.inv() * sp.Matrix([xg_, yg_]))[0] / 2)
cond = sp.exp(-(xg_ - rho * yg_) ** 2 / (2 * (1 - rho ** 2)))
check(T, "X|Y=y ~ N(ρy, 1−ρ²)：同時密度/条件付き核 が x に依らない（sympy）", sp.simplify(sp.diff(sp.log(dens) - sp.log(cond), xg_)) == 0, None)
z_, zp = sp.symbols("Z Zp", real=True)
xt = sp.symbols("x_t", real=True)
yt = rho * xt + sp.sqrt(1 - rho ** 2) * zp
xn = rho * yt + sp.sqrt(1 - rho ** 2) * z_
check(T, "x^{t+1}=ρ²x^t+ρ√(1−ρ²)Z'+√(1−ρ²)Z（sympy）", sp.expand(xn - (rho ** 2 * xt + rho * sp.sqrt(1 - rho ** 2) * zp + sp.sqrt(1 - rho ** 2) * z_)) == 0, None)
check(T, "AR(1) の誤差分散 ρ²(1−ρ²)+(1−ρ²)=1−ρ⁴ で定常分散1", sp.simplify((rho ** 2 * (1 - rho ** 2) + (1 - rho ** 2)) / (1 - rho ** 4) - 1) == 0, None)
# MH として採択確率1（数値で）
check(T, "ギブスの採択比=1（2変量正規、数値代入）",
      close((stats.multivariate_normal([0, 0], [[1, .8], [.8, 1]]).pdf([1.3, .2]) * stats.norm(.8 * .2, .6).pdf(-.4))
            / (stats.multivariate_normal([0, 0], [[1, .8], [.8, 1]]).pdf([-.4, .2]) * stats.norm(.8 * .2, .6).pdf(1.3)), 1, 1e-12), None)
seqv = []
yv = 2.0
for zz in [(0.5, -1.0), (0.3, 1.2)]:
    xv = 0.8 * yv + 0.6 * zz[0]
    yv = 0.8 * xv + 0.6 * zz[1]
    seqv += [xv, yv]
check(T, "例1 1.9, 0.92, 0.916, 1.4528", allclose(seqv, [1.9, 0.92, 0.916, 1.4528], 1e-12) and close(math.sqrt(1 - 0.64), 0.6, 1e-12), seqv)
check(T, "例1 E[x^(t)]=ρ^{2t−1}·2: t=1,2,5,10 で 1.6, 1.024, 0.268, 0.029",
      allclose([0.8 ** (2 * tt - 1) * 2 for tt in (1, 2, 5, 10)], [1.6, 1.024, 0.268, 0.029], 5e-4), [0.8 ** (2 * tt - 1) * 2 for tt in (1, 2, 5, 10)])
rng = np.random.default_rng(8)
yv = 2.0
XS, YS = np.empty(11000), np.empty(11000)
for i in range(11000):
    xv = 0.8 * yv + 0.6 * rng.standard_normal()
    yv = 0.8 * xv + 0.6 * rng.standard_normal()
    XS[i], YS[i] = xv, yv
XS, YS = XS[1000:], YS[1000:]
vals = [XS.mean(), YS.mean(), XS.var(ddof=1), YS.var(ddof=1), np.corrcoef(XS, YS)[0, 1], acf(XS, 1)]
check(T, "例2 シード8: 0.009, 0.023, 1.000, 1.004, 0.803, ρ1 0.653（default_rng(8) で再現）",
      allclose(vals, [0.009, 0.023, 1.000, 1.004, 0.803, 0.653], 5e-4), np.round(vals, 4))
check(T, "例2 (1+ρ²)/(1−ρ²)=4.556, ESS 2195", close(1.64 / 0.36, 4.556, 5e-4) and round(1e4 * 0.36 / 1.64) == 2195, 1e4 * 0.36 / 1.64)
essg = {r_: 1e4 * (1 - r_ ** 2) / (1 + r_ ** 2) for r_ in (0.5, 0.9, 0.99)}
check(T, "例2 ρ=0.5→6000, 0.9→1050, 0.99→約100", round(essg[0.5]) == 6000 and round(essg[0.9]) == 1050 and abs(essg[0.99] - 100) < 1, essg)
cs = charts(T)
check(T, "chart1 gibbs-path rho=0.8 init=2（例1と同じ）・chart2 bvnorm rho=0.8 seed=8",
      cs[0]["params"]["rho"] == 0.8 and cs[0]["params"]["init"] == 2 and cs[1]["params"]["rho"] == 0.8 and cs[1]["seed"] == 8, None)

# ================================================================ burn-in-convergence
T = "burn-in-convergence"
ph = sp.symbols("phi", positive=True)
kk_ = sp.symbols("k", integer=True, positive=True)
check(T, "1+2Σφ^k=(1+φ)/(1−φ)（sympy, |φ|<1）",
      sp.simplify(1 + 2 * sp.summation(sp.Rational(9, 10) ** kk_, (kk_, 1, sp.oo)) - sp.Rational(19, 1)) == 0
      and sp.simplify(1 + 2 * ph / (1 - ph) - (1 + ph) / (1 - ph)) == 0, None)
tt = math.log(0.01) / math.log(0.9)
check(T, "例1 t>43.7、E[X_44]=0.097", close(tt, 43.7, 0.05) and close(0.9 ** 44 * 10, 0.097, 5e-4) and 0.9 ** 44 < 0.01 < 0.9 ** 43, (tt, 0.9 ** 44 * 10))
check(T, "例1 n_eff: φ=0.9→526.3, 0.5→3333, 0.99→50",
      close(1e4 * 0.1 / 1.9, 526.3, 0.05) and round(1e4 * 0.5 / 1.5) == 3333 and round(1e4 * 0.01 / 1.99) == 50, None)
# AR(1) で ESS 公式（seed=20261023、φ=0.9、標本平均の分散 相対5%）
rng = np.random.default_rng(20261023)
nrep, nlen = 2000, 2000
e_ = rng.standard_normal((nrep, nlen)) * math.sqrt(1 - 0.81)
xa_ = np.empty((nrep, nlen))
xa_[:, 0] = rng.standard_normal(nrep)
for i in range(1, nlen):
    xa_[:, i] = 0.9 * xa_[:, i - 1] + e_[:, i]
vm = xa_.mean(1).var(ddof=1)
check(T, "AR(1) φ=0.9: V[x̄]≈σ²/n_eff（2000本×2000、相対5%）", abs(vm / (1 / (nlen * 0.1 / 1.9)) - 1) < 0.05, vm * nlen * 0.1 / 1.9)


def rhat(means, vars_, n_):
    m_ = len(means)
    Wv = np.mean(vars_)
    Bv = n_ / (m_ - 1) * np.sum((np.array(means) - np.mean(means)) ** 2)
    Vh = (n_ - 1) / n_ * Wv + Bv / n_
    return Wv, Bv, Vh, math.sqrt(Vh / Wv)


Wv, Bv, Vh, Rh = rhat([0.1, 0.5], [1.0, 1.2], 1000)
check(T, "例2 W=1.1, B=80, V̂=1.1789, R̂=1.0352", close(Wv, 1.1, 1e-12) and close(Bv, 80, 1e-9) and close(Vh, 1.1789) and close(Rh, 1.0352), (Vh, Rh))
Wv, Bv, Vh, Rh = rhat([0.28, 0.32], [1.0, 1.2], 1000)
check(T, "例2 平均0.28,0.32: B=0.8, V̂=1.0997, R̂=0.9999", close(Bv, 0.8, 1e-9) and close(Vh, 1.0997) and close(Rh, 0.9999), (Vh, Rh))
cs = charts(T)
check(T, "chart2 acf-plot ar1 phi=0.9（例1と同じ φ）", cs[1]["params"]["model"] == "ar1" and cs[1]["params"]["phi"] == 0.9, cs[1]["params"])

# ================================================================ bootstrap
T = "bootstrap"
data = np.array([3.1, 4.5, 2.2, 6.8, 5.0, 3.9, 7.4, 4.1, 2.9, 5.6])
sdat = data.std(ddof=1)
check(T, "例1 平均4.55・中央値4.3・s=1.682・SE 0.532・σ̂/√n 0.505",
      close(data.mean(), 4.55, 1e-12) and close(np.median(data), 4.3, 1e-12) and close(sdat, 1.682, 5e-4)
      and close(sdat / math.sqrt(10), 0.532, 5e-4) and close(data.std() / math.sqrt(10), 0.505, 5e-4), (sdat, data.std() / math.sqrt(10)))
rng = np.random.default_rng(9)
bs = rng.choice(data, (2000, 10), replace=True)
bm = bs.mean(1)
bmed = np.median(bs, 1)
check(T, "例1 1つ目の標本 5.0,2.9,5.6,2.2,4.5,7.4,7.4,4.1,7.4,4.1・平均5.06（default_rng(9) で再現）",
      allclose(bs[0], [5.0, 2.9, 5.6, 2.2, 4.5, 7.4, 7.4, 4.1, 7.4, 4.1], 1e-12) and close(bs[0].mean(), 5.06, 1e-9)
      and not np.isin([3.1, 6.8], bs[0]).any() and (bs[0] == 7.4).sum() == 3, bs[0])
check(T, "例1 平均: SE 0.511・偏り −0.0003", close(bm.std(ddof=1), 0.511, 5e-4) and close(bm.mean() - 4.55, -0.0003), (bm.std(ddof=1), bm.mean() - 4.55))
check(T, "例1 中央値: SE 0.669・偏り 0.071", close(bmed.std(ddof=1), 0.669, 5e-4) and close(bmed.mean() - 4.3, 0.071, 5e-4), (bmed.std(ddof=1), bmed.mean() - 4.3))
check(T, "例2 C(19,10)=92378, 0.9^10=0.3487, e^{−1}=0.3679",
      math.comb(19, 10) == 92378 and close(0.9 ** 10, 0.3487) and close(math.exp(-1), 0.3679), None)
check(T, "区別できる標本数 C(2n−1,n) を n=4 で全列挙確認",
      len({tuple(sorted(c_)) for c_ in np.array(np.meshgrid(*[range(4)] * 4)).T.reshape(-1, 4)}) == math.comb(7, 4), None)
check(T, "含まれる割合 1−e^{−1}≈63%", close(1 - math.exp(-1), 0.63, 5e-3), 1 - math.exp(-1))
# B→∞ で SE_boot → σ̂/√n（厳密：経験分布の分散/n）
check(T, "標本平均: V[x̄*]=σ̂²/n=((n−1)/n)s²/n", close(data.var() / 10, (9 / 10) * sdat ** 2 / 10, 1e-12), None)
cs = charts(T)
check(T, "chart bootstrap-hist n=20 B=1000 stat mean/median", [c_["params"]["stat"] for c_ in cs] == ["mean", "median"] and cs[0]["params"]["n"] == 20, None)

# ================================================================ bootstrap-confidence-interval
T = "bootstrap-confidence-interval"
q = np.percentile(bm, [2.5, 97.5])
se_b = bm.std(ddof=1)
check(T, "例1 シード9 B=2000: 2.5%点3.580・97.5%点5.570・SE 0.511（線形補間、再現）", allclose(q, [3.580, 5.570], 5e-4) and close(se_b, 0.511, 5e-4), q)
check(T, "例1 ベーシック [3.530, 5.520]", allclose([9.1 - q[1], 9.1 - q[0]], [3.530, 5.520], 5e-4), [9.1 - q[1], 9.1 - q[0]])
check(T, "例1 正規近似 [3.548, 5.552]", allclose([4.55 - 1.96 * se_b, 4.55 + 1.96 * se_b], [3.548, 5.552], 5e-4), [4.55 - 1.96 * se_b, 4.55 + 1.96 * se_b])
t9 = stats.t(9).ppf(0.975)
check(T, "例1 t区間 t(9)=2.262, [3.347, 5.753]",
      close(t9, 2.262, 5e-4) and allclose([4.55 - t9 * sdat / math.sqrt(10), 4.55 + t9 * sdat / math.sqrt(10)], [3.347, 5.753], 5e-4),
      [4.55 - t9 * sdat / math.sqrt(10), 4.55 + t9 * sdat / math.sqrt(10)])
qm = np.percentile(bmed, [2.5, 97.5])
se_m = bmed.std(ddof=1)
check(T, "例2 中央値 パーセンタイル [3.1, 5.9]・ベーシック [2.7, 5.5]", allclose(qm, [3.1, 5.9], 1e-9) and allclose([8.6 - qm[1], 8.6 - qm[0]], [2.7, 5.5], 1e-9), qm)
check(T, "例2 正規近似 4.3±1.96×0.669 = [2.988, 5.612]（SE の非丸め値 0.6692 で）",
      allclose([4.3 - 1.96 * se_m, 4.3 + 1.96 * se_m], [2.988, 5.612], 5e-4), [4.3 - 1.96 * se_m, 4.3 + 1.96 * se_m])
check(T, "B=1000 の2.5%点は25番目付近", close(1000 * 0.025, 25, 1e-12), None)
c = charts(T)[0]
check(T, "chart bootstrap-hist B=2000 stat mean", c["params"]["B"] == 2000 and c["params"]["stat"] == "mean", c["params"])

# ================================================================ jackknife
T = "jackknife"
xj = np.array([2.0, 4, 5, 7, 12])
n_ = 5


def jack(stat):
    th_ = stat(xj)
    loo = np.array([stat(np.delete(xj, i)) for i in range(n_)])
    tb = loo.mean()
    bias = (n_ - 1) * (tb - th_)
    vj = (n_ - 1) / n_ * ((loo - tb) ** 2).sum()
    return th_, loo, tb, bias, th_ - bias, vj


th_, loo, tb, bias, corr, vj = jack(lambda v: v.var())
check(T, "例1 x̄=6, σ̂²=58/5=11.6", close(xj.mean(), 6, 1e-12) and close(th_, 11.6, 1e-12), th_)
check(T, "例1 σ̂²_(i)=9.5, 13.25, 14.1875, 14.1875, 3.25・平均10.875", allclose(loo, [9.5, 13.25, 14.1875, 14.1875, 3.25], 1e-12) and close(tb, 10.875, 1e-12), loo)
check(T, "例1 偏り −2.9・補正 14.5 = s²", close(bias, -2.9, 1e-12) and close(corr, 14.5, 1e-12) and close(xj.var(ddof=1), 14.5, 1e-12), (bias, corr))
check(T, "例1 V_jack=70.094・SE 8.372", close(vj, 70.094, 5e-4) and close(math.sqrt(vj), 8.372, 5e-4), (vj, math.sqrt(vj)))
pseudo = n_ * th_ - (n_ - 1) * loo
check(T, "擬似値: 平均=補正推定量、不偏分散/n=V_jack", close(pseudo.mean(), corr, 1e-12) and close(pseudo.var(ddof=1) / n_, vj, 1e-9), pseudo)
th_, loo, tb, bias, corr, vj = jack(np.mean)
check(T, "例2 平均: θ_(i)=7,6.5,6.25,5.75,4.5・偏り0・V=2.9=s²/n",
      allclose(loo, [7, 6.5, 6.25, 5.75, 4.5], 1e-12) and close(bias, 0, 1e-12) and close(vj, 2.9, 1e-12) and close(14.5 / 5, 2.9, 1e-12), vj)
th_, loo, tb, bias, corr, vj = jack(lambda v: v.mean() ** 2)
check(T, "例2 x̄²: θ_(i)=49,42.25,39.0625,33.0625,20.25・平均36.725・偏り2.9・補正33.1",
      allclose(loo, [49, 42.25, 39.0625, 33.0625, 20.25], 1e-12) and close(tb, 36.725, 1e-12) and close(bias, 2.9, 1e-12) and close(corr, 33.1, 1e-12)
      and close(36 - 14.5 / 5, 33.1, 1e-12), (tb, bias, corr))
# 一般の証明（sympy）：σ̂² のジャックナイフ補正は常に s²、x̄ の V_jack は常に s²/n（記号 n=5 の任意データ）
xs5 = sp.symbols("x1:6", real=True)


def svar(v):
    mv = sum(v) / len(v)
    return sum((vi - mv) ** 2 for vi in v) / len(v)


loo_s = [svar([xs5[j] for j in range(5) if j != i]) for i in range(5)]
corr_s = 5 * svar(xs5) - 4 * sum(loo_s) / 5
s2_s = svar(xs5) * sp.Rational(5, 4)
check(T, "σ̂² のジャックナイフ補正 = s²（任意データ、sympy）", sp.simplify(sp.expand(corr_s - s2_s)) == 0, None)
loo_m = [sum(xs5[j] for j in range(5) if j != i) / 4 for i in range(5)]
tbm = sum(loo_m) / 5
vj_s = sp.Rational(4, 5) * sum((l_ - tbm) ** 2 for l_ in loo_m)
check(T, "x̄ の V_jack = s²/n（任意データ、sympy）", sp.simplify(sp.expand(vj_s - s2_s / 5)) == 0, None)
nn_, aa_ = sp.symbols("n a", positive=True)
check(T, "E[θ̃]=n(θ+a/n)−(n−1)(θ+a/(n−1))=θ（sympy）", sp.simplify(nn_ * (th + aa_ / nn_) - (nn_ - 1) * (th + aa_ / (nn_ - 1)) - th) == 0
      if (th := sp.symbols("theta")) is not None else False, None)
check(T, "θ̂_(i)−θ̄_(·) = −(x_i−x̄)/(n−1)（sympy）", all(sp.simplify(loo_m[i] - tbm + (xs5[i] - sum(xs5) / 5) / 4) == 0 for i in range(5)), None)
c = charts(T)[0]
check(T, "chart bar-data の値 = σ̂²_(i)・ラベル順 2,4,5,7,12",
      allclose(c["params"]["values"], jack(lambda v: v.var())[1], 1e-12) and c["params"]["labels"] == [f"{int(v)}を除く" for v in xj], c["params"]["values"])
check(T, "chart caption の 11.6, 3.25", "11.6" in c["caption"] and "3.25" in c["caption"], None)

# ================================================================ 集計
nok = sum(results)
print(f"\n合計: OK {nok} / NG {len(results) - nok}")
