"""第13章（ノンパラメトリック法）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from itertools import combinations, permutations
from math import comb, pi
from scipy import stats

q = Q(13)

# --- 順位とタイ ---
rk = stats.rankdata([3, 5, 5, 8, 5, 10])
q.check("q13-rank-and-ties-1", rk[1] == 3 and all(abs(w - 3) > 0.1 for w in [2, 4, 2.5]), "5 の平均順位は 3、誤答は異なる")
x = np.array([1, 2, 2, 2, 3, 4, 4, 5, 6, 7], dtype=float)  # タイ 3個と2個
r = stats.rankdata(x)
Cfac = 1 - 30 / 990
q.num("q13-rank-and-ties-2", Cfac)
q.check("q13-rank-and-ties-2", abs(r.var() - (100 - 1) / 12 * Cfac) < 1e-12, "平均順位の分散 = (N²-1)/12 × C")

# --- ARE ---
q.check("q13-nonparametric-methods-overview-1", abs(3 / pi - 0.955) < 5e-4 and abs(2 / pi - 0.637) < 5e-4,
        "順位和は 3/π≈0.955（0.64 は符号検定）＝誤り")

# --- 符号検定 ---
q.num("q13-sign-test-1", stats.binom.sf(8, 11, 0.5))
q.check("q13-sign-test-1", abs(stats.binom.sf(8, 11, 0.5) - 67 / 2048) < 1e-12, "67/2048")
q.num("q13-sign-test-2", (62 - 50 - 0.5) / 5)
# 非対称な分布（中央値0の指数分布をずらしたもの）でも P(D>0)=1/2
m = np.log(2)
q.check("q13-sign-test-3", abs(stats.expon(loc=-m).sf(0) - 0.5) < 1e-12, "非対称でも中央値0なら P(D>0)=1/2（対称性不要＝誤り）")

# --- 符号付き順位 ---
d = np.array([1.2, -0.4, 2.5, 0.8, -1.5, 3.1, 0.6, 2.0])
rr = stats.rankdata(np.abs(d))
Tp = rr[d > 0].sum()
q.num("q13-wilcoxon-signed-rank-test-1", Tp)
q.num("q13-wilcoxon-signed-rank-test-2", (160 - 105) / np.sqrt(20 * 21 * 41 / 24))
# 対称性：非対称な中央値0の分布では E[T+] が n(n+1)/4 からずれる（数値で確認）
rng = np.random.default_rng(1)
sim = rng.exponential(size=(20000, 10)) - m
Tps = np.array([stats.rankdata(np.abs(s))[s > 0].sum() for s in sim])
q.check("q13-wilcoxon-signed-rank-test-3", abs(Tps.mean() - 27.5) > 1.0,
        "対称でない（中央値0の）分布では T+ の平均が n(n+1)/4 からずれる＝対称性が必要")

# --- 順位和 ---
sums = [sum(c) for c in combinations(range(1, 9), 3)]
pW = sum(s <= 12 for s in sums) / comb(8, 3)
q.num("q13-wilcoxon-rank-sum-test-1", pW)
q.check("q13-wilcoxon-rank-sum-test-1", abs(stats.mannwhitneyu([12, 15, 21], [10, 14, 18, 25, 30],
        alternative="less", method="exact").pvalue - pW) < 1e-12, "scipy の正確検定と一致")
q.num("q13-wilcoxon-rank-sum-test-2", (160 - 130) / np.sqrt(325))

# --- U ---
U = sum(a > b for a in [12, 15, 21] for b in [10, 14, 18, 25, 30])
q.num("q13-mann-whitney-u-test-1", U)
q.check("q13-mann-whitney-u-test-1", stats.mannwhitneyu([12, 15, 21], [10, 14, 18, 25, 30]).statistic == U, "scipy と一致")

# --- クラスカル・ウォリス ---
g = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
H = stats.kruskal(*g).statistic
q.num("q13-kruskal-wallis-test-1", H)
q.check("q13-kruskal-wallis-test-1", H > stats.chi2.ppf(0.99, 2), "χ²(2) 上側1%点 9.21 を超える")
q.check("q13-kruskal-wallis-test-2", len({4 - 1, 4, 40 - 4}) == 3, "自由度 k-1=3、誤答 4・36 は異なる")
a, b = [1.3, 2.8, 4.1, 5.5, 7.2], [3.3, 6.1, 6.6, 8.0, 9.4, 10.2]
Hk = stats.kruskal(a, b).statistic
W = stats.rankdata(a + b)[:5].sum(); n1, n2 = 5, 6; N = 11
Zw = (W - n1 * (N + 1) / 2) / np.sqrt(n1 * n2 * (N + 1) / 12)
q.check("q13-kruskal-wallis-test-3", abs(Hk - Zw ** 2) < 1e-9, "k=2 で H = Z²（正しい）")

# --- フリードマン ---
b_, k_ = 6, 3
R = np.array([8, 12, 16])
Qf = 12 / (b_ * k_ * (k_ + 1)) * (R ** 2).sum() - 3 * b_ * (k_ + 1)
q.num("q13-friedman-test-1", Qf)
ranks = np.array([[1, 2, 3], [1, 2, 3], [2, 1, 3], [1, 3, 2], [2, 1, 3], [1, 3, 2]])
q.check("q13-friedman-test-1", (ranks.sum(0) == R).all() and abs(stats.friedmanchisquare(*ranks.T).statistic - Qf) < 1e-9,
        "順位和 8,12,16 を作るデータで scipy と一致")
S, bb = sp.symbols("S b", positive=True)
R2 = bb + S; R1 = 3 * bb - R2
Qs = sp.Rational(12, 6) / bb * ((R1 - 3 * bb / 2) ** 2 + (R2 - 3 * bb / 2) ** 2)
q.check("q13-friedman-test-2", sp.simplify(Qs - (S - bb / 2) ** 2 / (bb / 4)) == 0, "k=2 で Q=Z²（sympy）")

# --- スピアマン・ケンドール ---
X5, Y5 = [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]
q.num("q13-spearman-rank-correlation-1", stats.spearmanr(X5, Y5).statistic)
q.num("q13-spearman-rank-correlation-2", 0.45 * np.sqrt(25))
q.num("q13-kendall-rank-correlation-1", stats.kendalltau(X5, Y5).statistic)
VS = 10 * 9 * 25 / 18
q.num("q13-kendall-rank-correlation-2", (21 - 1) / np.sqrt(VS))
# V[S] を n=6 の全並べ替えで確認（式の確認）
Ss = [sum(np.sign(j - i) * np.sign(p[j] - p[i]) for i in range(6) for j in range(i + 1, 6)) for p in permutations(range(6))]
q.check("q13-kendall-rank-correlation-2", abs(np.var(Ss) - 6 * 5 * 17 / 18) < 1e-9, "V[S]=n(n-1)(2n+5)/18 を全列挙で確認")

# --- 並べ替え ---
z = np.array([3, 4, 6, 7, 9])
Ts = np.array([sum(c) for c in combinations(z, 2)])
q.num("q13-permutation-test-1", (Ts >= 16).mean())
q.num("q13-permutation-test-2", Ts.var())
q.check("q13-permutation-test-2", abs(2 * 3 / 20 * ((z - z.mean()) ** 2).sum() - Ts.var()) < 1e-12, "公式と全列挙が一致")

# --- 正規近似の分散 ---
v = 10 * 11 * 21 / 24
signs = np.array(list(np.ndindex(*(2,) * 10)))
Tall = signs @ np.arange(1, 11)
q.check("q13-normal-approximation-rank-tests-1", abs(Tall.var() - v) < 1e-9 and v == 96.25
        and all(abs(w - v) > 1 for w in [192.5, 27.5, 385]), "2^10 通りの全列挙で 96.25、誤答は異なる")

# --- ARE ---
q.num("q13-asymptotic-relative-efficiency-1", 100 / (2 / pi))

# --- ラン ---
q.num("q13-runs-test-1", 2 * 8 * 12 / 20 + 1)

# --- 一致対・不一致対 ---
pts = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3)]
C = D = 0
for (x1, y1), (x2, y2) in combinations(pts, 2):
    s_ = (x2 - x1) * (y2 - y1)
    C += s_ > 0; D += s_ < 0
gam = (C - D) / (C + D)
tb = stats.kendalltau(*zip(*pts)).statistic
q.check("q13-concordant-discordant-pairs-1", abs(gam - 1 / 3) < 1e-12 and abs(tb - 0.25) < 1e-12
        and abs((C - D) / 10 - 0.2) < 1e-12 and abs(gam - 3 / 5) > 0.1, "γ=1/3、τ_b=1/4・τ_a=1/5 は誤答")

# --- 導出問題（追加） ---
def _deriv13():
    import itertools
    import numpy as _N
    import sympy as _S
    from scipy import stats as _st
    n, r = _S.symbols("n r", positive=True, integer=True)
    E = _S.summation(r / 2, (r, 1, n)); V = _S.summation(r ** 2 / 4, (r, 1, n))
    ok = _S.simplify(E - n * (n + 1) / 4) == 0 and _S.simplify(V - n * (n + 1) * (2 * n + 1) / 24) == 0
    # n=8 で全 2^8 通りの符号を列挙した厳密分布と一致
    nn = 8
    vals = _N.array([sum(k + 1 for k in range(nn) if s[k]) for s in itertools.product([0, 1], repeat=nn)], float)
    ok &= abs(vals.mean() - nn * (nn + 1) / 4) < 1e-12 and abs(vals.var() - nn * (nn + 1) * (2 * nn + 1) / 24) < 1e-12
    q.check("q13-wilcoxon-signed-rank-test-4", bool(ok), "sympy: Σr/2、Σr²/4、n=8 の全列挙で E=18, V=51")
    k = _S.symbols("k", positive=True, integer=True)
    SS = _S.summation(k ** 2, (k, 1, n)) - n * ((n + 1) / 2) ** 2
    ok2 = _S.simplify(SS - n * (n ** 2 - 1) / 12) == 0
    rng = _N.random.default_rng(13)
    for nn in (5, 9, 20):
        for _ in range(20):
            R = rng.permutation(nn) + 1; S = rng.permutation(nn) + 1
            rs = 1 - 6 * ((R - S) ** 2).sum() / (nn * (nn ** 2 - 1))
            ok2 &= abs(rs - _st.spearmanr(R, S)[0]) < 1e-12
    ok2 &= abs(1 - 6 * ((_N.arange(1, 7) - _N.arange(6, 0, -1)) ** 2).sum() / (6 * 35) + 1) < 1e-12
    q.check("q13-spearman-rank-correlation-3", bool(ok2), "sympy: 平方和 n(n²-1)/12、乱数の順位で scipy.spearmanr と一致、逆順で -1")
_deriv13()

# --- 4択の追加分 ---
_n1, _n2 = 6, 8; _N = _n1 + _n2
_W = np.array([sum(c) for c in combinations(range(1, _N + 1), _n1)], dtype=float)
_EW, _VW = _W.mean(), _W.var()
q.check("q13-wilcoxon-rank-sum-test-3",
        abs(_EW - 45) < 1e-9 and abs(_VW - 60) < 1e-9
        and abs(_n2 * (_N + 1) / 2 - 45) > 1 and abs(_n1 * _n2 / 2 - 45) > 1
        and round(np.sqrt(_VW), 2) == 7.75 and abs(7.75 - 60) > 1,
        f"全 C(14,6) 通りの順位和から E[W]={_EW}、V[W]={_VW:.4f}。誤答 E=60（n2）、24（U）、V=7.75（標準偏差）")
# フリードマン：作業者ごとの定数（ブロック効果）を足しても統計量は変わらないが、クラスカル・ウォリスは変わる
_rng = np.random.default_rng(13)
_base = _rng.exponential(1.0, size=(10, 3)) + np.array([0.0, 0.3, 0.6])
_blk = _base + _rng.exponential(5.0, size=(10, 1))
_fr0 = stats.friedmanchisquare(*_base.T).statistic; _fr1 = stats.friedmanchisquare(*_blk.T).statistic
_kw0 = stats.kruskal(*_base.T).statistic; _kw1 = stats.kruskal(*_blk.T).statistic
q.check("q13-friedman-test-3", abs(_fr0 - _fr1) < 1e-12 and abs(_kw0 - _kw1) > 1e-6,
        f"ブロック効果を加えてもフリードマン Q は不変（{_fr0:.3f}→{_fr1:.3f}）、KW は変化（{_kw0:.3f}→{_kw1:.3f}）。順位和・符号付き順位は2群用")

# --- 追加：答えが「正しい」の正誤問題と、重要度Cの3語 ---
import numpy as _np
from itertools import permutations as _perms, combinations as _combs
from scipy import stats as _st
# ケンドールの τ：n=5 の全順列で -1≤τ≤1、τ=1 は恒等順列だけ
_ts = {}
for _pm in _perms(range(5)):
    _ts[_pm] = _st.kendalltau(range(5), _pm).statistic
_ones = [p for p, v in _ts.items() if abs(v - 1) < 1e-12]
_xs = _np.arange(1, 8, dtype=float)
q.check("q13-kendall-rank-correlation-3", all(-1 - 1e-12 <= v <= 1 + 1e-12 for v in _ts.values()) and _ones == [tuple(range(5))]
        and abs(_st.kendalltau(_xs, _np.exp(_xs)).statistic - 1) < 1e-12 and _st.pearsonr(_xs, _np.exp(_xs))[0] < 0.95,
        "n=5 の120順列で τ∈[-1,1]、τ=1 は順位が一致するときだけ。y=e^x で τ=1 だがピアソン相関<1")

# 2標本KS：m=n=4 の全70通りの並びから正確分布、正規・指数の乱数でも同じ分布
def _ks_stat(x, y):
    return _st.ks_2samp(x, y).statistic
_exact = {}
for _c in _combs(range(8), 4):
    _x = [i for i in range(8) if i in _c]; _y = [i for i in range(8) if i not in _c]
    _d = round(_ks_stat(_x, _y), 6); _exact[_d] = _exact.get(_d, 0) + 1 / 70
_p_exact = sum(v for d, v in _exact.items() if d >= 0.75 - 1e-9)
_rng = _np.random.default_rng(1313)
_sim = {}
for _nm, _gen in [("norm", lambda: _rng.normal(size=8)), ("expon", lambda: _rng.exponential(size=8))]:
    _cnt = 0
    for _ in range(20000):
        _z = _gen(); _cnt += _ks_stat(_z[:4], _z[4:]) >= 0.75 - 1e-9
    _sim[_nm] = _cnt / 20000
_zz = _rng.normal(size=12)
_inv = abs(_ks_stat(_zz[:5], _zz[5:]) - _ks_stat(_np.exp(_zz[:5]), _np.exp(_zz[5:]))) < 1e-12
q.check("q13-kolmogorov-smirnov-two-sample-1", _inv and all(abs(v - _p_exact) < 0.01 for v in _sim.values()),
        f"単調変換で D 不変、P(D≥0.75) は正確 {_p_exact:.4f}、正規 {_sim['norm']:.4f}、指数 {_sim['expon']:.4f}")

# 中央値検定
_A = [21, 34, 28, 40, 25, 37, 31, 45]; _B = [18, 26, 22, 30, 15, 24, 33, 20]
_M = _np.median(_A + _B)
_tab = [[sum(v > _M for v in _A), sum(v <= _M for v in _A)], [sum(v > _M for v in _B), sum(v <= _M for v in _B)]]
_chi = _st.chi2_contingency(_tab, correction=False)
q.num("q13-median-test-1", _chi[0])
q.check("q13-median-test-1", _M == 27 and _tab == [[6, 2], [2, 6]] and abs(_chi[1] - 0.0455) < 5e-4
        and abs(_st.fisher_exact(_tab)[1] - 0.132) < 5e-4, "M=27、表 6,2/2,6、p=0.046、フィッシャー両側 p=0.132")

# ヨンクヒール・タプストラ
_g = [[1.2, 2.5, 1.9], [2.2, 3.1, 2.8], [3.0, 3.6, 2.7]]
_U = {(i, j): sum(a < b for a in _g[i] for b in _g[j]) for i in range(3) for j in range(i + 1, 3)}
_J = sum(_U.values()); _N = 9; _ni = [3, 3, 3]
_EJ = (_N**2 - sum(k * k for k in _ni)) / 4
_VJ = (_N**2 * (2 * _N + 3) - sum(k * k * (2 * k + 3) for k in _ni)) / 72
q.num("q13-jonckheere-terpstra-test-1", (_J - _EJ) / _np.sqrt(_VJ))
# E[J], V[J] を全並べ替え（9!/(3!3!3!)=1680通り）で確認
_vals = sorted(sum(_g, []))
_Js = []
for _c1 in _combs(range(9), 3):
    _rest = [i for i in range(9) if i not in _c1]
    for _c2 in _combs(_rest, 3):
        _c3 = [i for i in _rest if i not in _c2]
        _gg = [[_vals[i] for i in _c1], [_vals[i] for i in _c2], [_vals[i] for i in _c3]]
        _Js.append(sum(sum(a < b for a in _gg[i] for b in _gg[j]) for i in range(3) for j in range(i + 1, 3)))
q.check("q13-jonckheere-terpstra-test-1", _U == {(0, 1): 8, (0, 2): 9, (1, 2): 6} and len(_Js) == 1680
        and abs(_np.mean(_Js) - _EJ) < 1e-9 and abs(_np.var(_Js) - _VJ) < 1e-9,
        "U=8,9,6 で J=23、全1680通りの並べ替えで E[J]=13.5、V[J]=20.25")

q.done()
