"""第25章 検算スクリプト（因子分析・グラフィカルモデル）。
4教科2因子モデル（Λ・共通性・独自分散・ΛΛ'+Ψ）、回転（45°・30°）、バリマックス基準と tan4θ の角度公式、
プロマックス（k=4）の U・Φ・因子パターン・因子構造、因子得点（回帰法・バートレット法）、
相関行列の固有値、最尤因子分析（F・T・自由度・p 値・残差）、精度行列と偏相関、
4変数連鎖の逆行列、共分散選択の逸脱度と −n log(1−r²) の一致、DAG の確率計算を
numpy / scipy / sympy で再計算し、本文と照合する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5 程度、第3位なら 5e-4 程度。各チェックに明記）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
乱数（seed=20261023）を使うチェックは、各チェックに許容誤差を明記する。
"""
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml
from scipy import stats, optimize

ROOT = Path(__file__).resolve().parent.parent
results = []
rng = np.random.default_rng(20261023)


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol + 1e-12


def allclose(xs, claims, tol=5e-5):
    xs = np.asarray(xs, float).ravel()
    claims = np.asarray(claims, float).ravel()
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-12))


def front(term):
    txt = (ROOT / "content" / "ch25" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term, cid):
    return [c for c in front(term).get("charts", []) if c["id"] == cid]


def r(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


def pcorr_from_prec(Om):
    d = np.sqrt(np.diag(Om))
    P = -Om / np.outer(d, d)
    np.fill_diagonal(P, 1.0)
    return P


SUBJ = ["国語", "英語", "数学", "理科"]
Lam = np.array([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
Psi = np.diag([0.32, 0.42, 0.32, 0.42])
R4 = Lam @ Lam.T + Psi

# ================================================================ factor-analysis-model
t = "factor-analysis-model"
h2 = (Lam ** 2).sum(1)
check(t, "例1 共通性 (0.68,0.58,0.68,0.58)", allclose(h2, [0.68, 0.58, 0.68, 0.58], 1e-12), r(h2))
check(t, "例1 ΛΛ'+Ψ の対角がすべて1", allclose(np.diag(R4), [1] * 4, 1e-12), r(np.diag(R4)))
check(t, "例1 r12=0.56+0.06=0.62, 国語-数学 0.32", close(R4[0, 1], 0.62, 1e-12) and close(R4[0, 2], 0.32, 1e-12),
      (R4[0, 1], R4[0, 2]))
Rclaim = np.array([[1, .62, .32, .38], [.62, 1, .38, .42], [.32, .38, 1, .62], [.38, .42, .62, 1]])
check(t, "例1 R 行列全体", allclose(R4, Rclaim, 1e-12), r(R4))
lam1 = np.array([0.9, 0.8, 0.7, 0.6])
R1 = np.outer(lam1, lam1)
np.fill_diagonal(R1, 1)
check(t, "例2 r12=0.72, r34=0.42", close(R1[0, 1], 0.72, 1e-12) and close(R1[2, 3], 0.42, 1e-12), (R1[0, 1], R1[2, 3]))
check(t, "例2 テトラッド r12r34=r13r24=0.3024 (0.63×0.48)",
      close(R1[0, 1] * R1[2, 3], 0.3024, 1e-12) and close(R1[0, 2] * R1[1, 3], 0.3024, 1e-12)
      and close(R1[0, 2], 0.63, 1e-12) and close(R1[1, 3], 0.48, 1e-12), R1[0, 2] * R1[1, 3])
check(t, "試験 λ1²=r12 r13/r23=0.72×0.63/0.56=0.81", close(R1[0, 1] * R1[0, 2] / R1[1, 2], 0.81, 1e-12),
      R1[0, 1] * R1[0, 2] / R1[1, 2])


def dof(p, m):
    return ((p - m) ** 2 - (p + m)) / 2


def dof_count(p, m):
    return p * (p + 1) / 2 - (p * m + p - m * (m - 1) / 2)


ok = all(dof(p, m) == dof_count(p, m) for p in range(1, 15) for m in range(0, p))
check(t, "自由度 {(p−m)²−(p+m)}/2 = p(p+1)/2 −(pm+p−m(m−1)/2)（p≤14 全 m）", ok, "一致")
check(t, "p=4: m=1 で 2, m=2 で −1", dof(4, 1) == 2 and dof(4, 2) == -1, (dof(4, 1), dof(4, 2)))
# 記号的に Σ=ΛΛ'+Ψ の導出：x=Λf+ε のシミュレーション
n = 400000
f = rng.standard_normal((n, 2))
e = rng.standard_normal((n, 4)) * np.sqrt(np.diag(Psi))
X = f @ Lam.T + e
C = np.cov(X.T)
check(t, "シミュレーション x=Λf+ε の標本共分散 ≈ ΛΛ'+Ψ（40万点, 許容 0.01）", np.abs(C - R4).max() < 0.01,
      f"最大誤差 {np.abs(C - R4).max():.4f}")
pts = charts(t, "points-2d")[0]["params"]["points"]
check(t, "charts points-2d = Λ（ラベル順 国語・英語・数学・理科）",
      [p_["label"] for p_ in pts] == SUBJ and allclose([[p_["x"], p_["y"]] for p_ in pts], Lam, 1e-12), "一致")
bd = charts(t, "bar-data")[0]["params"]
check(t, "charts bar-data = 共通性・caption の独自分散 0.32/0.42",
      bd["labels"] == SUBJ and allclose(bd["values"], h2, 1e-12) and allclose(1 - h2, [0.32, 0.42, 0.32, 0.42], 1e-12),
      bd["values"])

# ================================================================ common-factor-unique-factor
t = "common-factor-unique-factor"
check(t, "例1 国語 0.64+0.04=0.68, ψ1=0.32; 英語 0.49+0.09=0.58, ψ2=0.42",
      close(1 - h2[0], 0.32, 1e-12) and close(1 - h2[1], 0.42, 1e-12), r(1 - h2))
check(t, "例2 共通部分 68, 独自 32, 誤差 15, 特殊 17", (round(100 * 0.68), round(100 * 0.32), round(100 * 0.15), round(32 - 15)) == (68, 32, 15, 17),
      (68, 32, 15, 17))
bd = charts(t, "bar-data")[0]["params"]
check(t, "charts bar-data = 独自分散", bd["labels"] == SUBJ and allclose(bd["values"], np.diag(Psi), 1e-12), bd["values"])
# 分散分解の記号導出
l1, l2, ps = sp.symbols("l1 l2 psi", real=True)
c12 = sp.symbols("c12", real=True)
Vf = sp.Matrix([[1, 0], [0, 1]])
lv = sp.Matrix([l1, l2])
var = (lv.T * Vf * lv)[0] + ps
check(t, "sympy: V[x_j]=λ1²+λ2²+ψ（Cov[f]=I）", sp.simplify(var - (l1 ** 2 + l2 ** 2 + ps)) == 0, var)

# ================================================================ factor-loading
t = "factor-loading"
cs = (Lam ** 2).sum(0)
check(t, "例1 列の2乗和 1.26, 1.26; 合計 2.52 = 4 の 63%", allclose(cs, [1.26, 1.26], 1e-12) and close(cs.sum() / 4, 0.63, 1e-12), r(cs))
check(t, "例1 英語-理科 0.21+0.21=0.42", close(R4[1, 3], 0.42, 1e-12), R4[1, 3])
# Cov[x,f]=Λ をシミュレーションで
covxf = (X - X.mean(0)).T @ (f - f.mean(0)) / (n - 1)
check(t, "シミュレーション Cov[x,f] ≈ Λ（許容 0.01）", np.abs(covxf - Lam).max() < 0.01, f"最大誤差 {np.abs(covxf - Lam).max():.4f}")
pts = charts(t, "points-2d")[0]["params"]["points"]
check(t, "charts points-2d = Λ", [p_["label"] for p_ in pts] == SUBJ and allclose([[p_["x"], p_["y"]] for p_ in pts], Lam, 1e-12), "一致")


# ================================================================ promax（後で何度も使うので先に計算）
def promax(A, k=4):
    P = A * np.abs(A) ** (k - 1)
    U0 = np.linalg.solve(A.T @ A, A.T @ P)
    d = np.diag(np.linalg.inv(U0.T @ U0))
    U = U0 @ np.diag(np.sqrt(d))
    Phi = np.linalg.inv(U.T @ U)
    return P, U0, U, A @ U, Phi


Pt, U0, Upm, Lpm, Phi = promax(Lam, 4)
phi = Phi[0, 1]
struct = Lpm @ Phi
check(t, "例2 プロマックス φ=0.590, 国語パターン (0.865,−0.072)", close(phi, 0.590, 5e-4) and allclose(Lpm[0], [0.865, -0.072], 5e-4),
      (round(phi, 4), r(Lpm[0])))
check(t, "例2 因子構造 国語 (0.823, 0.438)", allclose(struct[0], [0.823, 0.438], 5e-4), r(struct[0]))
check(t, "例2 構造の式 (0.865−0.072×0.590, 0.865×0.590−0.072)（丸め値で）",
      allclose([0.865 - 0.072 * 0.590, 0.865 * 0.590 - 0.072], [0.823, 0.438], 6e-4), r([0.865 - 0.072 * 0.590, 0.865 * 0.590 - 0.072]))

# ================================================================ communality
t = "communality"
check(t, "例1 h1²=0.68, h2²=0.58, ψ1=0.32", close(h2[0], 0.68, 1e-12) and close(h2[1], 0.58, 1e-12), r(h2[:2]))


def rotT(deg):
    th = math.radians(deg)
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


L30 = Lam @ rotT(30)
check(t, "例1 30°回転で国語 (0.7928, −0.2268)", allclose(L30[0], [0.7928, -0.2268], 5e-5), r(L30[0]))
check(t, "例1 0.7928²=0.6285, 0.2268²=0.0514, 和 0.68", close(0.7928 ** 2, 0.6285, 5e-5) and close(0.2268 ** 2, 0.0514, 5e-5)
      and close((L30[0] ** 2).sum(), 0.68, 1e-12), (round(0.7928 ** 2, 4), round(0.2268 ** 2, 4), (L30[0] ** 2).sum()))
Rinv1 = np.linalg.inv(R1)
smc = 1 - 1 / np.diag(Rinv1)
check(t, "例2 R⁻¹ の対角 (2.6432, 2.2012, 1.7408, 1.4599)", allclose(np.diag(Rinv1), [2.6432, 2.2012, 1.7408, 1.4599], 5e-5), r(np.diag(Rinv1)))
check(t, "例2 SMC (0.6217, 0.5457, 0.4256, 0.3150)", allclose(smc, [0.6217, 0.5457, 0.4256, 0.3150], 5e-5), r(smc))
check(t, "例2 SMC < 真の共通性", bool(np.all(smc < lam1 ** 2)), r(lam1 ** 2))
# SMC = 回帰の R²（直接計算）
r2 = []
for j in range(4):
    idx = [i for i in range(4) if i != j]
    b = np.linalg.solve(R1[np.ix_(idx, idx)], R1[idx, j])
    r2.append(R1[j, idx] @ b)
check(t, "SMC=1−1/r^jj が回帰の重相関係数の2乗と一致", allclose(r2, smc, 1e-12), r(r2))
# 乱数で SMC ≤ 共通性（1因子・2因子モデル）
okk = True
for _ in range(300):
    m = rng.integers(1, 3)
    L_ = rng.uniform(-0.9, 0.9, (6, m))
    hh = (L_ ** 2).sum(1)
    if hh.max() >= 0.98:
        continue
    Rr = L_ @ L_.T + np.diag(1 - hh)
    s_ = 1 - 1 / np.diag(np.linalg.inv(Rr))
    okk &= bool(np.all(s_ <= hh + 1e-12))
check(t, "乱数 300 組（6変数, 1〜2因子）で SMC ≤ 共通性（許容 1e-12）", okk, "成立")
check(t, "例3 h² = 0.64, 0.36, 0.25", allclose([.48 * .40 / .30, .48 * .30 / .40, .40 * .30 / .48], [0.64, 0.36, 0.25], 1e-12), "一致")
check(t, "例3 ヘイウッド 0.6×0.5/0.3 = 1.00", close(0.6 * 0.5 / 0.3, 1.0, 1e-12), 0.6 * 0.5 / 0.3)
ld = charts(t, "line-data")[0]["params"]
check(t, "charts line-data 真の共通性と SMC", allclose(ld["series"][0]["y"], lam1 ** 2, 1e-12)
      and allclose(ld["series"][1]["y"], smc, 5e-5) and ld["x"] == [1, 2, 3, 4], ld["series"][1]["y"])
# 斜交でも ΛΦΛ' の対角は同じ
check(t, "斜交（プロマックス）でも ΛΦΛ' の対角 = 0.68,0.58,…", allclose(np.diag(Lpm @ Phi @ Lpm.T), h2, 1e-12), r(np.diag(Lpm @ Phi @ Lpm.T)))

# ================================================================ factor-rotation / varimax
A_exact = Lam @ (np.array([[1, 1], [1, -1]]) / math.sqrt(2))  # T は対称・直交で T²=I
check("factor-rotation", "例1 初期解 A=ΛT の値 (0.7071,0.4243),(0.7071,0.2828),…",
      allclose(A_exact, [[0.7071, 0.4243], [0.7071, 0.2828], [0.7071, -0.4243], [0.7071, -0.2828]], 5e-5), r(A_exact))
Tm = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
check("factor-rotation", "例1 T は直交行列", allclose(Tm @ Tm.T, np.eye(2), 1e-12), "TT'=I")
check("factor-rotation", "例1 AT = Λ（国語 (0.8,0.2)）", allclose(A_exact @ Tm, Lam, 1e-12), r(A_exact @ Tm))
Ar = np.array([[0.7071, 0.4243], [0.7071, 0.2828], [0.7071, -0.4243], [0.7071, -0.2828]])
check("factor-rotation", "例1 丸め値の A でも AT ≈ Λ（小数第4位）", allclose(Ar @ Tm, Lam, 5e-5), r(Ar @ Tm))
check("factor-rotation", "例1 共通性 回転前 0.5+0.18=0.68 = 回転後", close(0.5 + 0.18, 0.68, 1e-12)
      and allclose((A_exact ** 2).sum(1), h2, 1e-12) and close(A_exact[0, 1] ** 2, 0.18, 1e-12), r((A_exact ** 2).sum(1)))


def vmx(L):
    L2 = L ** 2
    return float(np.sum((L2 ** 2).mean(0) - L2.mean(0) ** 2))


def rot_rows(L, deg):  # JS（rotation-angle）と同じ：L T(θ)
    return L @ rotT(deg)


grid = np.arange(-90, 90.0001, 0.01)
Vg = np.array([vmx(rot_rows(A_exact, g)) for g in grid])
imax = np.where(np.abs(Vg - Vg.max()) < 1e-12)[0]
check("factor-rotation", "例1 V(0)=0.0025, 0.01°刻みで ±45° で最大 0.1313",
      close(vmx(A_exact), 0.0025, 1e-12) and close(Vg.max(), 0.13125, 1e-12) and sorted(set(np.round(grid[imax], 2))) == [-45.0, 45.0],
      (vmx(A_exact), Vg.max(), sorted(set(np.round(grid[imax], 2)))))
check("factor-rotation", "例1 θ=45° の回転で (0.8,−0.2),(0.7,−0.3),(0.2,−0.8),(0.3,−0.7)",
      allclose(rot_rows(A_exact, 45), Lam * [1, -1], 1e-12), r(rot_rows(A_exact, 45)))
check("factor-rotation", "例2 プロマックス φ=0.590, 国語 (0.865,−0.072)", close(phi, 0.590, 5e-4) and allclose(Lpm[0], [0.865, -0.072], 5e-4), round(phi, 4))
# 回転の不定性・斜交の Φ=(T'T)^{-1}
okk = True
for _ in range(200):
    L_ = rng.normal(size=(5, 2))
    Q, _r = np.linalg.qr(rng.normal(size=(2, 2)))
    okk &= np.allclose((L_ @ Q) @ (L_ @ Q).T, L_ @ L_.T, atol=1e-12)
    Tn = rng.normal(size=(2, 2))
    Tinv = np.linalg.inv(Tn)
    Ph = Tinv @ Tinv.T
    okk &= np.allclose(Ph, np.linalg.inv(Tn.T @ Tn), atol=1e-9)
    okk &= np.allclose((L_ @ Tn) @ Ph @ (L_ @ Tn).T, L_ @ L_.T, atol=1e-9)
check("factor-rotation", "乱数 200 組で ΛT(ΛT)'=ΛΛ'、Φ=T⁻¹T⁻¹'=(T'T)⁻¹、(ΛT)Φ(ΛT)'=ΛΛ'（許容 1e-9）", okk, "成立")
for i, cc in enumerate(charts("factor-rotation", "points-2d")):
    pts = cc["params"]["points"]
    xy = np.array([[p_["x"], p_["y"]] for p_ in pts])
    target = A_exact if i == 0 else Lam
    check("factor-rotation", f"charts points-2d #{i + 1}（{'回転前 A' if i == 0 else '回転後 AT'}）",
          [p_["label"] for p_ in pts] == SUBJ and allclose(xy, target, 5e-5), r(xy))
for term in ["factor-rotation", "varimax-rotation"]:
    ra = charts(term, "rotation-angle")[0]
    check(term, "charts rotation-angle の loadings = 初期解 A、labels、angle=0（範囲 −90〜90 内）",
          allclose(ra["params"]["loadings"], A_exact, 5e-5) and ra["params"]["labels"] == SUBJ
          and ra["sliders"][0]["min"] <= ra["params"]["angle"] <= ra["sliders"][0]["max"], ra["params"]["loadings"])
    # JS と同じ計算で ±45° で最大
    Lj = np.array(ra["params"]["loadings"])
    vals = {g: vmx(rot_rows(Lj, g)) for g in range(-90, 91)}
    best = max(vals.values())
    check(term, "charts rotation-angle：整数角で V 最大は ±45°（caption）",
          sorted(g for g, v in vals.items() if v > best - 1e-6) == [-45, 45], round(best, 4))

t = "varimax-rotation"
L2c = A_exact[:, 1] ** 2
check(t, "例 第2列の2乗 0.18,0.08,0.18,0.08 の分散 0.0025; 第1列 0", allclose(L2c, [.18, .08, .18, .08], 1e-12)
      and close(L2c.var(), 0.0025, 1e-12) and close((A_exact[:, 0] ** 2).var(), 0, 1e-12), L2c.var())
u = A_exact[:, 0] ** 2 - A_exact[:, 1] ** 2
v = 2 * A_exact[:, 0] * A_exact[:, 1]
p = 4
check(t, "例 u=(0.32,0.42,0.32,0.42), v=(0.6,0.4,−0.6,−0.4)", allclose(u, [.32, .42, .32, .42], 1e-12) and allclose(v, [.6, .4, -.6, -.4], 1e-12), (r(u), r(v)))
check(t, "例 Σu=1.48, Σv=0, Σuv=0, Σ(u²−v²)=−0.4824",
      close(u.sum(), 1.48, 1e-12) and close(v.sum(), 0, 1e-12) and close((u * v).sum(), 0, 1e-12) and close((u ** 2 - v ** 2).sum(), -0.4824, 1e-12),
      (u.sum(), (u ** 2 - v ** 2).sum()))
num = 2 * (p * (u * v).sum() - u.sum() * v.sum())
den = p * (u ** 2 - v ** 2).sum() - (u.sum() ** 2 - v.sum() ** 2)
check(t, "例 分子 0, 分母 −4.12 → 4θ=180°, θ=45°", close(num, 0, 1e-12) and close(den, -4.12, 1e-12)
      and close(math.degrees(math.atan2(num, den)) / 4, 45, 1e-9), (num, den, math.degrees(math.atan2(num, den)) / 4))


def tan4_theta(L):
    u = L[:, 0] ** 2 - L[:, 1] ** 2
    v = 2 * L[:, 0] * L[:, 1]
    p = len(L)
    num = 2 * (p * (u * v).sum() - u.sum() * v.sum())
    den = p * (u ** 2 - v ** 2).sum() - (u.sum() ** 2 - v.sum() ** 2)
    return math.degrees(math.atan2(num, den)) / 4


okk = True
worst = 0
for _ in range(200):
    L_ = rng.normal(size=(int(rng.integers(3, 9)), 2))
    th = tan4_theta(L_)
    gg = np.arange(-90, 90, 0.05)
    best = max(vmx(rot_rows(L_, g)) for g in gg)
    vt = vmx(rot_rows(L_, th))
    worst = max(worst, best - vt)
    okk &= vt >= best - 1e-6
check(t, "tan4θ 公式（atan2 で象限選択）の θ が、乱数 200 組で格子探索（0.05°）の最大以上（許容 1e-6）", okk, f"最大不足 {worst:.2e}")
# sympy で dV/dθ=0 の条件が公式と一致することを確認
th_ = sp.symbols("theta", real=True)
a = sp.symbols("a1:5", real=True)
b = sp.symbols("b1:5", real=True)
c_, s_ = sp.cos(th_), sp.sin(th_)
rows = [(a[j] * c_ + b[j] * s_, -a[j] * s_ + b[j] * c_) for j in range(4)]
Vs = 0
for k in range(2):
    sq = [rw[k] ** 2 for rw in rows]
    Vs += sum(q ** 2 for q in sq) / 4 - (sum(sq) / 4) ** 2
dV = sp.diff(Vs, th_)
us = [a[j] ** 2 - b[j] ** 2 for j in range(4)]
vs = [2 * a[j] * b[j] for j in range(4)]
NUM = 2 * (4 * sum(us[j] * vs[j] for j in range(4)) - sum(us) * sum(vs))
DEN = 4 * sum(us[j] ** 2 - vs[j] ** 2 for j in range(4)) - (sum(us) ** 2 - sum(vs) ** 2)
# dV/dθ ∝ NUM cos4θ − DEN sin4θ を、乱数値代入で確認
okk = True
for _ in range(20):
    sub = {sym: float(rng.normal()) for sym in a + b}
    sub[th_] = float(rng.uniform(-1, 1))
    lhs = float(dV.subs(sub))
    rhs = float((NUM * sp.cos(4 * th_) - DEN * sp.sin(4 * th_)).subs(sub))
    okk &= abs(lhs - rhs / 16) < 1e-9 * max(1, abs(rhs))
check(t, "sympy: dV/dθ = (分子·cos4θ − 分母·sin4θ)/p²（p=4 で /16, 乱数20点代入, 許容 1e-9）→ 停留点は tan4θ=分子/分母", okk, "成立")
check(t, "例 Λ* の列 2乗平均 0.315, 4乗平均 0.16485, 分散 0.065625, V=0.13125",
      close((Lam[:, 0] ** 2).mean(), 0.315, 1e-12) and close((Lam[:, 0] ** 4).mean(), 0.16485, 1e-12)
      and close((Lam[:, 0] ** 2).var(), 0.065625, 1e-12) and close(vmx(Lam), 0.13125, 1e-12), vmx(Lam))
ld = charts(t, "line-data")[0]["params"]
xs = ld["x"]
Vline = [vmx(rot_rows(A_exact, g)) for g in xs]
check(t, "charts line-data（0〜90° 5°刻み, 19点）= V(θ)（許容 5e-5）", xs == list(range(0, 91, 5)) and allclose(ld["series"][0]["y"], Vline, 5e-5), r(Vline))

# ================================================================ promax-rotation
t = "promax-rotation"
check(t, "例 目標 P（k=4, 符号つき）", allclose(Pt, [[.4096, .0016], [.2401, .0081], [.0016, .4096], [.0081, .2401]], 1e-12), r(Pt))
check(t, "読み解き 0.8⁴=0.41, 0.2⁴=0.0016", close(0.8 ** 4, 0.41, 5e-3) and close(0.2 ** 4, 0.0016, 1e-12), 0.8 ** 4)
check(t, "例 U=[[1.1774,−0.3843],[−0.3843,1.1774]]", allclose(Upm, [[1.1774, -0.3843], [-0.3843, 1.1774]], 5e-5), r(Upm))
check(t, "例 Φ=(U'U)⁻¹ の対角1・φ=0.590", allclose(np.diag(Phi), [1, 1], 1e-12) and close(phi, 0.590, 5e-4), r(Phi))
check(t, "例 Λ=AU（小数第3位）", allclose(Lpm, [[.865, -.072], [.709, .084], [-.072, .865], [.084, .709]], 5e-4), r(Lpm))
check(t, "例 国語行 0.8×1.1774+0.2×(−0.3843)=0.865, 0.8×(−0.3843)+0.2×1.1774=−0.072",
      allclose([0.8 * 1.1774 - 0.2 * 0.3843, -0.8 * 0.3843 + 0.2 * 1.1774], [0.865, -0.072], 5e-4), "一致")
check(t, "例 共通性 0.748+0.005−0.073=0.68（丸め値で）",
      close(0.865 ** 2, 0.748, 5e-4) and close(0.072 ** 2, 0.005, 5e-4) and close(2 * 0.865 * 0.072 * 0.590, 0.073, 5e-4)
      and close(Lpm[0, 0] ** 2 + Lpm[0, 1] ** 2 + 2 * Lpm[0, 0] * Lpm[0, 1] * phi, 0.68, 1e-12),
      Lpm[0, 0] ** 2 + Lpm[0, 1] ** 2 + 2 * Lpm[0, 0] * Lpm[0, 1] * phi)
check(t, "ΛΦΛ' = AA'（Σ 不変）", allclose(Lpm @ Phi @ Lpm.T, Lam @ Lam.T, 1e-12), "一致")
check(t, "構造 国語 (0.823, 0.438)", allclose(struct[0], [0.823, 0.438], 5e-4), r(struct[0]))
pts = charts(t, "points-2d")[0]["params"]["points"]
exp = {"国語（バリマックス）": Lam[0], "数学（バリマックス）": Lam[2], "国語（プロマックス）": Lpm[0],
       "英語（プロマックス）": Lpm[1], "数学（プロマックス）": Lpm[2], "理科（プロマックス）": Lpm[3]}
okk = len(pts) == 6 and all(allclose([p_["x"], p_["y"]], exp[p_["label"]], 5e-4) for p_ in pts)
check(t, "charts points-2d（バリマックス2点・プロマックス4点）", okk, [(p_["label"], p_["x"], p_["y"]) for p_ in pts])
check(t, "charts 英語（プロマックス）0.7089, 0.0842（小数第4位）", allclose(Lpm[1], [0.7089, 0.0842], 5e-5), r(Lpm[1]))
check(t, "caption 因子間相関 0.590", close(phi, 0.590, 5e-4), round(phi, 4))
# R の promax と同じ：カイザー正規化つきバリマックスからでも同じ解（ここでは Λ が varimax 解であることを確認）


def varimax_normalized(L, iters=500):
    hh = np.sqrt((L ** 2).sum(1))
    Ln = L / hh[:, None]
    g = np.arange(-90, 90, 0.001)
    vals = [vmx(rot_rows(Ln, x)) for x in g]
    best = g[int(np.argmax(vals))]
    return rot_rows(Ln, best) * hh[:, None], best


Lv, bestang = varimax_normalized(A_exact)
check(t, "カイザー正規化つきバリマックスでも A から ±45°（列の符号・順序を除き Λ）",
      abs(abs(bestang) - 45) < 1e-3 and (allclose(np.abs(Lv), Lam, 1e-4) or allclose(np.abs(Lv)[:, ::-1], Lam, 1e-4)), (bestang, r(np.abs(Lv))))

# ================================================================ factor-score
t = "factor-score"
psi1 = np.array([0.19, 0.36, 0.51, 0.64])
x = np.array([1.0, 0.5, 0.8, -0.2])
w = lam1 / psi1
s = lam1 @ w
check(t, "λ/ψ = (4.7368, 2.2222, 1.3725, 0.9375)", allclose(w, [4.7368, 2.2222, 1.3725, 0.9375], 5e-5), r(w))
check(t, "s=7.5642", close(s, 7.5642, 5e-5), round(s, 5))
wB, wR = w / s, w / (1 + s)
fB, fR = wB @ x, wR @ x
check(t, "バートレット重み (0.6262,0.2938,0.1815,0.1239), f̂_B=0.893", allclose(wB, [.6262, .2938, .1815, .1239], 5e-5) and close(fB, 0.893, 5e-4), (r(wB), round(fB, 4)))
check(t, "回帰法 重み (0.5531,0.2595,0.1603,0.1095), 1+s=8.5642, f̂_R=0.789",
      allclose(wR, [.5531, .2595, .1603, .1095], 5e-5) and close(1 + s, 8.5642, 5e-5) and close(fR, 0.789, 5e-4), (r(wR), round(fR, 4)))
check(t, "回帰法の各項 0.5531+0.1297+0.1282−0.0219", allclose(wR * x, [.5531, .1297, .1282, -.0219], 5e-5), r(wR * x))
Rinv = np.linalg.inv(R1)
check(t, "Λ'R⁻¹x を直接計算しても 0.789", close(lam1 @ Rinv @ x, 0.789, 5e-4) and close(lam1 @ Rinv @ x, fR, 1e-12), lam1 @ Rinv @ x)
check(t, "比 s/(1+s)=0.883, 回帰法得点の分散 λ'Σ⁻¹λ=0.883, 1/s=0.132",
      close(s / (1 + s), 0.883, 5e-4) and close(lam1 @ Rinv @ lam1, s / (1 + s), 1e-12) and close(1 / s, 0.132, 5e-4) and close(fR / fB, s / (1 + s), 1e-12),
      (s / (1 + s), 1 / s))
# 一般の多因子でウッドベリー・回帰法の2形の一致
okk = True
for _ in range(100):
    L_ = rng.normal(size=(6, 2))
    Ps = np.diag(rng.uniform(0.2, 1, 6))
    Sg = L_ @ L_.T + Ps
    Pi = np.linalg.inv(Ps)
    W1 = L_.T @ np.linalg.inv(Sg)
    W2 = np.linalg.inv(np.eye(2) + L_.T @ Pi @ L_) @ L_.T @ Pi
    okk &= np.allclose(W1, W2, atol=1e-9)
    okk &= np.allclose(np.linalg.inv(Sg), Pi - Pi @ L_ @ np.linalg.inv(np.eye(2) + L_.T @ Pi @ L_) @ L_.T @ Pi, atol=1e-9)
    WB = np.linalg.inv(L_.T @ Pi @ L_) @ L_.T @ Pi
    okk &= np.allclose(WB @ L_, np.eye(2), atol=1e-9)  # 条件付き不偏
check(t, "乱数 100 組（6変数2因子）で回帰法の2形・ウッドベリー公式・バートレットの WΛ=I（許容 1e-9）", okk, "成立")
# シミュレーション：回帰法の MSE ≤ バートレット、バートレットの誤差分散 1/s
nn = 200000
ff = rng.standard_normal(nn)
xx = np.outer(ff, lam1) + rng.standard_normal((nn, 4)) * np.sqrt(psi1)
fb_, fr_ = xx @ wB, xx @ wR
check(t, "シミュレーション（20万人）: Var(f̂_R)≈0.883, Var(f̂_B−f)≈0.132, MSE_R≈1/(1+s) < MSE_B（許容 0.01）",
      abs(fr_.var() - s / (1 + s)) < 0.01 and abs((fb_ - ff).var() - 1 / s) < 0.01 and abs(((fr_ - ff) ** 2).mean() - 1 / (1 + s)) < 0.01
      and ((fr_ - ff) ** 2).mean() < ((fb_ - ff) ** 2).mean(),
      (round(fr_.var(), 4), round((fb_ - ff).var(), 4), round(((fr_ - ff) ** 2).mean(), 4)))
ld = charts(t, "line-data")[0]["params"]
check(t, "charts line-data の重み2系列・caption 0.883", allclose(ld["series"][0]["y"], wR, 5e-5) and allclose(ld["series"][1]["y"], wB, 5e-5), ld["series"][0]["y"])

# ================================================================ number-of-factors
t = "number-of-factors"
ev = np.sort(np.linalg.eigvalsh(R4))[::-1]
check(t, "例1 固有値 (2.3712, 0.8752, 0.3848, 0.3688)", allclose(ev, [2.3712, 0.8752, 0.3848, 0.3688], 5e-5), r(ev))
check(t, "例1 1 を超えるのは1個", int((ev > 1).sum()) == 1, int((ev > 1).sum()))
bd = charts(t, "bar-data")[0]["params"]
check(t, "charts bar-data = 固有値", bd["labels"] == ["1", "2", "3", "4"] and allclose(bd["values"], ev, 5e-5), bd["values"])
S = np.array([[1, .70, .65, .50], [.70, 1, .55, .50], [.65, .55, 1, .40], [.50, .50, .40, 1]])
detS = np.linalg.det(S)
N = 200
c0 = N - 1 - (2 * 4 + 5) / 6
T0 = -c0 * math.log(detS)
check(t, "例2 |S|=0.2004, 199−13/6=196.833, −log|S|=1.6074, T=316.40",
      close(detS, 0.2004, 5e-5) and close(c0, 196.833, 5e-4) and close(-math.log(detS), 1.6074, 5e-4) and close(T0, 316.40, 5e-2),
      (round(detS, 5), round(c0, 4), round(-math.log(detS), 5), round(T0, 3)))
check(t, "例2 m=0 の自由度 6, χ²(6) 上側5%点 12.59", dof(4, 0) == 6 and close(stats.chi2.ppf(0.95, 6), 12.59, 5e-3), stats.chi2.ppf(0.95, 6))
check(t, "例2 m=0 で Σ̂=I（対角モデルの最尤解 diag(S)）", allclose(np.diag(S), [1] * 4, 0), "diag(S)=1")


# ---------- 最尤因子分析（1因子）
def F_ml(Sig, S):
    sgn, ld_ = np.linalg.slogdet(Sig)
    if sgn <= 0:
        return 1e10
    return ld_ + np.trace(S @ np.linalg.inv(Sig)) - np.linalg.slogdet(S)[1] - len(S)


def obj(th):
    l_ = th[:4]
    ps_ = th[4:]
    return F_ml(np.outer(l_, l_) + np.diag(ps_), S)


best = None
for st in range(10):
    x0 = np.concatenate([rng.uniform(0.3, 0.9, 4), rng.uniform(0.2, 0.8, 4)])
    res = optimize.minimize(obj, x0, method="L-BFGS-B", bounds=[(-2, 2)] * 4 + [(1e-4, 2)] * 4,
                            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
    if best is None or res.fun < best.fun:
        best = res
best = optimize.minimize(obj, best.x, method="BFGS", options={"gtol": 1e-12, "maxiter": 10000})  # 仕上げ（制約なしで内点解）
lh = np.abs(best.x[:4])
ph = best.x[4:]
Fmin = best.fun
Sh = np.outer(lh, lh) + np.diag(ph)
cm = N - 1 - (2 * 4 + 5) / 6 - 2 * 1 / 3
Tm1 = cm * Fmin
pv = stats.chi2.sf(Tm1, 2)
check(t, "例2 m=1 最尤 F=log(|Σ̂|/|S|)=0.012343（tr(SΣ̂⁻¹)=p を確認）",
      close(Fmin, 0.012343, 5e-7) and close(np.trace(S @ np.linalg.inv(Sh)), 4, 1e-6)
      and close(np.linalg.slogdet(Sh)[1] - math.log(detS), Fmin, 1e-6), (Fmin, np.trace(S @ np.linalg.inv(Sh))))
check(t, "例2 196.167×0.012343=2.421, p=0.298, 5.99", close(cm, 196.167, 5e-4) and close(Tm1, 2.421, 5e-4) and close(pv, 0.298, 5e-4)
      and close(stats.chi2.ppf(0.95, 2), 5.99, 5e-3), (round(cm, 4), round(Tm1, 4), round(pv, 4)))

# ================================================================ maximum-likelihood-factor-analysis
t = "maximum-likelihood-factor-analysis"
check(t, "λ̂=(0.888,0.791,0.718,0.580)", allclose(lh, [.888, .791, .718, .580], 5e-4), r(lh))
check(t, "Ψ̂=diag(0.211,0.375,0.485,0.664)", allclose(ph, [.211, .375, .485, .664], 5e-4), r(ph))
check(t, "性質1 対角一致 λ̂²+ψ̂=1（0.789+0.211）", allclose(np.diag(Sh), [1] * 4, 1e-5) and close(lh[0] ** 2, 0.789, 5e-4), r(np.diag(Sh), 6))
check(t, "r̂12=0.702, 差 −0.002; 最大残差 2-4 で 0.50−0.459=0.041",
      close(Sh[0, 1], 0.702, 5e-4) and close(S[0, 1] - Sh[0, 1], -0.002, 5e-4) and close(Sh[1, 3], 0.459, 5e-4)
      and close(S[1, 3] - Sh[1, 3], 0.041, 5e-4), (round(Sh[0, 1], 4), round(Sh[1, 3], 4)))
iu = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
resid = np.array([S[i, j] - Sh[i, j] for i, j in iu])
check(t, "最大残差の組が 2-4", iu[int(np.argmax(np.abs(resid)))] == (1, 3), r(resid))
check(t, "F=0.012343, 自由度 2, T=2.421, 5.991, p=0.298",
      close(Fmin, 0.012343, 5e-7) and dof(4, 1) == 2 and close(Tm1, 2.421, 5e-4) and close(stats.chi2.ppf(0.95, 2), 5.991, 5e-4) and close(pv, 0.298, 5e-4),
      (round(Fmin, 7), round(Tm1, 4), round(stats.chi2.ppf(0.95, 2), 4), round(pv, 4)))
bd = charts(t, "bar-data")[0]["params"]
check(t, "charts bar-data 残差6組（小数第4位, 許容 1e-4）・すべて ±0.05 以内",
      bd["labels"] == ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"] and allclose(bd["values"], resid, 1e-4) and np.all(np.abs(resid) < 0.05),
      r(resid))
# 尤度比 = nF の導出：対数尤度の差
ll = lambda Sig: -N / 2 * (np.linalg.slogdet(Sig)[1] + np.trace(S @ np.linalg.inv(Sig)) + 4 * math.log(2 * math.pi))
check(t, "2(ℓ(S)−ℓ(Σ̂)) = nF", close(2 * (ll(S) - ll(Sh)), N * Fmin, 1e-8), 2 * (ll(S) - ll(Sh)))
check(t, "F(S)=0", close(F_ml(S, S), 0, 1e-12), F_ml(S, S))
# 尺度不変：共分散行列（単位を変える）でも標準化解が同じ
Dg = np.diag([2.0, 0.5, 3.0, 1.5])
S2 = Dg @ S @ Dg
obj2 = lambda th: F_ml(np.outer(th[:4], th[:4]) + np.diag(th[4:]), S2)
x0 = np.concatenate([np.diag(Dg) * lh, np.diag(Dg) ** 2 * ph])
res2 = optimize.minimize(obj2, x0 * 1.1, method="L-BFGS-B", bounds=[(-10, 10)] * 4 + [(1e-4, 20)] * 4,
                         options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
check(t, "性質2 尺度不変（D S D で λ̂ が D 倍、F 同じ, 許容 1e-4）",
      allclose(np.abs(res2.x[:4]) / np.diag(Dg), lh, 1e-4) and close(res2.fun, Fmin, 1e-8), r(np.abs(res2.x[:4]) / np.diag(Dg)))
# シミュレーション：1因子モデルが正しいとき T の分布 ≈ χ²(2)（小規模）
Ttrue = []
Ltrue = lh
Strue = np.outer(Ltrue, Ltrue) + np.diag(1 - Ltrue ** 2)
Lc = np.linalg.cholesky(Strue)
for _ in range(400):
    Z = rng.standard_normal((N, 4)) @ Lc.T
    Ss = np.cov(Z.T, bias=True)
    o = lambda th: F_ml(np.outer(th[:4], th[:4]) + np.diag(th[4:]), Ss)
    rr = optimize.minimize(o, np.concatenate([Ltrue * np.sqrt(np.diag(Ss)), np.diag(Ss) * (1 - Ltrue ** 2)]),
                           method="L-BFGS-B", bounds=[(-3, 3)] * 4 + [(1e-4, 3)] * 4)
    Ttrue.append(cm * rr.fun)
Ttrue = np.array(Ttrue)
check(t, "シミュレーション（400回, n=200）: T の平均 ≈ 2（許容 0.3）・棄却率 ≈ 0.05（許容 0.03）",
      abs(Ttrue.mean() - 2) < 0.3 and abs((Ttrue > stats.chi2.ppf(0.95, 2)).mean() - 0.05) < 0.03,
      (round(Ttrue.mean(), 3), (Ttrue > stats.chi2.ppf(0.95, 2)).mean()))

# ================================================================ graphical-model
t = "graphical-model"
Om3 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], float)
Sg3 = np.linalg.inv(Om3)
check(t, "例1 Σ=Ω⁻¹=(1/4)[[3,2,1],[2,4,2],[1,2,3]]", allclose(Sg3, np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]]) / 4, 1e-12), r(Sg3 * 4))
Cr = Sg3 / np.sqrt(np.outer(np.diag(Sg3), np.diag(Sg3)))
check(t, "例1 r12=r23=2/√12=0.577, r13=1/3", close(Cr[0, 1], 2 / math.sqrt(12), 1e-12) and close(Cr[0, 1], 0.577, 5e-4)
      and close(Cr[1, 2], Cr[0, 1], 1e-12) and close(Cr[0, 2], 1 / 3, 1e-12), r(Cr))
P3 = pcorr_from_prec(Om3)
check(t, "例1 偏相関 ρ12·3=0.5, ρ23·1=0.5, ρ13·2=0", allclose([P3[0, 1], P3[1, 2], P3[0, 2]], [.5, .5, 0], 1e-12), r(P3))
check(t, "例1 r13−r12 r23 = 1/3−4/12 = 0", close(Cr[0, 2] - Cr[0, 1] * Cr[1, 2], 0, 1e-12), Cr[0, 2] - Cr[0, 1] * Cr[1, 2])
Sc = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 3]], float)
Omc = np.linalg.inv(Sc)
check(t, "例2 合流 Σ と Ω=[[2,1,−1],[1,2,−1],[−1,−1,1]]", allclose(Omc, [[2, 1, -1], [1, 2, -1], [-1, -1, 1]], 1e-12), r(Omc))
check(t, "例2 ρ12·3=−0.5、Ω に0なし", close(pcorr_from_prec(Omc)[0, 1], -0.5, 1e-12) and np.all(np.abs(Omc) > 1e-9), pcorr_from_prec(Omc)[0, 1])
nn = 400000
z1, z2, ee = rng.standard_normal((3, nn))
z3 = z1 + z2 + ee
Zc = np.vstack([z1, z2, z3])
Cs = np.cov(Zc)
Pc = pcorr_from_prec(np.linalg.inv(Cs))
check(t, "シミュレーション（40万点）: X3=X1+X2+e で r12≈0, ρ12·3≈−0.5, Σ≈[[1,0,1],[0,1,1],[1,1,3]]（許容 0.01）",
      abs(np.corrcoef(Zc)[0, 1]) < 0.01 and abs(Pc[0, 1] + 0.5) < 0.01 and np.abs(Cs - Sc).max() < 0.02,
      (round(np.corrcoef(Zc)[0, 1], 4), round(Pc[0, 1], 4)))
check(t, "パラメータ数 2^10−1=1023, 1+9×2=19", 2 ** 10 - 1 == 1023 and 1 + 9 * 2 == 19, (1023, 19))
g = charts(t, "graph-network")
check(t, "charts #1 無向 X1−X2−X3, 辺ラベル 0.5,0.5", g[0]["params"]["edges"] == [[0, 1], [1, 2]] and g[0]["params"]["directed"] == 0
      and allclose([float(e_) for e_ in g[0]["params"]["edgeLabels"]], [P3[0, 1], P3[1, 2]], 1e-12), g[0]["params"]["edgeLabels"])
check(t, "charts #2 有向 X1→X3←X2, caption の 1/3・−0.5", g[1]["params"]["edges"] == [[0, 2], [1, 2]] and g[1]["params"]["directed"] == 1, g[1]["params"]["edges"])

# ================================================================ undirected-graphical-model / precision-matrix
Rc = np.array([[1, .6, .3, .24], [.6, 1, .5, .4], [.3, .5, 1, .8], [.24, .4, .8, 1]])
Oc = np.linalg.inv(Rc)
Pcc = pcorr_from_prec(Oc)
t = "precision-matrix"
check(t, "例2 R⁻¹（小数第4位）", allclose(Oc, [[1.5625, -.9375, 0, 0], [-.9375, 1.8958, -.6667, 0], [0, -.6667, 3.1111, -2.2222], [0, 0, -2.2222, 2.7778]], 5e-5), r(Oc))
check(t, "例2 (1,3),(1,4),(2,4) がちょうど 0（許容 1e-12）", allclose([Oc[0, 2], Oc[0, 3], Oc[1, 3]], [0, 0, 0], 1e-12), r([Oc[0, 2], Oc[0, 3], Oc[1, 3]], 15))
check(t, "例2 偏相関 0.545, 0.275, 0.756", allclose([Pcc[0, 1], Pcc[1, 2], Pcc[2, 3]], [.545, .275, .756], 5e-4), r([Pcc[0, 1], Pcc[1, 2], Pcc[2, 3]]))
check(t, "例2 丸め値での式 0.9375/√(1.5625×1.8958) 等", allclose([0.9375 / math.sqrt(1.5625 * 1.8958), 0.6667 / math.sqrt(1.8958 * 3.1111), 2.2222 / math.sqrt(3.1111 * 2.7778)],
                                                                   [.545, .275, .756], 5e-4), "一致")
check(t, "例1 r13=(1/4)/√(3/4·3/4)=1/3, ρ12=0.5", close(0.25 / math.sqrt(0.75 * 0.75), 1 / 3, 1e-12) and close(P3[0, 1], 0.5, 1e-12), "一致")
b = np.linalg.solve(Sg3[1:, 1:], Sg3[1:, 0])
resv = Sg3[0, 0] - Sg3[0, 1:] @ b
check(t, "例1 X1 を (X2,X3) に回帰: 係数 (0.5, 0), 残差分散 0.5 = 1/ω11 = −ω1j/ω11",
      allclose(b, [0.5, 0], 1e-12) and close(resv, 0.5, 1e-12) and allclose(b, -Om3[0, 1:] / Om3[0, 0], 1e-12), (r(b), resv))
# 一般: 偏相関の式・回帰係数・条件付き分散を乱数で
okk = True
for _ in range(200):
    p_ = int(rng.integers(3, 7))
    M = rng.normal(size=(p_, p_ + 2))
    Sg = M @ M.T
    Om = np.linalg.inv(Sg)
    i, j = 0, 1
    rest = list(range(2, p_))
    Sa = Sg[np.ix_([i, j], [i, j])] - Sg[np.ix_([i, j], rest)] @ np.linalg.solve(Sg[np.ix_(rest, rest)], Sg[np.ix_(rest, [i, j])])
    pc = Sa[0, 1] / math.sqrt(Sa[0, 0] * Sa[1, 1])
    okk &= abs(pc + Om[0, 1] / math.sqrt(Om[0, 0] * Om[1, 1])) < 1e-9
    okk &= np.allclose(Sa, np.linalg.inv(Om[np.ix_([i, j], [i, j])]), atol=1e-8)
    oth = list(range(1, p_))
    bb = np.linalg.solve(Sg[np.ix_(oth, oth)], Sg[oth, 0])
    okk &= np.allclose(bb, -Om[0, oth] / Om[0, 0], atol=1e-8)
    okk &= abs(Sg[0, 0] - Sg[0, oth] @ bb - 1 / Om[0, 0]) < 1e-8
check(t, "乱数 200 組で ρ=−ω_ij/√(ω_ii ω_jj)・条件付き共分散=Ω_aa⁻¹・回帰係数 −ω_ij/ω_ii・残差分散 1/ω_ii（許容 1e-8）", okk, "成立")
# 3変数公式との一致（sympy）
r12, r13, r23 = sp.symbols("r12 r13 r23", real=True)
Rs = sp.Matrix([[1, r12, r13], [r12, 1, r23], [r13, r23, 1]])
Ri = Rs.inv()
pc_prec = -Ri[0, 1] / sp.sqrt(Ri[0, 0] * Ri[1, 1])
pc_form = (r12 - r13 * r23) / sp.sqrt((1 - r13 ** 2) * (1 - r23 ** 2))
vals_ok = all(abs(float((pc_prec - pc_form).subs({r12: a1, r13: a2, r23: a3}))) < 1e-12
              for a1, a2, a3 in [(0.6, 0.4, 0.5), (0.3, -0.2, 0.7), (-0.5, 0.1, 0.2)])
num_ok = sp.simplify(-Ri[0, 1] * Rs.det() - (r12 - r13 * r23)) == 0
check(t, "sympy: 3変数で −r^12/√(r^11 r^22) = (r12−r13r23)/√((1−r13²)(1−r23²))（分子が恒等的に一致・数値3点）", num_ok and vals_ok, "成立")
# 2×2 逆行列の符号（sympy）
wi, wj, wij = sp.symbols("w_ii w_jj w_ij", positive=True)
Ai = sp.Matrix([[wi, wij], [wij, wj]]).inv()
dd = sp.symbols("Delta", positive=True)  # 正定値なら ω_ii ω_jj − ω_ij² > 0
exprc = (Ai[0, 1] / sp.sqrt(Ai[0, 0] * Ai[1, 1])).subs(wi * wj - wij ** 2, dd)
exprc = sp.simplify(exprc.subs(wj, (dd + wij ** 2) / wi))
check(t, "sympy: 2×2 逆行列の相関 = −ω_ij/√(ω_ii ω_jj)（正定値で）", sp.simplify(exprc + wij / sp.sqrt(dd + wij ** 2)) == 0, exprc)
check(t, "SMC=1−1/r^ii（例2 の R で回帰 R² と一致）",
      all(close(1 - 1 / Oc[k, k], Rc[k, [q for q in range(4) if q != k]] @ np.linalg.solve(Rc[np.ix_([q for q in range(4) if q != k], [q for q in range(4) if q != k])], Rc[[q for q in range(4) if q != k], k]), 1e-12) for k in range(4)),
      r(1 - 1 / np.diag(Oc)))
gn = charts(t, "graph-network")[0]["params"]
check(t, "charts graph-network 辺 1-2,2-3,3-4・ラベル偏相関", gn["edges"] == [[0, 1], [1, 2], [2, 3]]
      and allclose([float(e_) for e_ in gn["edgeLabels"]], [Pcc[0, 1], Pcc[1, 2], Pcc[2, 3]], 5e-4), gn["edgeLabels"])
bd = charts(t, "bar-data")[0]["params"]
check(t, "charts bar-data 偏相関6組（小数第4位）", bd["labels"] == ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"]
      and allclose(bd["values"], [Pcc[i, j] for i, j in iu], 5e-5), r([Pcc[i, j] for i, j in iu]))
check(t, "caption 通常の相関 1-3=0.30, 1-4=0.24, 2-4=0.40", allclose([Rc[0, 2], Rc[0, 3], Rc[1, 3]], [.3, .24, .4], 1e-12), "一致")

t = "undirected-graphical-model"
check(t, "例1 R⁻¹ の (1,3),(1,4),(2,4)=0、隣どうしは非0", allclose([Oc[0, 2], Oc[0, 3], Oc[1, 3]], [0, 0, 0], 1e-12)
      and min(abs(Oc[0, 1]), abs(Oc[1, 2]), abs(Oc[2, 3])) > 0.1, "連鎖")


def pc_given(Sg, i, j, cond):
    idx = [i, j] + list(cond)
    O = np.linalg.inv(Sg[np.ix_(idx, idx)])
    return -O[0, 1] / math.sqrt(O[0, 0] * O[1, 1])


check(t, "例1 ρ14·2=(0.24−0.24)/√(0.64×0.84)=0, ρ13·2=0", close(pc_given(Rc, 0, 3, [1]), 0, 1e-12) and close(pc_given(Rc, 0, 2, [1]), 0, 1e-12)
      and close((0.24 - 0.6 * 0.4) / math.sqrt(0.64 * 0.84), 0, 1e-12), (pc_given(Rc, 0, 3, [1]), pc_given(Rc, 0, 2, [1])))
check(t, "例1 X1 ⊥ {X3,X4} | X2（条件付き共分散ブロックが 0）",
      allclose((Rc[np.ix_([0], [2, 3])] - Rc[np.ix_([0], [1])] @ Rc[np.ix_([1], [2, 3])]), [0, 0], 1e-12), "成立")
check(t, "例1 ρ14·23=0（対ごとのマルコフ性）", close(pc_given(Rc, 0, 3, [1, 2]), 0, 1e-12), pc_given(Rc, 0, 3, [1, 2]))
check(t, "例2 偏相関 0.545, 0.275, 0.756; r23=0.5 > 0.275", allclose([Pcc[0, 1], Pcc[1, 2], Pcc[2, 3]], [.545, .275, .756], 5e-4) and Rc[1, 2] > Pcc[1, 2], "一致")
gn = charts(t, "graph-network")[0]["params"]
check(t, "charts graph-network 辺・ラベル", gn["edges"] == [[0, 1], [1, 2], [2, 3]] and gn["nodes"] == ["X1", "X2", "X3", "X4"]
      and allclose([float(e_) for e_ in gn["edgeLabels"]], [Pcc[0, 1], Pcc[1, 2], Pcc[2, 3]], 5e-4), gn["edgeLabels"])
# 大域的マルコフ性：乱数の連鎖精度行列で分離集合の条件付き独立
okk = True
for _ in range(100):
    d1, d2, d3 = rng.uniform(-0.45, 0.45, 3)
    Om = np.eye(5)
    for k, val in enumerate(rng.uniform(-0.45, 0.45, 4)):
        Om[k, k + 1] = Om[k + 1, k] = val
    if np.linalg.eigvalsh(Om).min() <= 0.05:
        continue
    Sg = np.linalg.inv(Om)
    okk &= abs(pc_given(Sg, 0, 4, [2])) < 1e-9 and abs(pc_given(Sg, 1, 4, [3])) < 1e-9 and abs(pc_given(Sg, 0, 3, [1])) < 1e-9
check(t, "乱数 100 組（5変数の連鎖）で分離集合1個を与えた偏相関が 0（許容 1e-9）", okk, "成立")

# ================================================================ covariance-selection
t = "covariance-selection"
S3 = np.array([[1, .6, .4], [.6, 1, .5], [.4, .5, 1]])
P3s = pcorr_from_prec(np.linalg.inv(S3))
check(t, "例 標本偏相関 0.504, 0.355, 0.144（公式 0.1443）",
      allclose([P3s[0, 1], P3s[1, 2], P3s[0, 2]], [.504, .355, .144], 5e-4) and close((0.4 - 0.3) / math.sqrt(0.64 * 0.75), 0.1443, 5e-5)
      and close(P3s[0, 2], 0.1443, 5e-5), r([P3s[0, 1], P3s[1, 2], P3s[0, 2]]))
Sh3 = S3.copy()
Sh3[0, 2] = Sh3[2, 0] = S3[0, 1] * S3[1, 2] / S3[1, 1]
check(t, "例 σ̂13=0.30, Σ̂⁻¹ の (1,3)=0", close(Sh3[0, 2], 0.30, 1e-12) and close(np.linalg.inv(Sh3)[0, 2], 0, 1e-12), np.linalg.inv(Sh3)[0, 2])
dS, dH = np.linalg.det(S3), np.linalg.det(Sh3)
dev = 100 * math.log(dH / dS)
dev2 = -100 * math.log(1 - P3s[0, 2] ** 2)
check(t, "例 |S|=0.47, |Σ̂|=0.48, log(0.48/0.47)=0.02105, dev=2.105",
      close(dS, 0.47, 1e-12) and close(dH, 0.48, 1e-12) and close(math.log(dH / dS), 0.02105, 5e-6) and close(dev, 2.105, 5e-4), (dS, dH, round(dev, 5)))
check(t, "例 −100 log(1−0.1443²)=−100 log(0.97917)=2.105（両形が一致, 許容 1e-10）",
      close(dev2, dev, 1e-10) and close(1 - P3s[0, 2] ** 2, 0.97917, 5e-6), (round(dev2, 6), round(1 - P3s[0, 2] ** 2, 6)))
check(t, "例 χ²(1) 上側5%点 3.841, p=0.147", close(stats.chi2.ppf(0.95, 1), 3.841, 5e-4) and close(stats.chi2.sf(dev, 1), 0.147, 5e-4),
      (round(stats.chi2.ppf(0.95, 1), 4), round(stats.chi2.sf(dev, 1), 4)))
check(t, "tr(SΣ̂⁻¹)=p", close(np.trace(S3 @ np.linalg.inv(Sh3)), 3, 1e-12), np.trace(S3 @ np.linalg.inv(Sh3)))
# Σ̂ が最尤解：ω13=0 の制約下で数値最大化と一致
def negll(th):
    Om = np.array([[th[0], th[3], 0], [th[3], th[1], th[4]], [0, th[4], th[2]]])
    sg, ld_ = np.linalg.slogdet(Om)
    if sg <= 0:
        return 1e10
    return -ld_ + np.trace(S3 @ Om)


ro = optimize.minimize(negll, [1.5, 1.5, 1.5, -0.3, -0.3], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 40000})
Omh = np.array([[ro.x[0], ro.x[3], 0], [ro.x[3], ro.x[1], ro.x[4]], [0, ro.x[4], ro.x[2]]])
check(t, "ω13=0 制約付きの最尤数値解 = Σ̂（許容 1e-5）", allclose(np.linalg.inv(Omh), Sh3, 1e-5), r(np.linalg.inv(Omh), 5))
# 一般の p で 1辺除去: dev = −n log(1−r²)（反復比例スケーリングを自前実装して確認）


def ips_remove_edge(S, i, j, iters=2000):
    p_ = len(S)
    cliques = [[k for k in range(p_) if k != j], [k for k in range(p_) if k != i]]
    K = np.eye(p_)
    for _ in range(iters):
        for Cq in cliques:
            Sg = np.linalg.inv(K)
            Cn = [k for k in range(p_) if k not in Cq]
            Kcc_new = np.linalg.inv(S[np.ix_(Cq, Cq)]) + K[np.ix_(Cq, Cn)] @ np.linalg.solve(K[np.ix_(Cn, Cn)], K[np.ix_(Cn, Cq)])
            K[np.ix_(Cq, Cq)] = Kcc_new
    return np.linalg.inv(K)


okk = True
worst = 0
for _ in range(30):
    p_ = int(rng.integers(3, 6))
    M = rng.normal(size=(p_, 30))
    Ss = np.cov(M)
    Ss = Ss / np.sqrt(np.outer(np.diag(Ss), np.diag(Ss)))
    Sh_ = ips_remove_edge(Ss, 0, 1, 300)
    Kh = np.linalg.inv(Sh_)
    rp = pcorr_from_prec(np.linalg.inv(Ss))[0, 1]
    d1_ = math.log(np.linalg.det(Sh_) / np.linalg.det(Ss))
    d2_ = -math.log(1 - rp ** 2)
    mask = np.ones((p_, p_), bool)
    mask[0, 1] = mask[1, 0] = False
    worst = max(worst, abs(d1_ - d2_))
    okk &= abs(Kh[0, 1]) < 1e-8 and np.allclose(Sh_[mask], Ss[mask], atol=1e-8) and abs(d1_ - d2_) < 1e-8
check(t, "乱数 30 組（3〜5変数, 反復比例スケーリング）で 1辺除去の log(|Σ̂|/|S|) = −log(1−r²)（許容 1e-8）", okk, f"最大差 {worst:.1e}")
# 帰無仮説のもとで dev ~ χ²(1) のシミュレーション
devs = []
Ltrue = np.linalg.cholesky(Sh3)
for _ in range(20000):
    Z = rng.standard_normal((100, 3)) @ Ltrue.T
    Ss = np.cov(Z.T, bias=True)
    rr = pcorr_from_prec(np.linalg.inv(Ss))[0, 2]
    devs.append(-100 * math.log(1 - rr ** 2))
devs = np.array(devs)
check(t, "シミュレーション（2万回, n=100, ω13=0）: dev の平均 ≈1（許容 0.05）・棄却率 ≈0.05（許容 0.008）",
      abs(devs.mean() - 1) < 0.05 and abs((devs > 3.841).mean() - 0.05) < 0.008, (round(devs.mean(), 4), round((devs > 3.841).mean(), 4)))
gn = charts(t, "graph-network")[0]["params"]
check(t, "charts graph-network 辺 1-2, 2-3・ラベル 0.504/0.355", gn["edges"] == [[0, 1], [1, 2]]
      and allclose([float(e_) for e_ in gn["edgeLabels"]], [P3s[0, 1], P3s[1, 2]], 5e-4), gn["edgeLabels"])
check(t, "グラフ・イメージ 辺のない組の相関 0.30 = 0.6×0.5", close(Sh3[0, 2], 0.6 * 0.5, 1e-12), Sh3[0, 2])

# ================================================================ directed-acyclic-graph
t = "directed-acyclic-graph"
pT = {1: 0.4, 0: 0.6}
pI = {1: 0.8, 0: 0.3}
pH = {1: 0.2, 0: 0.05}
joint = {}
for tt in (0, 1):
    for ii in (0, 1):
        for hh in (0, 1):
            joint[(tt, ii, hh)] = pT[tt] * (pI[tt] if ii else 1 - pI[tt]) * (pH[tt] if hh else 1 - pH[tt])
PI1 = sum(v_ for k, v_ in joint.items() if k[1] == 1)
PH1 = sum(v_ for k, v_ in joint.items() if k[2] == 1)
PIH = sum(v_ for k, v_ in joint.items() if k[1] == 1 and k[2] == 1)
PI0H1 = sum(v_ for k, v_ in joint.items() if k[1] == 0 and k[2] == 1)
check(t, "例1 P(I=1)=0.5, P(H=1)=0.11, P(I=1,H=1)=0.073 (0.064+0.009)",
      close(PI1, 0.5, 1e-12) and close(PH1, 0.11, 1e-12) and close(PIH, 0.073, 1e-12), (PI1, PH1, PIH))
check(t, "例1 P(I)P(H)=0.055, P(H|I=1)=0.146, P(H|I=0)=0.037/0.5=0.074",
      close(PI1 * PH1, 0.055, 1e-12) and close(PIH / PI1, 0.146, 1e-12) and close(PI0H1, 0.037, 1e-12) and close(PI0H1 / (1 - PI1), 0.074, 1e-12),
      (PI1 * PH1, PIH / PI1, PI0H1 / (1 - PI1)))
check(t, "例1 P(I=1,H=1|T=1)=0.16=0.8×0.2（T=0 でも積）", close(joint[(1, 1, 1)] / 0.4, 0.16, 1e-12)
      and all(close(joint[(tt, ii, hh)] / pT[tt], (sum(joint[(tt, ii, x_)] for x_ in (0, 1)) / pT[tt]) * (sum(joint[(tt, x_, hh)] for x_ in (0, 1)) / pT[tt]), 1e-12)
              for tt in (0, 1) for ii in (0, 1) for hh in (0, 1)), joint[(1, 1, 1)] / 0.4)
check(t, "例1 パラメータ 1+2+2=5 < 2³−1=7", 1 + 2 + 2 == 5 and 2 ** 3 - 1 == 7, (5, 7))
check(t, "例2 Ω=[[2,1,−1],[1,2,−1],[−1,−1,1]], ρ12·3=−0.5", allclose(Omc, [[2, 1, -1], [1, 2, -1], [-1, -1, 1]], 1e-12)
      and close(pcorr_from_prec(Omc)[0, 1], -0.5, 1e-12), "一致")
# シミュレーション：例1 の分岐
nn = 400000
Ts = rng.random(nn) < 0.4
Is = rng.random(nn) < np.where(Ts, 0.8, 0.3)
Hs = rng.random(nn) < np.where(Ts, 0.2, 0.05)
check(t, "シミュレーション（40万日）: P(H|I=1)≈0.146, P(H|I=0)≈0.074, P(I,H|T=1)≈0.16（許容 0.005）",
      abs(Hs[Is].mean() - 0.146) < 0.005 and abs(Hs[~Is].mean() - 0.074) < 0.005 and abs((Is & Hs)[Ts].mean() - 0.16) < 0.005,
      (round(Hs[Is].mean(), 4), round(Hs[~Is].mean(), 4), round((Is & Hs)[Ts].mean(), 4)))
gn = charts(t, "graph-network")[0]["params"]
check(t, "charts graph-network 気温→アイス売上, 気温→熱中症搬送（有向）", gn["nodes"] == ["気温", "アイス売上", "熱中症搬送"]
      and gn["edges"] == [[0, 1], [0, 2]] and gn["directed"] == 1, gn["edges"])

# ================================================================ 重要度Cの用語
t = "graphical-lasso"
Sg3 = np.array([[1, 0.6, 0.4], [0.6, 1, 0.5], [0.4, 0.5, 1.0]])


def glasso_admm(S, lam, rho=1.0, iters=20000):
    """対角に罰則をかけない graphical lasso を ADMM で解く（自作）。"""
    p = len(S)
    Zq, Uq = np.eye(p), np.zeros((p, p))
    M = lam * (1 - np.eye(p))
    for _ in range(iters):
        w, Q = np.linalg.eigh(rho * (Zq - Uq) - S)
        Xq = Q @ np.diag((w + np.sqrt(w ** 2 + 4 * rho)) / (2 * rho)) @ Q.T
        A = Xq + Uq
        Zq = np.sign(A) * np.maximum(np.abs(A) - M / rho, 0)
        Uq = Uq + Xq - Zq
    return Zq


def glasso_obj(Om, S, lam):
    return np.linalg.slogdet(Om)[1] - np.trace(S @ Om) - lam * np.abs(Om[~np.eye(len(S), dtype=bool)]).sum()


def kkt(Om, S, lam, tol=1e-6):
    W = np.linalg.inv(Om)
    ok = allclose(np.diag(W), np.diag(S), tol)
    for j in range(3):
        for k in range(j + 1, 3):
            if abs(Om[j, k]) > 1e-8:
                ok &= abs(W[j, k] - (S[j, k] + lam * np.sign(Om[j, k]))) < tol
            else:
                ok &= abs(W[j, k] - S[j, k]) <= lam + tol
    return bool(ok)


pc3 = lambda O: [-O[a, b] / math.sqrt(O[a, a] * O[b, b]) for a, b in [(0, 1), (1, 2), (0, 2)]]
check(t, "標本偏相関 0.504, 0.355, 0.144（共分散選択の項目と同じ S）", allclose(pc3(np.linalg.inv(Sg3)), [0.504, 0.355, 0.144], 5e-4), r(pc3(np.linalg.inv(Sg3))))
check(t, "λ=0 なら Ω̂ = S⁻¹", allclose(glasso_admm(Sg3, 0.0, iters=3000), np.linalg.inv(Sg3), 1e-8), "一致")
O2 = glasso_admm(Sg3, 0.2)
W2 = np.linalg.inv(O2)
check(t, "λ=0.2：辺はすべて残り Ŵ 非対角 0.4, 0.3, 0.2（s−0.2）", np.all(np.abs(O2[np.triu_indices(3, 1)]) > 1e-3) and allclose([W2[0, 1], W2[1, 2], W2[0, 2]], [0.4, 0.3, 0.2], 1e-6), r([W2[0, 1], W2[1, 2], W2[0, 2]]))
check(t, "λ=0.2：偏相関 0.364, 0.245, 0.092", allclose(pc3(O2), [0.364, 0.245, 0.092], 5e-4), r(pc3(O2)))
O4 = glasso_admm(Sg3, 0.4)
W4 = np.linalg.inv(O4)
check(t, "λ=0.4：ω̂13=0、ŵ12=0.2、ŵ23=0.1、ŵ13=0.02", abs(O4[0, 2]) < 1e-8 and allclose([W4[0, 1], W4[1, 2], W4[0, 2]], [0.2, 0.1, 0.02], 1e-6), r([W4[0, 1], W4[1, 2], W4[0, 2]]))
check(t, "λ=0.4：|0.02−0.4|=0.38 ≤ 0.4、偏相関 0.199, 0.098", close(abs(W4[0, 2] - 0.4), 0.38, 1e-6) and allclose(pc3(O4)[:2], [0.199, 0.098], 5e-4), r(pc3(O4)))
O5 = glasso_admm(Sg3, 0.5)
check(t, "λ=0.5：辺 1-2 だけが残り偏相関 0.1", abs(O5[0, 1]) > 1e-3 and abs(O5[1, 2]) < 1e-8 and abs(O5[0, 2]) < 1e-8 and close(pc3(O5)[0], 0.1, 1e-6), r(pc3(O5)))
O6 = glasso_admm(Sg3, 0.6)
O7 = glasso_admm(Sg3, 0.75)
check(t, "λ≥0.6（=max|s_jk|）で辺なし（λ=0.6, 0.75）", all(np.all(np.abs(O[np.triu_indices(3, 1)]) < 1e-8) for O in (O6, O7)), "成立")
check(t, "KKT 条件 ŵ=s+λ·sign(ω̂)（非0）、|ŵ−s|≤λ（0）が λ=0.2, 0.4, 0.5, 0.6 で成立（許容1e-6）",
      all(kkt(O, Sg3, l) for O, l in [(O2, 0.2), (O4, 0.4), (O5, 0.5), (O6, 0.6)]), "成立")
check(t, "本文の式の符号：正の相関では ω̂<0 なので ŵ = s − λ（= s + λ·sign(ω̂)）", O2[0, 1] < 0 and close(W2[0, 1], Sg3[0, 1] + 0.2 * np.sign(O2[0, 1]), 1e-6), r(O2[0, 1]))
okopt = True
for _ in range(500):
    Pm = O4 + 0.05 * (lambda E: E + E.T)(rng.normal(size=(3, 3)))
    if np.linalg.eigvalsh(Pm).min() > 0:
        okopt &= glasso_obj(Pm, Sg3, 0.4) <= glasso_obj(O4, Sg3, 0.4) + 1e-10
check(t, "λ=0.4 の解の近傍 500 点で目的関数が解を超えない（乱数）", okopt, "成立")
lam_b = (0.1 + math.sqrt(0.41)) / 2
lsym = sp.symbols("lam", positive=True)
roots = sp.solve(sp.Eq((sp.Rational(6, 10) - lsym) * (sp.Rational(5, 10) - lsym), sp.Rational(4, 10) - lsym), lsym)
check(t, "境目 (0.6−λ)(0.5−λ)=0.4−λ の正の解 λ≈0.370", len(roots) == 1 and close(float(roots[0]), 0.370, 5e-4) and close(lam_b, float(roots[0]), 1e-12), r(lam_b))
Ob1, Ob2 = glasso_admm(Sg3, lam_b - 0.005), glasso_admm(Sg3, lam_b + 0.005)
check(t, "ADMM：λ=境目−0.005 では ω̂13≠0、境目+0.005 では ω̂13=0", abs(Ob1[0, 2]) > 1e-4 and abs(Ob2[0, 2]) < 1e-8, (r(Ob1[0, 2]), r(Ob2[0, 2])))
gnp = charts(t, "graph-network")[0]["params"]
check(t, "charts graph-network：辺 [0,1],[1,2]、ラベル 0.199, 0.098（λ=0.4 の偏相関）", gnp["nodes"] == ["X1", "X2", "X3"] and gnp["edges"] == [[0, 1], [1, 2]]
      and allclose([float(v) for v in gnp["edgeLabels"]], pc3(O4)[:2], 5e-4) and gnp["directed"] == 0, gnp["edgeLabels"])
check(t, "よくある間違い：Σ̂ の 1-3 成分は 0 でない（0.02）、残った偏相関は標本値より小さい", W4[0, 2] > 0.01 and all(a < b for a, b in zip(pc3(O4)[:2], pc3(np.linalg.inv(Sg3))[:2])), "成立")

# ================================================================ 集計
okn = sum(results)
ngn = len(results) - okn
print(f"\n合計: OK {okn} / NG {ngn}")
