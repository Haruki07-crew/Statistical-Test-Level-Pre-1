"""第22章 検算スクリプト（主成分分析）。
2×2 の S=[[5,2],[2,2]] と 5人×2変数の例、8人×4教科の例を numpy で再計算し、本文の数値と照合する。
固有ベクトルの符号は任意なので、比較の前に本文の向き（第1成分または和が正など）にそろえる。
ラグランジュ乗数法の導出・角度表示・2×2 の固有値公式は sympy で確認。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など。各チェックに明記）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
乱数シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from pathlib import Path
import math
import numpy as np
import sympy as sp
import yaml

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
    if claims.size == 1:
        claims = np.full(xs.shape, claims[0])
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-12))


def front(term):
    txt = (ROOT / "content" / "ch22" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term):
    return front(term).get("charts", [])


def chart(term, cid, idx=0):
    return [c for c in charts(term) if c["id"] == cid][idx]


def eig_desc(M):
    w, V = np.linalg.eigh(M)
    return w[::-1], V[:, ::-1].copy()


def align(V, ref):
    """各列の符号を参照ベクトル（列ごと）との内積が正になるようにそろえる"""
    V = V.copy()
    for k in range(V.shape[1]):
        if V[:, k] @ ref[:, k] < 0:
            V[:, k] *= -1
    return V


def r(x):
    return np.round(np.asarray(x, float), 4).tolist()


# ---------------------------------------------------------------- データ
S2 = np.array([[5.0, 2.0], [2.0, 2.0]])
x1 = np.array([7, 9, 10, 11, 13], float)
x2 = np.array([3, 5, 5, 7, 5], float)
X5 = np.column_stack([x1, x2])
X = np.array([[72, 68, 60, 58], [85, 80, 70, 72], [60, 66, 78, 80], [90, 88, 92, 90],
              [55, 58, 50, 52], [78, 70, 85, 82], [66, 72, 64, 60], [82, 86, 74, 78]], float)
n, p = X.shape
mean = X.mean(0)
sd = X.std(0, ddof=1)
S = np.cov(X.T)
R = np.corrcoef(X.T)
lamS, HS = eig_desc(S)
lamR, HR = eig_desc(R)
# 本文の向き：第1主成分は全部正、第2主成分は国語が正
ref = np.ones((4, 4)); ref[:, 1] = [1, 1, -1, -1]; ref[:, 2] = [1, -1, 0, 0]; ref[:, 3] = [0, 0, -1, 1]
HS = align(HS, ref)
HR = align(HR, ref)
Xc = X - mean
U = Xc / sd
ZR = U @ HR
ZS = Xc @ HS

# ================================================================ eigenvalue-and-eigenvector
T = "eigenvalue-and-eigenvector"
a, b, c, lam = sp.symbols("a b c lambda", real=True)
A = sp.Matrix([[a, b], [b, c]])
cp = sp.expand((A - lam * sp.eye(2)).det())
check(T, "2×2 対称行列の固有方程式 = λ²−(a+c)λ+(ac−b²)", sp.simplify(cp - (lam**2 - (a + c) * lam + a * c - b**2)) == 0, cp)
sols = sp.solve(cp, lam)
formula = [((a + c) + s * sp.sqrt((a - c) ** 2 + 4 * b**2)) / 2 for s in (1, -1)]
check(T, "解の公式 ((a+c)±√((a−c)²+4b²))/2", all(any(sp.simplify(f - s_) == 0 for s_ in sols) for f in formula), formula)
check(T, "判別式 (a+c)²−4(ac−b²) = (a−c)²+4b²", sp.expand((a + c) ** 2 - 4 * (a * c - b**2) - ((a - c) ** 2 + 4 * b**2)) == 0, "恒等式")
hvec = sp.Matrix([b, lam - a])
ok = all(sp.simplify(((A - s_ * sp.eye(2)) * hvec.subs(lam, s_))[i]) == 0 for s_ in sols for i in range(2))
check(T, "固有ベクトル h ∝ (b, λ−a)", ok, "(A−λI)(b,λ−a)'=0")
rr = sp.symbols("r", real=True)
Rr = sp.Matrix([[1, rr], [rr, 1]])
ev = Rr.eigenvects()
vals = sorted([e[0] for e in ev], key=str)
okR = set(sp.simplify(v) for v in vals) == {1 + rr, 1 - rr}
vecs_ok = (Rr * sp.Matrix([1, 1]) - (1 + rr) * sp.Matrix([1, 1])).is_zero_matrix and (Rr * sp.Matrix([1, -1]) - (1 - rr) * sp.Matrix([1, -1])).is_zero_matrix
check(T, "相関行列 [[1,r],[r,1]] の固有値 1±r、固有ベクトル (1,1)/√2,(1,−1)/√2", okR and vecs_ok, vals)
w2, V2 = eig_desc(S2)
check(T, "例1 固有方程式 λ²−7λ+6", sp.expand((sp.Matrix(S2.astype(int)) - lam * sp.eye(2)).det()) == lam**2 - 7 * lam + 6, "λ²−7λ+6")
check(T, "例1 固有値 6, 1", allclose(w2, [6, 1], 1e-12), r(w2))
check(T, "例1 和 7=tr S、積 6=det S", close(w2.sum(), 7, 1e-12) and close(np.prod(w2), 6, 1e-12) and close(np.linalg.det(S2), 6, 1e-12), (w2.sum(), np.prod(w2)))
V2a = align(V2, np.array([[2, 1], [1, -2]], float))
check(T, "例1 h1=(2,1)/√5≈(0.894,0.447)", allclose(V2a[:, 0], np.array([2, 1]) / math.sqrt(5), 1e-12) and allclose(V2a[:, 0], [0.894, 0.447], 5e-4), r(V2a[:, 0]))
check(T, "例1 h2=(1,−2)/√5≈(0.447,−0.894)", allclose(V2a[:, 1], np.array([1, -2]) / math.sqrt(5), 1e-12) and allclose(V2a[:, 1], [0.447, -0.894], 5e-4), r(V2a[:, 1]))
check(T, "例1 内積 0", close(V2a[:, 0] @ V2a[:, 1], 0, 1e-12), V2a[:, 0] @ V2a[:, 1])
h1 = np.array([[2], [1]]) / math.sqrt(5); h2 = np.array([[1], [-2]]) / math.sqrt(5)
check(T, "例1 スペクトル分解 6·h1h1'+1·h2h2' = S（中間の (1/5)[[25,10],[10,10]] も）",
      allclose(6 * h1 @ h1.T + h2 @ h2.T, S2, 1e-12) and allclose(np.array([[25, 10], [10, 10]]) / 5, S2, 1e-12)
      and allclose(5 * h1 @ h1.T, [[4, 2], [2, 1]], 1e-12) and allclose(5 * h2 @ h2.T, [[1, -2], [-2, 4]], 1e-12), "一致")
r12 = 2 / math.sqrt(10)
check(T, "例2 r=2/√10≈0.6325", close(r12, 0.6325), r12)
wR2, VR2 = eig_desc(np.array([[1, r12], [r12, 1]]))
check(T, "例2 相関行列の固有値 1.6325, 0.3675", allclose(wR2, [1.6325, 0.3675]), r(wR2))
check(T, "例2 固有ベクトル 45度", allclose(np.abs(VR2), 1 / math.sqrt(2), 1e-12), r(VR2))
check(T, "例2 共分散の向き 約26.57度", close(math.degrees(math.atan2(1, 2)), 26.57, 5e-3), math.degrees(math.atan(0.5)))
check(T, "例2 第1固有値の割合 6/7≈0.857、1.6325/2≈0.816", close(6 / 7, 0.857, 5e-4) and close(wR2[0] / 2, 0.816, 5e-4), (6 / 7, wR2[0] / 2))
check(T, "例3 4教科 R の固有値 3.2501, 0.6287, 0.1012, 0.0200", allclose(lamR, [3.2501, 0.6287, 0.1012, 0.0200]), r(lamR))
check(T, "例3 和 4.0000 = tr R", close(lamR.sum(), 4, 1e-12) and close(np.trace(R), 4, 1e-12), lamR.sum())
check(T, "例3 det R≈0.0041（= 固有値の積）", close(np.linalg.det(R), 0.0041, 5e-5) and close(np.prod(lamR), np.linalg.det(R), 1e-12), np.linalg.det(R))
w_raw = np.linalg.eigh(R)[0]
check(T, "例3 numpy.linalg.eigh は昇順で返す", bool(np.all(np.diff(w_raw) > 0)), r(w_raw))
# レイリー商の最大値
dirs = rng.normal(size=(200000, 2)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
q = np.einsum("ij,jk,ik->i", dirs, S2, dirs)
check(T, "a'Aa ≤ λ1（単位ベクトル20万本、乱数）、等号は ±h1", q.max() <= 6 + 1e-12 and q.min() >= 1 - 1e-12 and close((h1.T @ S2 @ h1).item(), 6, 1e-12), (q.max(), q.min()))
# 対称行列の性質（乱数の非負定値行列）
B = rng.normal(size=(6, 5)); M = B.T @ B
wM, VM = np.linalg.eigh(M)
check(T, "対称非負定値行列：固有値実数・非負、H'H=I、tr=Σλ、det=Πλ（乱数行列）",
      bool(np.all(wM >= -1e-12)) and allclose(VM.T @ VM, np.eye(5), 1e-10) and close(np.trace(M), wM.sum(), 1e-9)
      and abs(np.linalg.det(M) - np.prod(wM)) <= 1e-8 * abs(np.linalg.det(M)) and allclose(VM @ np.diag(wM) @ VM.T, M, 1e-9), r(wM))
cc = chart(T, "pca-axes")
check(T, "chart pca-axes params rho=0.5, sx=sy=1 がスライダー範囲内", cc["params"]["rho"] == 0.5 and cc["params"]["sx"] == 1 and cc["params"]["sy"] == 1
      and all(s["min"] <= cc["params"][s["param"]] <= s["max"] for s in cc["sliders"]), cc["params"])
# 図のキャプション：sx=sy=1, rho=0.5, n=200 の標本で固有値 ≈ 1±ρ（乱数、500回の平均）
sims = []
for _ in range(500):
    Z2 = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=200)
    wv, VV = eig_desc(np.cov(Z2.T))
    ang = math.degrees(math.atan2(abs(VV[1, 0]), abs(VV[0, 0])))
    sims.append([*wv, ang])
sims = np.array(sims)
check(T, "chart caption：ρ=0.5,n=200 で標本固有値の平均≈1.5,0.5（許容 0.03）、主軸≈45度（平均、許容 1度）",
      close(sims[:, 0].mean(), 1.5, 0.03) and close(sims[:, 1].mean(), 0.5, 0.03) and close(sims[:, 2].mean(), 45, 1.0), r(sims.mean(0)))

# ================================================================ principal-component-analysis
T = "principal-component-analysis"
check(T, "例1 5人のデータの標本分散共分散行列 = [[5,2],[2,2]]", allclose(np.cov(X5.T), S2, 1e-12), np.cov(X5.T).tolist())
check(T, "例1 固有値 6,1、h1≈(0.894,0.447)、6/7≈0.857", allclose(w2, [6, 1], 1e-12) and allclose(V2a[:, 0], [0.894, 0.447], 5e-4) and close(6 / 7, 0.857, 5e-4), r(w2))
check(T, "例2 データの平均 (73.5,73.5,71.625,71.5)", allclose(mean, [73.5, 73.5, 71.625, 71.5], 1e-12), mean.tolist())
Rclaim = [[1, .902, .663, .656], [.902, 1, .638, .664], [.663, .638, 1, .976], [.656, .664, .976, 1]]
check(T, "例2 相関行列 R（小数第3位）", allclose(R, Rclaim, 5e-4), np.round(R, 4).tolist())
check(T, "例2 固有値 3.250,0.629,0.101,0.020（合計4）", allclose(lamR, [3.250, 0.629, 0.101, 0.020], 5e-4) and close(lamR.sum(), 4, 1e-12), r(lamR))
check(T, "例2 h1≈(0.494,0.491,0.506,0.509)", allclose(HR[:, 0], [0.494, 0.491, 0.506, 0.509], 5e-4), r(HR[:, 0]))
check(T, "例2 h2≈(0.499,0.515,−0.501,−0.484)", allclose(HR[:, 1], [0.499, 0.515, -0.501, -0.484], 5e-4), r(HR[:, 1]))
check(T, "例2 寄与率 3.250/4≈0.8125、0.629/4≈0.1572（厳密値）", close(lamR[0] / 4, 0.8125) and close(lamR[1] / 4, 0.1572), r(lamR / 4))
check(T, "例2 累積寄与率 0.9697", close(lamR[:2].sum() / 4, 0.9697), lamR[:2].sum() / 4)
check(T, "Σλk = tr Σ = ΣV[Xj]（4教科の共分散行列）", close(lamS.sum(), np.trace(S), 1e-9) and close(lamS.sum(), sd @ sd, 1e-9), lamS.sum())
check(T, "V[Yk]=λk、Cov[Yj,Yk]=0（4教科、相関行列）", allclose(np.cov(ZR.T), np.diag(lamR), 1e-10), r(np.diag(np.cov(ZR.T))))
cc = chart(T, "pca-axes")
check(T, "chart pca-axes rho=0.8, sx=sy=1、スライダー範囲内", cc["params"]["rho"] == 0.8 and all(s["min"] <= cc["params"][s["param"]] <= s["max"] for s in cc["sliders"]), cc["params"])
cb = chart(T, "bar-data")
cum = np.cumsum(cb["params"]["values"]) / 4
check(T, "chart bar-data 値 = R の固有値（小数第4位）", allclose(cb["params"]["values"], lamR), cb["params"]["values"])
check(T, "chart caption 第1で約81%、第2までで約97%", close(cum[0], 0.81, 5e-3) and close(cum[1], 0.97, 5e-3), r(cum))
# 大きい |ρ| で第1寄与率→1（母集団）
check(T, "グラフ説明：2変数相関行列の第1寄与率 (1+|ρ|)/2 は |ρ|→1 で 1、ρ=0 で 0.5", close((1 + 0.95) / 2, 0.975, 1e-12) and close(eig_desc(np.eye(2))[0][0] / 2, 0.5, 1e-12), "0.975 / 0.5")

# ================================================================ principal-component
T = "principal-component"
Y1 = (2 * (x1 - 10) + (x2 - 5)) / math.sqrt(5)
Y2 = ((x1 - 10) - 2 * (x2 - 5)) / math.sqrt(5)
check(T, "例1 平均 (10,5)", allclose(X5.mean(0), [10, 5], 1e-12), X5.mean(0).tolist())
check(T, "例1 第1主成分 (−3.578,−0.894,0,1.789,2.683)", allclose(Y1, [-3.578, -0.894, 0, 1.789, 2.683], 5e-4), r(Y1))
check(T, "例1 Y1² = (12.8,0.8,0,3.2,7.2)、分散 6", allclose(Y1**2, [12.8, 0.8, 0, 3.2, 7.2], 1e-12) and close(Y1.var(ddof=1), 6, 1e-12), r(Y1**2))
check(T, "例1 第2主成分 (0.447,−0.447,0,−1.342,1.342)、分散 1", allclose(Y2, [0.447, -0.447, 0, -1.342, 1.342], 5e-4) and close(Y2.var(ddof=1), 1, 1e-12), r(Y2))
check(T, "例1 共分散 0", close(np.cov(Y1, Y2)[0, 1], 0, 1e-12), np.cov(Y1, Y2)[0, 1])
check(T, "H'SH = diag(λ)（2×2）", allclose(V2a.T @ S2 @ V2a, np.diag([6, 1]), 1e-12), r(V2a.T @ S2 @ V2a))
check(T, "例2 係数・分散（R の h1,h2, λ=3.250,0.629）", allclose(HR[:, 0], [0.494, 0.491, 0.506, 0.509], 5e-4) and allclose(HR[:, 1], [0.499, 0.515, -0.501, -0.484], 5e-4)
      and allclose(lamR[:2], [3.250, 0.629], 5e-4), r(HR[:, :2]))
a1 = np.array([1, 0.0]); a45 = np.array([1, 1]) / math.sqrt(2)
check(T, "例3 a=(1,0) で 5、45度で (5+4+2)/2=5.5", close(a1 @ S2 @ a1, 5, 1e-12) and close(a45 @ S2 @ a45, 5.5, 1e-12), (a1 @ S2 @ a1, a45 @ S2 @ a45))
check(T, "例3 第1主成分方向 約26.57度で 6", close(math.degrees(math.atan(0.5)), 26.57, 5e-3) and close((h1.T @ S2 @ h1).item(), 6, 1e-12), math.degrees(math.atan(0.5)))
# 最小残差の見方：第1主成分の直線への垂線2乗和 = (n−1)λ2 が最小、回帰直線と傾きが違う
Xc5 = X5 - X5.mean(0)
ths = np.linspace(0, np.pi, 18001)
res = [((Xc5 @ np.array([-math.sin(t), math.cos(t)])) ** 2).sum() for t in ths]
tbest = ths[int(np.argmin(res))]
check(T, "直交距離2乗和の最小方向 = 第1主成分（26.57度）、最小値 = 4·λ2 = 4", close(math.degrees(tbest), 26.57, 0.01) and close(min(res), 4, 1e-6), (math.degrees(tbest), min(res)))
b_reg = np.cov(x1, x2)[0, 1] / x1.var(ddof=1)
check(T, "回帰直線（x2 on x1）の傾き 0.4 ≠ 第1主成分の傾き 0.5", close(b_reg, 0.4, 1e-12) and not close(b_reg, 0.5, 1e-3), b_reg)
check(T, "主成分の個数 ≤ min(p, n−1)：n=3, p=4 の乱数データで非零固有値は2個",
      int(np.sum(eig_desc(np.cov(rng.normal(size=(3, 4)).T))[0] > 1e-10)) == 2, "2")
# 負荷量の公式（よくある間違い）を4教科共分散行列で
corr_direct = [np.corrcoef(ZS[:, 0], X[:, j])[0, 1] for j in range(4)]
check(T, "よくある間違い：相関 = √λk h_jk/√σjj（4教科、共分散行列、第1主成分）", allclose(corr_direct, math.sqrt(lamS[0]) * HS[:, 0] / sd, 1e-10), r(corr_direct))
cc = chart(T, "pca-axes")
check(T, "chart pca-axes rho=0.6, sx=1.5, sy=1、スライダー範囲内", cc["params"]["rho"] == 0.6 and cc["params"]["sx"] == 1.5 and cc["params"]["sy"] == 1
      and all(s["min"] <= cc["params"][s["param"]] <= s["max"] for s in cc["sliders"]), cc["params"])

# ================================================================ lagrange-multiplier-pca
T = "lagrange-multiplier-pca"
a1s, a2s, l, mu = sp.symbols("a1 a2 lambda mu", real=True)
s11, s12, s22 = sp.symbols("s11 s12 s22", real=True)
Sig = sp.Matrix([[s11, s12], [s12, s22]])
av = sp.Matrix([a1s, a2s])
L = (av.T * Sig * av)[0] - l * ((av.T * av)[0] - 1)
grad = sp.Matrix([sp.diff(L, v) for v in (a1s, a2s)])
check(T, "∂L/∂a = 2Σa − 2λa（sympy、一般の 2×2 対称 Σ）", sp.simplify(grad - (2 * Sig * av - 2 * l * av)).is_zero_matrix, list(grad))
quad = (av.T * Sig * av)[0]
gq = sp.Matrix([sp.diff(quad, v) for v in (a1s, a2s)])
check(T, "∂(a'Aa)/∂a = 2Aa（A 対称）", sp.simplify(gq - 2 * Sig * av).is_zero_matrix, list(gq))
Ans = sp.Matrix([[s11, sp.Symbol("t")], [s12, s22]])
gn = sp.Matrix([sp.diff((av.T * Ans * av)[0], v) for v in (a1s, a2s)])
check(T, "非対称なら ∂(a'Aa)/∂a = (A+A')a ≠ 2Aa（対称のときの公式であること）", sp.simplify(gn - (Ans + Ans.T) * av).is_zero_matrix and not sp.simplify(gn - 2 * Ans * av).is_zero_matrix, "確認")
# 3次元の例で第2主成分の μ=0
S3 = np.array([[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]])
l3, H3 = eig_desc(S3)
h1n = H3[:, 0]; a2n = H3[:, 1]
mu_val = h1n @ (2 * S3 @ a2n - 2 * l3[1] * a2n)
check(T, "第2主成分：h1'(2Σa−2λa)=μ h1'h1 から μ=0（3×3 の数値例）", close(mu_val, 0, 1e-12), mu_val)
# 制約付き最大化を数値で：h1 と直交な単位ベクトルの最大分散 = λ2
Q = rng.normal(size=(200000, 3)); Q -= np.outer(Q @ h1n, h1n); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
qq = np.einsum("ij,jk,ik->i", Q, S3, Q)
check(T, "h1 に直交な単位ベクトルの a'Σa の最大 ≤ λ2 かつ近い（乱数20万本、許容 1e-3）", qq.max() <= l3[1] + 1e-12 and close(qq.max(), l3[1], 1e-3), (qq.max(), l3[1]))
# 例1 連立方程式
sol = sp.solve([5 * a1s + 2 * a2s - l * a1s, 2 * a1s + 2 * a2s - l * a2s, a1s**2 + a2s**2 - 1], [a1s, a2s, l], dict=True)
lams = sorted(set(s_[l] for s_ in sol))
sol6 = [s_ for s_ in sol if s_[l] == 6]
check(T, "例1 λ=1,6、λ=6 で a=±(2,1)/√5", lams == [1, 6] and all(sp.simplify(s_[a1s] - 2 * s_[a2s]) == 0 and sp.simplify(s_[a2s] ** 2 - sp.Rational(1, 5)) == 0 for s_ in sol6), sol6)
th = sp.symbols("theta", real=True)
ex = 5 * sp.cos(th) ** 2 + 4 * sp.sin(th) * sp.cos(th) + 2 * sp.sin(th) ** 2
check(T, "例2 a'Σa = 5cos²θ+4sinθcosθ+2sin²θ", sp.simplify((sp.Matrix([sp.cos(th), sp.sin(th)]).T * sp.Matrix([[5, 2], [2, 2]]) * sp.Matrix([sp.cos(th), sp.sin(th)]))[0] - ex) == 0, ex)
check(T, "例2 = 3.5+1.5cos2θ+2sin2θ", sp.simplify(sp.expand_trig(ex - (sp.Rational(7, 2) + sp.Rational(3, 2) * sp.cos(2 * th) + 2 * sp.sin(2 * th)))) == 0, "恒等式")
check(T, "例2 振幅 √(1.5²+2²)=2.5、最大 6・最小 1", close(math.hypot(1.5, 2), 2.5, 1e-12), math.hypot(1.5, 2))
tmax = math.atan2(2, 1.5) / 2
check(T, "例2 tan2θ=2/1.5、θ≈26.57度、tanθ=1/2", close(math.degrees(tmax), 26.57, 5e-3) and close(math.tan(tmax), 0.5, 1e-12), math.degrees(tmax))
check(T, "グラフ説明：最小は約116.57度で分散1（最大と90度差）", close(math.degrees(tmax) + 90, 116.57, 5e-3) and close(float(sp.N(ex.subs(th, tmax + math.pi / 2))), 1, 1e-12), math.degrees(tmax) + 90)
cl = chart(T, "line-data")
xs_ = np.array(cl["params"]["x"], float); ys_ = np.array(cl["params"]["series"][0]["y"], float)
yt = 3.5 + 1.5 * np.cos(2 * np.radians(xs_)) + 2 * np.sin(2 * np.radians(xs_))
check(T, "chart line-data：x は 0〜180 を 7.5 刻み（25点）", allclose(xs_, np.arange(0, 180.01, 7.5), 1e-12), len(xs_))
check(T, "chart line-data：y = 3.5+1.5cos2θ+2sin2θ（小数第4位）", allclose(ys_, yt), np.max(np.abs(ys_ - yt)))
check(T, "chart caption：最大値6（26.57度）、最小値1（116.57度）", close(yt.max(), 6, 0.05) and close(yt.min(), 1, 0.05), (yt.max(), yt.min()))

# ================================================================ contribution-ratio
T = "contribution-ratio"
check(T, "例1 c1=6/7≈0.8571、c2=1/7≈0.1429", close(6 / 7, 0.8571) and close(1 / 7, 0.1429), (6 / 7, 1 / 7))
check(T, "tr Σ = tr(HΛH') = Σλ（2×2）", close(np.trace(S2), w2.sum(), 1e-12), np.trace(S2))
check(T, "例2 S の固有値 516.42, 94.90, 13.07, 3.58、総分散 627.98", allclose(lamS, [516.42, 94.90, 13.07, 3.58], 5e-3) and close(lamS.sum(), 627.98, 5e-3), r(lamS))
check(T, "例2 S の寄与率 0.8224, 0.1511, 0.0208, 0.0057", allclose(lamS / lamS.sum(), [0.8224, 0.1511, 0.0208, 0.0057]), r(lamS / lamS.sum()))
check(T, "例2 R の固有値・寄与率 0.8125, 0.1572, 0.0253, 0.0050", allclose(lamR, [3.2501, 0.6287, 0.1012, 0.0200]) and allclose(lamR / 4, [0.8125, 0.1572, 0.0253, 0.0050]), r(lamR / 4))
check(T, "例2 各教科の分散 154.29, 107.14, 186.27, 180.29、和 = 627.98", allclose(sd**2, [154.29, 107.14, 186.27, 180.29], 5e-3) and close((sd**2).sum(), lamS.sum(), 1e-9), r(sd**2))
check(T, "例3 2×2 相関行列の第1寄与率 (1+|r|)/2：r=0.6 で 0.8、r=0 で 0.5", close(eig_desc(np.array([[1, .6], [.6, 1]]))[0][0] / 2, 0.8, 1e-12)
      and close(eig_desc(np.array([[1, -.6], [-.6, 1]]))[0][0] / 2, 0.8, 1e-12), "0.8 / 0.5")
cb = chart(T, "bar-data")
check(T, "chart bar-data = 共分散行列の寄与率、和 1", allclose(cb["params"]["values"], lamS / lamS.sum()) and close(sum(cb["params"]["values"]), 1, 1e-12), cb["params"]["values"])
cc = chart(T, "pca-axes")
check(T, "chart pca-axes rho=0.8、スライダー範囲内", cc["params"]["rho"] == 0.8 and all(s["min"] <= cc["params"][s["param"]] <= s["max"] for s in cc["sliders"]), cc["params"])
sims = []
for rho in (0.8, 0.0):
    vv = []
    for _ in range(500):
        Z2 = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=200)
        wv = eig_desc(np.cov(Z2.T))[0]
        vv.append(wv[0] / wv.sum())
    sims.append(np.mean(vv))
check(T, "chart caption：n=200 の標本で第1寄与率 ρ=0.8 →≈0.9（許容 0.01）、ρ=0 →0.5 より少し上（0.5〜0.56）", close(sims[0], 0.9, 0.01) and 0.5 < sims[1] < 0.56, r(sims))

# ================================================================ cumulative-contribution-ratio
T = "cumulative-contribution-ratio"
ev5 = np.array([2.4, 1.2, 0.7, 0.4, 0.3])
check(T, "例1 和 5", close(ev5.sum(), 5, 1e-12), ev5.sum())
check(T, "例1 寄与率 0.48,0.24,0.14,0.08,0.06", allclose(ev5 / 5, [0.48, 0.24, 0.14, 0.08, 0.06], 1e-12), r(ev5 / 5))
cum5 = np.cumsum(ev5) / 5
check(T, "例1 累積 0.48,0.72,0.86,0.94,1.00", allclose(cum5, [0.48, 0.72, 0.86, 0.94, 1.0], 1e-12), r(cum5))
mk = lambda cstar: int(np.argmax(cum5 >= cstar - 1e-12)) + 1
check(T, "例1 70%→2個、80%→3個、90%→4個", (mk(0.7), mk(0.8), mk(0.9)) == (2, 3, 4), (mk(0.7), mk(0.8), mk(0.9)))
cR = np.cumsum(lamR) / 4
check(T, "例2 C1≈0.8125、C2≈0.9697、C3≈0.9950", allclose(cR[:3], [0.8125, 0.9697, 0.9950]), r(cR))
check(T, "例2 80%→1個、90%→2個", int(np.argmax(cR >= 0.8)) + 1 == 1 and int(np.argmax(cR >= 0.9)) + 1 == 2, "1, 2")
recon = []
for m in (1, 2, 3):
    P = HS[:, :m] @ HS[:, :m].T
    recon.append(((Xc - Xc @ P) ** 2).sum() / (Xc**2).sum())
cS = np.cumsum(lamS) / lamS.sum()
check(T, "例3 再構成誤差/全体 = 0.1776 = 1−C1（共分散行列）", close(recon[0], 0.1776) and close(recon[0], 1 - cS[0], 1e-12) and close(1 - 516.42 / 627.98, 0.1776), r(recon))
check(T, "例3 第2主成分まで 0.0265 = 1−C2", close(recon[1], 0.0265) and close(recon[1], 1 - cS[1], 1e-12), recon[1])
check(T, "捨てた第k主成分の得点の2乗和 = (n−1)λk", allclose((ZS**2).sum(0), (n - 1) * lamS, 1e-8), r((ZS**2).sum(0)))
# 同じ次元の線形近似のなかで最小（乱数の射影と比較）
worst = 1e9
for _ in range(2000):
    Qr, _ = np.linalg.qr(rng.normal(size=(4, 1)))
    worst = min(worst, ((Xc - Xc @ Qr @ Qr.T) ** 2).sum() / (Xc**2).sum())
check(T, "1次元の射影近似で主成分が誤差最小（乱数方向2000本より小さい）", recon[0] <= worst + 1e-12, (recon[0], worst))
cb = chart(T, "bar-data")
cumc = np.cumsum(cb["params"]["values"]) / sum(cb["params"]["values"])
check(T, "chart bar-data = 例1 の固有値、caption の 0.72・0.86", cb["params"]["values"] == [2.4, 1.2, 0.7, 0.4, 0.3] and allclose(cumc[1:3], [0.72, 0.86], 1e-12), r(cumc))

# ================================================================ kaiser-criterion
T = "kaiser-criterion"
ev6 = np.array([3.1, 1.6, 0.45, 0.35, 0.28, 0.22])
check(T, "例1 和 6（相関行列として整合）", close(ev6.sum(), 6, 1e-12), ev6.sum())
check(T, "例1 固有値1以上は2個、累積寄与率 (3.1+1.6)/6≈0.783", int((ev6 >= 1).sum()) == 2 and close(4.7 / 6, 0.783, 5e-4), 4.7 / 6)
check(T, "例2 4教科：1以上は1個、累積寄与率 0.8125", int((lamR >= 1).sum()) == 1 and close(lamR[0] / 4, 0.8125), lamR[0] / 4)
check(T, "固有値の平均 = 1（相関行列）", close(lamR.mean(), 1, 1e-12), lamR.mean())
# 母集団は無相関でも標本固有値は 1 を超える（乱数）
cnt = []
for _ in range(2000):
    Zs = rng.normal(size=(50, 10))
    cnt.append(int((np.linalg.eigvalsh(np.corrcoef(Zs.T)) >= 1).sum()))
check(T, "無相関（p=10, n=50, 2000回）でも標本相関行列の固有値1以上の個数は平均 ≥ 3（基準が多めに見積もる）", np.mean(cnt) >= 3, np.mean(cnt))
cl = chart(T, "line-data")
s0 = cl["params"]["series"]
check(T, "chart line-data：固有値の系列 = 例1、基準線 = 1、x=1..6", s0[0]["y"] == ev6.tolist() and s0[1]["y"] == [1] * 6 and cl["params"]["x"] == [1, 2, 3, 4, 5, 6], s0[0]["y"])

# ================================================================ pca-on-correlation-matrix
T = "pca-on-correlation-matrix"
D = np.diag(sd)
check(T, "D⁻¹SD⁻¹ = R（4教科）", allclose(np.linalg.inv(D) @ S @ np.linalg.inv(D), R, 1e-12), "一致")
check(T, "R の固有ベクトルは D h（正規化後）とは一致しない", not allclose(align(((D @ HS) / np.linalg.norm(D @ HS, axis=0)), HR), HR, 1e-3), "不一致")
check(T, "例1 S の固有値 6,1、方向 (0.894,0.447)、寄与率 0.857", allclose(w2, [6, 1], 1e-12) and allclose(V2a[:, 0], [0.894, 0.447], 5e-4) and close(6 / 7, 0.857, 5e-4), r(V2a[:, 0]))
check(T, "例1 r≈0.6325、R の固有値 1.6325, 0.3675、方向 (0.707,0.707)、寄与率 0.816", close(r12, 0.6325) and allclose(wR2, [1.6325, 0.3675]) and allclose(np.abs(VR2[:, 0]), [0.707, 0.707], 5e-4) and close(wR2[0] / 2, 0.816, 5e-4), r(wR2))
X10 = X.copy(); X10[:, 0] *= 10
lam10, H10 = eig_desc(np.cov(X10.T)); H10 = align(H10, ref)
lamR10, HR10 = eig_desc(np.corrcoef(X10.T)); HR10 = align(HR10, ref)
check(T, "例2 共分散（元の点数）の h1 (0.468,0.383,0.567,0.559)、寄与率 0.8224", allclose(HS[:, 0], [0.468, 0.383, 0.567, 0.559], 5e-4) and close(lamS[0] / lamS.sum(), 0.8224), r(HS[:, 0]))
check(T, "例2 国語10倍の h1 (0.992,0.075,0.073,0.071)、寄与率 0.9859", allclose(H10[:, 0], [0.992, 0.075, 0.073, 0.071], 5e-4) and close(lam10[0] / lam10.sum(), 0.9859), (r(H10[:, 0]), lam10[0] / lam10.sum()))
check(T, "例2 相関行列（どちらでも）(0.494,0.491,0.506,0.509)、0.8125、換算前後で同一", allclose(HR[:, 0], [0.494, 0.491, 0.506, 0.509], 5e-4) and close(lamR[0] / 4, 0.8125)
      and allclose(lamR10, lamR, 1e-12) and allclose(HR10, HR, 1e-12), r(HR10[:, 0]))
check(T, "例2 国語の分散 154.29 → 15428.57、寄与率 98.6%", close(sd[0] ** 2, 154.29, 5e-3) and close(np.cov(X10.T)[0, 0], 15428.57, 5e-3) and close(lam10[0] / lam10.sum(), 0.986, 5e-4), np.cov(X10.T)[0, 0])
check(T, "表：相関行列の負荷量 √λ h、共分散行列の負荷量 √λ h/√σjj が相関係数に一致", allclose([np.corrcoef(ZR[:, 0], X[:, j])[0, 1] for j in range(4)], math.sqrt(lamR[0]) * HR[:, 0], 1e-10), "一致")
cc = chart(T, "pca-axes")
check(T, "chart pca-axes rho=0.6, sx=sy=1、スライダー範囲内（sx 0.3〜3）", cc["params"]["rho"] == 0.6 and cc["params"]["sx"] == 1 and all(s["min"] <= cc["params"][s["param"]] <= s["max"] for s in cc["sliders"]), cc["params"])
# 母集団 sx を大きくすると第1軸が x1 軸へ倒れる
angs = []
for sx in (1, 2, 3):
    Sg = np.array([[sx**2, 0.6 * sx], [0.6 * sx, 1]])
    vv = eig_desc(Sg)[1][:, 0]
    angs.append(math.degrees(math.atan2(abs(vv[1]), abs(vv[0]))))
check(T, "グラフ説明：ρ=0.6 で sx=1,2,3 のとき第1軸の角度が 45度から単調減少", close(angs[0], 45, 1e-9) and angs[0] > angs[1] > angs[2], r(angs))

# ================================================================ principal-component-loading
T = "principal-component-loading"
LS2 = np.sqrt(w2) * V2a / np.sqrt(np.diag(S2))[:, None]
check(T, "例1 r(Y1,X1)=2√6/5≈0.9798、r(Y1,X2)=√0.6≈0.7746", close(LS2[0, 0], 2 * math.sqrt(6) / 5, 1e-12) and close(LS2[0, 0], 0.9798) and close(LS2[1, 0], math.sqrt(0.6), 1e-12) and close(LS2[1, 0], 0.7746), r(LS2[:, 0]))
check(T, "例1 r(Y2,X1)=0.2、r(Y2,X2)≈−0.6325", close(LS2[0, 1], 0.2, 1e-12) and close(LS2[1, 1], -0.6325), r(LS2[:, 1]))
check(T, "例1 0.96+0.04=1、0.6+0.4=1", allclose(LS2**2, [[0.96, 0.04], [0.6, 0.4]], 1e-12), r(LS2**2))
check(T, "例1 公式 = 実データで計算した相関（5人）", allclose([np.corrcoef(Y1, x1)[0, 1], np.corrcoef(Y1, x2)[0, 1], np.corrcoef(Y2, x1)[0, 1], np.corrcoef(Y2, x2)[0, 1]], LS2.T.ravel(), 1e-12), "一致")
check(T, "Cov[X,Yk] = Σhk = λk hk（4教科、共分散行列）", allclose([np.cov(X[:, j], ZS[:, 0])[0, 1] for j in range(4)], lamS[0] * HS[:, 0], 1e-9), "一致")
LR = np.sqrt(lamR) * HR
check(T, "例2 負荷量表 第1列 (0.891,0.886,0.911,0.917)", allclose(LR[:, 0], [0.891, 0.886, 0.911, 0.917], 5e-4), r(LR[:, 0]))
check(T, "例2 負荷量表 第2列 (0.395,0.409,−0.397,−0.384)", allclose(LR[:, 1], [0.395, 0.409, -0.397, -0.384], 5e-4), r(LR[:, 1]))
check(T, "例2 2乗和 (0.950,0.952,0.989,0.988)", allclose((LR[:, :2] ** 2).sum(1), [0.950, 0.952, 0.989, 0.988], 5e-4), r((LR[:, :2] ** 2).sum(1)))
check(T, "例2 国語 √3.2501×0.4943≈0.891、h11≈0.4943", close(math.sqrt(3.2501) * 0.4943, 0.891, 5e-4) and close(HR[0, 0], 0.4943), HR[0, 0])
check(T, "例2 第1列の2乗和 3.250 = λ1、行の2乗和（全主成分）= 1", close((LR[:, 0] ** 2).sum(), 3.250, 5e-4) and allclose((LR**2).sum(1), 1, 1e-12), (LR[:, 0] ** 2).sum())
check(T, "例2 各教科とも2主成分で95%以上", bool(np.all((LR[:, :2] ** 2).sum(1) >= 0.95)), r((LR[:, :2] ** 2).sum(1)))
check(T, "例3 λ1=516.42、h1≈(0.468,0.383,0.567,0.559)、SD 12.42,10.35,13.65,13.43", close(lamS[0], 516.42, 5e-3) and allclose(HS[:, 0], [0.468, 0.383, 0.567, 0.559], 5e-4) and allclose(sd, [12.42, 10.35, 13.65, 13.43], 5e-3), r(sd))
LS = math.sqrt(lamS[0]) * HS[:, 0] / sd
check(T, "例3 √516.42≈22.725、h11≈0.4677、sd1≈12.421、国語 0.856", close(math.sqrt(lamS[0]), 22.725, 5e-4) and close(HS[0, 0], 0.4677) and close(sd[0], 12.421, 5e-4) and close(LS[0], 0.856, 5e-4), r(LS))
check(T, "例3 英語 0.842、数学 0.944、理科 0.947", allclose(LS[1:], [0.842, 0.944, 0.947], 5e-4), r(LS))
check(T, "例3 得点と国語の相関を直接計算しても 0.856", close(np.corrcoef(ZS[:, 0], X[:, 0])[0, 1], 0.856, 5e-4), np.corrcoef(ZS[:, 0], X[:, 0])[0, 1])
cp2 = chart(T, "points-2d")
pts = cp2["params"]["points"]
check(T, "chart points-2d：4教科の (x,y) = 第1・第2負荷量（小数第4位）", [q["label"] for q in pts] == ["国語", "英語", "数学", "理科"]
      and allclose([[q["x"], q["y"]] for q in pts], LR[:, :2]), [[q["x"], q["y"]] for q in pts])
check(T, "chart points-2d：全点が単位円内", all(q["x"] ** 2 + q["y"] ** 2 <= 1 for q in pts), [round(q["x"] ** 2 + q["y"] ** 2, 4) for q in pts])

# ================================================================ principal-component-score
T = "principal-component-score"
check(T, "Z'Z/(n−1) = Λ（共分散行列）", allclose(ZS.T @ ZS / (n - 1), np.diag(lamS), 1e-9), r(np.diag(ZS.T @ ZS / (n - 1))))
check(T, "得点の平均0・Σz_ik z_il=0、X_c = Z H'", allclose(ZS.mean(0), 0, 1e-10) and allclose(ZR.T @ ZR - np.diag(np.diag(ZR.T @ ZR)), 0, 1e-10) and allclose(ZS @ HS.T, Xc, 1e-10), "確認")
check(T, "例1 z11=−8/√5≈−3.578、5人の得点・合計0・分散6", close(Y1[0], -8 / math.sqrt(5), 1e-12) and close(Y1[0], -3.578, 5e-4) and close(Y1.sum(), 0, 1e-12) and close((Y1**2).sum() / 4, 6, 1e-12), r(Y1))
check(T, "例2 平均 (73.5,73.5,71.625,71.5)、SD (12.421,10.351,13.648,13.427)", allclose(mean, [73.5, 73.5, 71.625, 71.5], 1e-12) and allclose(sd, [12.421, 10.351, 13.648, 13.427], 5e-4), r(sd))
check(T, "例2 u4≈(1.328,1.401,1.493,1.378)", allclose(U[3], [1.328, 1.401, 1.493, 1.378], 5e-4), r(U[3]))
check(T, "例2 h1=(0.4943,0.4915,0.5055,0.5085)", allclose(HR[:, 0], [0.4943, 0.4915, 0.5055, 0.5085]), r(HR[:, 0]))
check(T, "例2 z41≈2.800（丸めた値の内積でも）", close(ZR[3, 0], 2.800, 5e-4) and close(np.dot([0.4943, 0.4915, 0.5055, 0.5085], [1.328, 1.401, 1.493, 1.378]), 2.800, 5e-4), ZR[3, 0])
check(T, "例2 第1得点の最大は4番、最小は5番 −3.012", int(np.argmax(ZR[:, 0])) == 3 and int(np.argmin(ZR[:, 0])) == 4 and close(ZR[4, 0], -3.012, 5e-4), r(ZR[:, 0]))
check(T, "例2 第2得点：最小3番 −1.456、最大2番 0.827", int(np.argmin(ZR[:, 1])) == 2 and int(np.argmax(ZR[:, 1])) == 1 and close(ZR[2, 1], -1.456, 5e-4) and close(ZR[1, 1], 0.827, 5e-4), r(ZR[:, 1]))
check(T, "例2 第1得点の分散 3.2501", close(ZR[:, 0].var(ddof=1), 3.2501), ZR[:, 0].var(ddof=1))
check(T, "例3 共分散 h1=(0.4677,0.3834,0.5669,0.5593)", allclose(HS[:, 0], [0.4677, 0.3834, 0.5669, 0.5593]), r(HS[:, 0]))
check(T, "例3 1番目のずれ (−1.5,−5.5,−11.625,−13.5)", allclose(Xc[0], [-1.5, -5.5, -11.625, -13.5], 1e-12), Xc[0].tolist())
check(T, "例3 z11≈−16.95（丸め係数でも）、分散 516.42", close(ZS[0, 0], -16.95, 5e-3) and close(np.dot([0.4677, 0.3834, 0.5669, 0.5593], Xc[0]), -16.95, 5e-3) and close(ZS[:, 0].var(ddof=1), 516.42, 5e-3), ZS[0, 0])
check(T, "例3 相関行列からの 1番目の得点 −1.263", close(ZR[0, 0], -1.263, 5e-4), ZR[0, 0])
cp2 = chart(T, "points-2d")
pts = cp2["params"]["points"]
check(T, "chart points-2d：8人の (x,y) = 相関行列の第1・第2得点（小数第4位）", [q["label"] for q in pts] == [str(i) for i in range(1, 9)]
      and allclose([[q["x"], q["y"]] for q in pts], ZR[:, :2]), np.round(ZR[:, :2], 5).tolist())

# ================================================================ scree-plot
T = "scree-plot"
d6 = -np.diff(ev6)
check(T, "例1 和 6、隣との差 1.5,1.15,0.10,0.07,0.06", close(ev6.sum(), 6, 1e-12) and allclose(d6, [1.5, 1.15, 0.10, 0.07, 0.06], 1e-12), r(d6))
check(T, "例1 累積寄与率 0.783、カイザー基準も2個", close(4.7 / 6, 0.783, 5e-4) and int((ev6 >= 1).sum()) == 2, 4.7 / 6)
check(T, "例2 固有値 3.250,0.629,0.101,0.020、第2→第3の下がり幅 0.528", allclose(lamR, [3.250, 0.629, 0.101, 0.020], 5e-4) and close(lamR[1] - lamR[2], 0.528, 5e-4), lamR[1] - lamR[2])
check(T, "縦軸を寄与率にしても形は定数倍", allclose((ev6 / 6) * 6, ev6, 1e-12), "定数倍")
cb = chart(T, "bar-data")
check(T, "chart bar-data = 例1 の固有値、cumulative あり", cb["params"]["values"] == ev6.tolist() and cb["params"].get("cumulative") is True, cb["params"]["values"])

# ================================================================ singular-value-decomposition
T = "singular-value-decomposition"
check(T, "例1 X_c と X_c'X_c=[[20,8],[8,8]]", allclose(Xc5, [[-3, -2], [-1, 0], [0, 0], [1, 2], [3, 0]], 1e-12) and allclose(Xc5.T @ Xc5, [[20, 8], [8, 8]], 1e-12), (Xc5.T @ Xc5).tolist())
check(T, "例1 固有方程式 λ²−28λ+96、固有値 24,4", sp.expand((sp.Matrix([[20, 8], [8, 8]]) - lam * sp.eye(2)).det()) == lam**2 - 28 * lam + 96 and allclose(eig_desc(Xc5.T @ Xc5)[0], [24, 4], 1e-12), "24, 4")
Uu, dd, Vt = np.linalg.svd(Xc5, full_matrices=False)
Vv = align(Vt.T, np.array([[2, 1], [1, -2]], float))
sgn = np.sign(np.diag(Vt.T.T @ Vv))
Uu = Uu * sgn
check(T, "例1 特異値 √24≈4.899、2；/4 で 6,1", allclose(dd, [math.sqrt(24), 2], 1e-12) and close(dd[0], 4.899, 5e-4) and allclose(dd**2 / 4, [6, 1], 1e-12), r(dd))
check(T, "例1 v1=(2,1)/√5、v2=(1,−2)/√5", allclose(Vv, np.array([[2, 1], [1, -2]]) / math.sqrt(5), 1e-12), r(Vv))
u1c = np.array([-8, -2, 0, 4, 6]) / math.sqrt(120)
check(T, "例1 u1=(−8,−2,0,4,6)/√120≈(−0.730,−0.183,0,0.365,0.548)", allclose(Uu[:, 0], u1c, 1e-12) and allclose(u1c, [-0.730, -0.183, 0, 0.365, 0.548], 5e-4), r(Uu[:, 0]))
check(T, "例1 d1 u1 = 主成分得点", allclose(dd[0] * Uu[:, 0], Y1, 1e-12), r(dd[0] * Uu[:, 0]))
check(T, "例1 第1項近似の誤差2乗和 = 4 = d2²", close(((Xc5 - dd[0] * np.outer(Uu[:, 0], Vv[:, 0])) ** 2).sum(), 4, 1e-12), ((Xc5 - dd[0] * np.outer(Uu[:, 0], Vv[:, 0])) ** 2).sum())
check(T, "X'X=VD²V'、XX'=UD²U'（例1）", allclose(Vv @ np.diag(dd**2) @ Vv.T, Xc5.T @ Xc5, 1e-10) and allclose(Uu @ np.diag(dd**2) @ Uu.T, Xc5 @ Xc5.T, 1e-10), "一致")
d4 = np.linalg.svd(Xc, compute_uv=False)
check(T, "例2 特異値 60.12, 25.77, 9.57, 5.01", allclose(d4, [60.12, 25.77, 9.57, 5.01], 5e-3), r(d4))
check(T, "例2 d²/7 = 516.42, 94.90, 13.07, 3.58 = S の固有値", allclose(d4**2 / 7, lamS, 1e-9) and allclose(d4**2 / 7, [516.42, 94.90, 13.07, 3.58], 5e-3), r(d4**2 / 7))
U4, _, Vt4 = np.linalg.svd(Xc, full_matrices=False)
V4 = align(Vt4.T, HS)
U4 = U4 * np.sign(np.diag(Vt4 @ V4))
check(T, "例2 h_k = v_k、Z = X_c V = U D（4教科）", allclose(V4, HS, 1e-10) and allclose(U4 * d4, ZS, 1e-9), "一致")
# エッカート・ヤング
Xm = (U4[:, :2] * d4[:2]) @ V4[:, :2].T
check(T, "エッカート・ヤング：階数2近似の誤差 = d3²+d4²、乱数の階数2行列より小さい", close(((Xc - Xm) ** 2).sum(), (d4[2:] ** 2).sum(), 1e-8)
      and all(((Xc - rng.normal(size=(8, 2)) @ rng.normal(size=(2, 4))) ** 2).sum() > ((Xc - Xm) ** 2).sum() for _ in range(2000)), (d4[2:] ** 2).sum())
cb = chart(T, "bar-data")
check(T, "chart bar-data = 特異値の2乗（小数第2位）", allclose(cb["params"]["values"], d4**2, 5e-3), (cb["params"]["values"], r(d4**2)))
cumd = np.cumsum(d4**2) / (d4**2).sum()
check(T, "chart caption：d²/7 = 固有値、累積割合 = 累積寄与率、第2項までで97%以上", allclose(cumd, cS, 1e-12) and cumd[1] >= 0.97, r(cumd))
# 中心化しないと第1特異ベクトルは平均方向
v_raw = np.linalg.svd(X, full_matrices=False)[2][0]
check(T, "よくある間違い：中心化しない第1右特異ベクトルは平均ベクトルの方向に近い（cos>0.99）、h1 とは不一致", abs(v_raw @ mean / np.linalg.norm(mean)) > 0.99 and not allclose(np.abs(v_raw), np.abs(HS[:, 0]), 1e-3), abs(v_raw @ mean / np.linalg.norm(mean)))

# ----------------------------------------------------------------
ok = sum(results)
ng = len(results) - ok
print(f"\n合計: OK {ok} / NG {ng}")
