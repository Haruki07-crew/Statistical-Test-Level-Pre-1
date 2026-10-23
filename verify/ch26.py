"""第26章 検算スクリプト（その他の多変量解析）。
古典的MDS（二重中心化・固有値・座標）、正準相関分析（固有値問題・q=1 で重相関・正準負荷量・
バートレット検定）、コレスポンデンス分析（標準化残差の SVD・イナーシャ=χ²/n・主座標・推移公式）、
数量化 I・II・III 類、SEM（丁度識別の解・最尤推定・χ²・df・RMSEA）、パス解析の効果分解、
ICA の尖度の例を numpy/scipy で独立に再計算し、本文と照合する。
固有ベクトル・座標の符号は任意なので、軸ごとに主張値と符号をそろえてから比較する
（行と列・x 側と y 側のように連動する組は、同じ符号反転だけを許す）。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など。各チェックに明記）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
乱数（seed=20261023）を使うチェックは各チェックに許容誤差を明記する。
"""
from pathlib import Path
import math
import numpy as np
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
    if claims.size == 1:
        claims = np.full(xs.shape, claims[0])
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-12))


def front(term):
    txt = (ROOT / "content" / "ch26" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def chart(term, cid, idx=0):
    return [c for c in front(term).get("charts", []) if c["id"] == cid][idx]


def r(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


def align(v, claim):
    """v の符号を claim に合わせる（内積が正になるように）。"""
    v = np.asarray(v, float)
    return v if float(v @ np.asarray(claim, float)) >= 0 else -v


def cmds(D):
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    lam, E = np.linalg.eigh(B)
    idx = np.argsort(lam)[::-1]
    return B, lam[idx], E[:, idx]


def exkurt(x):
    x = x - x.mean()
    return np.mean(x ** 4) / np.mean(x ** 2) ** 2 - 3


# ================================================================ 多次元尺度構成法
T = "multidimensional-scaling"
D1 = np.array([[0, 4, 5, 3], [4, 0, 3, 5], [5, 3, 0, 4], [3, 5, 4, 0]], float)
A1 = -D1 ** 2 / 2
check(T, "例1 a_ij の行平均・列平均・全体平均がすべて -6.25",
      allclose(A1.mean(1), [-6.25] * 4, 1e-12) and allclose(A1.mean(0), [-6.25] * 4, 1e-12)
      and close(A1.mean(), -6.25, 1e-12), r(A1.mean(1)))
B1, lam1, E1 = cmds(D1)
B1c = np.array([[6.25, -1.75, -6.25, 1.75], [-1.75, 6.25, 1.75, -6.25],
                [-6.25, 1.75, 6.25, -1.75], [1.75, -6.25, -1.75, 6.25]])
check(T, "例1 B 行列（=a_ij+6.25）", allclose(B1, B1c, 1e-12) and allclose(B1, A1 + 6.25, 1e-12), r(B1))
check(T, "例1 固有値 16, 9, 0, 0", allclose(lam1, [16, 9, 0, 0], 1e-9), r(lam1))
e1 = np.array([-1, 1, 1, -1]) / 2
e2 = np.array([1, 1, -1, -1]) / 2
check(T, "例1 Be1=16e1, Be2=9e2（長さ1）",
      allclose(B1 @ e1, 16 * e1, 1e-12) and allclose(B1 @ e2, 9 * e2, 1e-12)
      and close(e1 @ e1, 1, 1e-12) and close(e2 @ e2, 1, 1e-12), "")
X1 = np.column_stack([4 * e1, 3 * e2])
check(T, "例1 座標 A(-2,1.5) B(2,1.5) C(2,-1.5) D(-2,-1.5)",
      allclose(X1, [[-2, 1.5], [2, 1.5], [2, -1.5], [-2, -1.5]], 1e-12), X1.tolist())
check(T, "例1 復元座標の距離が D と完全一致", allclose(np.linalg.norm(X1[:, None] - X1[None], axis=2), D1, 1e-12), "")
check(T, "例1 4×3 の長方形（重心原点）", close(X1.mean(0)[0], 0, 1e-12) and close(X1.mean(0)[1], 0, 1e-12), "")

D2 = np.array([[0, 2, 5, 8], [2, 0, 3, 7], [5, 3, 0, 4], [8, 7, 4, 0]], float)
B2, lam2, E2 = cmds(D2)
check(T, "例2 固有値 38.853, 3.955, 0, -1.058（第3位 5e-4）", allclose(lam2, [38.853, 3.955, 0, -1.058], 5e-4), r(lam2))
check(T, "例2 一直線なら d_AD=9（本文は 8）", D2[0, 1] + D2[1, 2] + D2[2, 3] == 9, "")
check(T, "例2 1次元の当てはまり 0.908", close(lam2[0] / (lam2[0] + lam2[1]), 0.908, 5e-4), r(lam2[0] / (lam2[0] + lam2[1])))
x1 = align(np.sqrt(lam2[0]) * E2[:, 0], [3.396, 2.236, -1.018, -4.614])
check(T, "例2 第1次元座標 3.396, 2.236, -1.018, -4.614", allclose(x1, [3.396, 2.236, -1.018, -4.614], 5e-4), r(x1, 3))
X2 = E2[:, :2] * np.sqrt(lam2[:2])
dh = np.linalg.norm(X2[:, None] - X2[None], axis=2)
check(T, "例2 2次元の復元距離 d_AB≈2.254, d_AD≈8.020", close(dh[0, 1], 2.254, 5e-4) and close(dh[0, 3], 8.020, 5e-4),
      (r(dh[0, 1]), r(dh[0, 3])))
c = chart(T, "points-2d")
P = np.array([[p["x"], p["y"]] for p in c["params"]["points"]])
X2a = np.column_stack([align(X2[:, 0], P[:, 0]), align(X2[:, 1], P[:, 1])])
check(T, "charts points-2d の4駅の座標（軸ごとの符号を除き小数第4位）", allclose(X2a, P, 5e-5), r(X2a))
check(T, "charts ラベル A〜D駅", [p["label"] for p in c["params"]["points"]] == ["A駅", "B駅", "C駅", "D駅"], "")
check(T, "グラフ説明: 第2次元のずれは小さい（|y|<|x| の範囲 1.19 以下）", np.max(np.abs(P[:, 1])) < 1.2, r(np.max(np.abs(P[:, 1]))))

# 座標データのユークリッド距離 → 古典的MDS = 主成分得点（符号を除く）
Z = rng.normal(size=(12, 3)) @ np.array([[2, 0.5, 0], [0, 1, 0.3], [0, 0, 0.4]])
Dz = np.linalg.norm(Z[:, None] - Z[None], axis=2)
_, lz, Ez = cmds(Dz)
Zc = Z - Z.mean(0)
U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
pcs = Zc @ Vt.T
mds_sc = Ez[:, :3] * np.sqrt(lz[:3])
ok = all(np.allclose(np.abs(mds_sc[:, k]), np.abs(pcs[:, k]), atol=1e-9) for k in range(3))
check(T, "座標の距離→古典的MDS=主成分得点（符号を除く、乱数12点×3次元、許容1e-9）", ok and allclose(lz[3:], 0, 1e-9), r(lz[:4]))
# 回転・反転・平行移動で距離不変
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
Z2 = Z @ Q + rng.normal(size=3)
check(T, "回転・反転・平行移動で距離は不変（許容1e-9）",
      allclose(np.linalg.norm(Z2[:, None] - Z2[None], axis=2), Dz, 1e-9), "")

# ================================================================ 古典的MDS
T = "classical-mds"
Dl = np.zeros((4, 4))
dd = {(0, 1): 2, (0, 2): 5, (0, 3): 9, (1, 2): 3, (1, 3): 7, (2, 3): 4}
for (i, j), v in dd.items():
    Dl[i, j] = Dl[j, i] = v
check(T, "例1 A の第1行 (0,-2,-12.5,-40.5)", allclose(-Dl[0] ** 2 / 2, [0, -2, -12.5, -40.5], 1e-12), "")
Bl, ll, El = cmds(Dl)
xv = np.array([4, 2, -1, -5.])
check(T, "例1 B=xx^T（x=(4,2,-1,-5)）", allclose(Bl, np.outer(xv, xv), 1e-9), r(Bl))
check(T, "例1 固有値 46,0,0,0", allclose(ll, [46, 0, 0, 0], 1e-9), r(ll))
xc = align(np.sqrt(ll[0]) * El[:, 0], xv)
check(T, "例1 座標 ±(4,2,-1,-5)、差で道のりが戻る",
      allclose(xc, xv, 1e-9) and allclose(np.abs(xc[:, None] - xc[None]), Dl, 1e-9), r(xc))
check(T, "例1 トレース 46 = Σ_{i<j} d²/n = 184/4",
      close(np.trace(Bl), 46, 1e-9) and sum(v ** 2 for v in dd.values()) == 184, np.trace(Bl))
Dt = np.array([[0, 1, 2], [1, 0, 4], [2, 4, 0]], float)
Bt, lt, Et = cmds(Dt)
check(T, "例2 B 行列（第3位 5e-4）",
      allclose(Bt, [[-0.667, 0.833, -0.167], [0.833, 3.333, -4.167], [-0.167, -4.167, 4.333]], 5e-4), r(Bt, 4))
check(T, "例2 固有値 8.083, 0, -1.083", allclose(lt, [8.083, 0, -1.083], 5e-4), r(lt))
check(T, "例2 b11<0", Bt[0, 0] < 0, r(Bt[0, 0]))
xt = align(np.sqrt(lt[0]) * Et[:, 0], [-0.220, -1.891, 2.111])
check(T, "例2 1次元座標 (-0.220,-1.891,2.111)", allclose(xt, [-0.220, -1.891, 2.111], 5e-4), r(xt, 3))
check(T, "例2 当てはまり 1（正のみ）, 0.882（絶対値）",
      close(lt[0] / lt[lt > 1e-9].sum(), 1, 1e-12) and close(lt[0] / np.abs(lt).sum(), 0.882, 5e-4),
      r(lt[0] / np.abs(lt).sum()))
c = chart(T, "bar-data")
check(T, "charts bar-data の固有値 8.0826, 0, -1.0826", allclose(lt, c["params"]["values"], 5e-5), r(lt))
# J1=0, B は固有値0を持つ, 固有値 = 座標の2乗和
check(T, "J1=0 で B1=0（固有値0を必ず持つ）", allclose(Bt @ np.ones(3), 0, 1e-12) and allclose(B2 @ np.ones(4), 0, 1e-12), "")
check(T, "固有値=その次元の座標の2乗和（例2 MDS の2次元）",
      allclose((X2 ** 2).sum(0), lam2[:2], 1e-9), r((X2 ** 2).sum(0)))
# 二重中心化の成分式 = JAJ
At = -Dt ** 2 / 2
Bs = At - At.mean(1)[:, None] - At.mean(0)[None] + At.mean()
check(T, "b_ij=a_ij-ā_i.-ā_.j+ā.. が JAJ と一致", allclose(Bs, Bt, 1e-12), "")
# ユークリッド距離なら B は半正定値（乱数）
okpsd = True
for _ in range(50):
    W = rng.normal(size=(7, rng.integers(1, 5)))
    _, lw, _ = cmds(np.linalg.norm(W[:, None] - W[None], axis=2))
    okpsd &= lw.min() > -1e-9 and int((lw > 1e-9).sum()) == min(W.shape[1], 6)
check(T, "ユークリッド距離なら B は半正定値・正の固有値数=次元（乱数50組、許容1e-9）", okpsd, "")
# 導出の途中式
W = rng.normal(size=(6, 2))
W -= W.mean(0)
Aw = -0.5 * np.linalg.norm(W[:, None] - W[None], axis=2) ** 2
cvec = (W ** 2).sum(1)
check(T, "ā_i.=-(c_i+c̄)/2, ā..=-c̄, B=XX^T（重心原点、乱数）",
      allclose(Aw.mean(1), -(cvec + cvec.mean()) / 2, 1e-9) and close(Aw.mean(), -cvec.mean(), 1e-9)
      and allclose(Aw - Aw.mean(1)[:, None] - Aw.mean(0)[None] + Aw.mean(), W @ W.T, 1e-9), "")

# ================================================================ 正準相関分析
T = "canonical-correlation-analysis"
R11 = np.array([[1, 0.4], [0.4, 1]])
R22 = np.array([[1, 0.3], [0.3, 1]])
R12 = np.array([[0.5, 0.3], [0.4, 0.2]])
R21 = R12.T
check(T, "R11^-1, R22^-1 の形", allclose(np.linalg.inv(R11), np.array([[1, -0.4], [-0.4, 1]]) / 0.84, 1e-12)
      and allclose(np.linalg.inv(R22), np.array([[1, -0.3], [-0.3, 1]]) / 0.91, 1e-12), "")
K = R12 @ np.linalg.inv(R22) @ R21
M = np.linalg.inv(R11) @ K
check(T, "R12 R22^-1 R21（小数第4位）", allclose(K, [[0.2747, 0.2132], [0.2132, 0.1670]]), r(K))
check(T, "M=R11^-1 R12 R22^-1 R21（小数第4位）", allclose(M, [[0.2255, 0.1743], [0.1230, 0.0973]]), r(M))
check(T, "特性方程式 tr=0.3229, det=0.000523", close(np.trace(M), 0.3229) and close(np.linalg.det(M), 0.000523, 5e-7),
      (r(np.trace(M)), r(np.linalg.det(M), 6)))
ev, evec = np.linalg.eig(M)
o = np.argsort(ev.real)[::-1]
ev = ev.real[o]
evec = evec.real[:, o]
rho = np.sqrt(ev)
check(T, "ρ1²=0.3212, ρ2²=0.0016", allclose(ev, [0.3212, 0.0016]), r(ev, 6))
check(T, "ρ1=0.567, ρ2=0.040", allclose(rho, [0.567, 0.040], 5e-4), r(rho))
a = [align(evec[:, k] / np.sqrt(evec[:, k] @ R11 @ evec[:, k]), [1, 1] if k == 0 else [1, -1]) for k in range(2)]
b = [np.linalg.inv(R22) @ R21 @ a[k] / rho[k] for k in range(2)]
check(T, "a1=(0.758,0.416), b1=(0.877,0.285)", allclose(a[0], [0.758, 0.416], 5e-4) and allclose(b[0], [0.877, 0.285], 5e-4),
      (r(a[0]), r(b[0])))
# もう一方の固有値問題でも同じ ρ²
ev2 = np.sort(np.linalg.eigvals(np.linalg.inv(R22) @ R21 @ np.linalg.inv(R11) @ R12).real)[::-1]
check(T, "Σ22^-1Σ21Σ11^-1Σ12 の固有値も同じ ρ²", allclose(ev2, ev, 1e-12), r(ev2, 6))
# 直接最大化
Rfull = np.block([[R11, R12], [R21, R22]])


def negcorr(t):
    aa, bb = t[:2], t[2:]
    return -(aa @ R12 @ bb) / np.sqrt(aa @ R11 @ aa) / np.sqrt(bb @ R22 @ bb)


best = min((optimize.minimize(negcorr, rng.normal(size=4), method="Nelder-Mead",
                              options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000}) for _ in range(10)),
           key=lambda z: z.fun)
check(T, "相関の直接最大化（Nelder-Mead、許容1e-6）= ρ1", close(-best.fun, rho[0], 1e-6), r(-best.fun, 6))
check(T, "ρ1 ≥ 群をまたぐ相関の最大絶対値 0.5", rho[0] >= np.abs(R12).max(), "")
okm = True
for _ in range(200):
    Cr = np.corrcoef(rng.normal(size=(30, 5)) @ rng.normal(size=(5, 5)), rowvar=False)
    er = np.linalg.eigvals(np.linalg.inv(Cr[:2, :2]) @ Cr[:2, 2:] @ np.linalg.inv(Cr[2:, 2:]) @ Cr[2:, :2]).real
    okm &= math.sqrt(er.max()) >= np.abs(Cr[:2, 2:]).max() - 1e-12
check(T, "ρ1 ≥ 群をまたぐ相関の最大絶対値（乱数200組、p=2,q=3）", okm, "")
check(T, "ラグランジュ条件 R12 b=ρ R11 a, R21 a=ρ R22 b",
      allclose(R12 @ b[0], rho[0] * R11 @ a[0], 1e-12) and allclose(R21 @ a[0], rho[0] * R22 @ b[0], 1e-12), "")
check(T, "M は非対称", not np.allclose(M, M.T), "")
check(T, "グラフ: bvnorm の rho=0.57 ≈ ρ1", close(front(T)["charts"][0]["params"]["rho"], rho[0], 5e-3), "")
# 例2 q=1
rv = np.array([0.5, 0.4])
r2 = rv @ np.linalg.inv(R11) @ rv
check(T, "例2 ρ1²=0.25/0.84=0.2976, ρ1=0.546",
      close(r2, 0.2976) and close((0.25 + 0.16 - 2 * 0.4 * 0.5 * 0.4), 0.25, 1e-12) and close(math.sqrt(r2), 0.546, 5e-4), r(r2))
# q=1 で重相関と一致（乱数データ、恒等式）
okq = True
for _ in range(20):
    Xd = rng.normal(size=(50, 3))
    yd = Xd @ rng.normal(size=3) + rng.normal(size=50)
    C = np.corrcoef(np.column_stack([Xd, yd]).T)
    rc2 = C[:3, 3] @ np.linalg.inv(C[:3, :3]) @ C[:3, 3]
    Xc = np.column_stack([np.ones(50), Xd])
    res = yd - Xc @ np.linalg.lstsq(Xc, yd, rcond=None)[0]
    R2 = 1 - res @ res / ((yd - yd.mean()) @ (yd - yd.mean()))
    okq &= abs(rc2 - R2) < 1e-9
check(T, "q=1 の ρ1² = 重回帰の決定係数（乱数20組、許容1e-9）", okq, "")
xx = rng.normal(size=100)
yy = -0.6 * xx + rng.normal(size=100)
rxy = np.corrcoef(xx, yy)[0, 1]
check(T, "p=q=1 なら ρ1=|r_xy|", close(math.sqrt(rxy ** 2), abs(rxy), 1e-12) and rxy < 0, r(rxy))
# 標本での確認（シミュレーション）
Ls = np.linalg.cholesky(Rfull)
Xs = rng.normal(size=(200000, 4)) @ Ls.T
Cs = np.corrcoef(Xs.T)
Ms = np.linalg.inv(Cs[:2, :2]) @ Cs[:2, 2:] @ np.linalg.inv(Cs[2:, 2:]) @ Cs[2:, :2]
rs = np.sqrt(np.sort(np.linalg.eigvals(Ms).real)[::-1])
check(T, "シミュレーション 20万点で ρ1≈0.567（許容0.01）", close(rs[0], rho[0], 0.01), r(rs))

# ================================================================ 正準変量
T = "canonical-variate"
check(T, "例1 ρ1=0.5668, a1=(0.7579,0.4162)", close(rho[0], 0.5668) and allclose(a[0], [0.7579, 0.4162]), (r(rho[0]), r(a[0])))
check(T, "a1'R11a1 = 1（主張値で 1.000）",
      close(0.7579 ** 2 + 0.4162 ** 2 + 2 * 0.4 * 0.7579 * 0.4162, 1, 5e-4), r(0.7579 ** 2 + 0.4162 ** 2 + 2 * 0.4 * 0.7579 * 0.4162))
check(T, "R21a1=(0.5454,0.3106)", allclose(R21 @ a[0], [0.5454, 0.3106]), r(R21 @ a[0]))
check(T, "R22^-1R21a1=(0.4970,0.1615)", allclose(np.linalg.inv(R22) @ R21 @ a[0], [0.4970, 0.1615]), r(np.linalg.inv(R22) @ R21 @ a[0]))
check(T, "b1=(0.877,0.285)", allclose(b[0], [0.877, 0.285], 5e-4), r(b[0]))
L1x, L1y = R11 @ a[0], R22 @ b[0]
check(T, "正準負荷量 R11a1=(0.924,0.719), R22b1=(0.962,0.548)",
      allclose(L1x, [0.924, 0.719], 5e-4) and allclose(L1y, [0.962, 0.548], 5e-4), (r(L1x), r(L1y)))
check(T, "交差負荷量 (0.524,0.408)=R12b1=ρ1R11a1",
      allclose(rho[0] * L1x, [0.524, 0.408], 5e-4) and allclose(R12 @ b[0], rho[0] * L1x, 1e-12), r(rho[0] * L1x))
# 正準変量の相関構造
Wm = np.zeros((4, 4))
Wm[:2, 0], Wm[:2, 1], Wm[2:, 2], Wm[2:, 3] = a[0], a[1], b[0], b[1]
Cuv = Wm.T @ Rfull @ Wm
target = np.array([[1, 0, rho[0], 0], [0, 1, 0, rho[1]], [rho[0], 0, 1, 0], [0, rho[1], 0, 1]])
check(T, "Corr(Uk,Vk)=ρk、それ以外の組は無相関、分散1", allclose(Cuv, target, 1e-10), r(Cuv))
check(T, "Corr(x,Vk)=ρk R11 ak（k=2 も）", allclose(R12 @ b[1], rho[1] * R11 @ a[1], 1e-12), "")
c = chart(T, "points-2d")
P = {p["label"]: (p["x"], p["y"]) for p in c["params"]["points"]}
L2x, L2y = R11 @ a[1], R22 @ b[1]
cl = np.array([P["運動時間"], P["睡眠時間"], P["体力"], P["集中力"]])
s2 = 1 if L2x @ cl[:2, 1] >= 0 else -1   # a2, b2 は同時に符号反転のみ可（ρ2>0 を保つ）
mine = np.column_stack([np.r_[L1x, L1y], s2 * np.r_[L2x, L2y]])
check(T, "charts points-2d の正準負荷量（第2軸は a2,b2 同時の符号反転のみ許す、小数第4位）", allclose(mine, cl, 5e-5), r(mine))
check(T, "charts の点は単位円内（p=q=2 では厳密に単位円上、表示の丸めで 1e-4 まで許容）", bool(np.all(np.linalg.norm(cl, axis=1) <= 1 + 1e-4)), r(np.linalg.norm(cl, axis=1)))
# バートレット
n, p, q = 100, 2, 2
cf = n - 1 - (p + q + 1) / 2
check(T, "補正係数 96.5", cf == 96.5, cf)
sl = -(np.log(1 - ev[0]) + np.log(1 - ev[1]))
chi0, chi1 = cf * sl, -cf * np.log(1 - ev[1])
check(T, "k=0: Σ=0.3891, χ²=37.55", close(sl, 0.3891) and close(chi0, 37.55, 5e-3), (r(sl), r(chi0)))
check(T, "主張値の丸め ρ²=0.3212,0.0016 から計算しても 37.55（入力の丸めの伝播で 0.02 まで許容）",
      close(-96.5 * (math.log(0.6788) + math.log(0.9984)), 37.55, 0.02), r(-96.5 * (math.log(0.6788) + math.log(0.9984))))
check(T, "χ²(4) 上側5%点 9.488, χ²(1) 3.841",
      close(stats.chi2.ppf(0.95, 4), 9.488, 5e-4) and close(stats.chi2.ppf(0.95, 1), 3.841, 5e-4), "")
check(T, "k=1: χ²=0.157 < 3.841", close(chi1, 0.157, 5e-4) and chi1 < 3.841 and chi0 > 9.488, r(chi1))
lam_w = (1 - ev).prod()
check(T, "ウィルクスのラムダ 0.678 = |R|/(|R11||R22|)",
      close(lam_w, 0.678, 5e-4) and close(np.linalg.det(Rfull) / np.linalg.det(R11) / np.linalg.det(R22), lam_w, 1e-12), r(lam_w))
# H0 のもとでのバートレット統計量の分布（シミュレーション）
stat = []
for _ in range(4000):
    Xh = rng.normal(size=(100, 4))
    C = np.corrcoef(Xh.T)
    e = np.linalg.eigvals(np.linalg.inv(C[:2, :2]) @ C[:2, 2:] @ np.linalg.inv(C[2:, 2:]) @ C[2:, :2]).real
    stat.append(-cf * np.log(1 - e).sum())
stat = np.array(stat)
check(T, "H0 で統計量の平均≈4、棄却率≈0.05（4000回、許容 0.15・0.012）",
      abs(stat.mean() - 4) < 0.15 and abs((stat > 9.488).mean() - 0.05) < 0.012, (r(stat.mean()), r((stat > 9.488).mean())))

# ================================================================ コレスポンデンス分析
T = "correspondence-analysis"
N = np.array([[30, 15, 5], [20, 20, 20], [10, 15, 35]], float)
nn = N.sum()
Pm = N / nn
rr, cc = Pm.sum(1), Pm.sum(0)
check(T, "n=170, r=(0.2941,0.3529,0.3529), c=(0.3529,0.2941,0.3529)",
      nn == 170 and allclose(rr, [0.2941, 0.3529, 0.3529]) and allclose(cc, [0.3529, 0.2941, 0.3529]), (r(rr), r(cc)))
Ex = nn * np.outer(rr, cc)
check(T, "期待度数 e11=17.647", close(Ex[0, 0], 17.647, 5e-4), r(Ex[0, 0]))
chi2, pval, dof, _ = stats.chi2_contingency(N, correction=False)
check(T, "χ²=33.481, 自由度4, p<0.001", close(chi2, 33.481, 5e-4) and dof == 4 and pval < 0.001, (r(chi2), pval))
check(T, "全イナーシャ 0.1969", close(chi2 / nn, 0.1969), r(chi2 / nn))
S = (Pm - np.outer(rr, cc)) / np.sqrt(np.outer(rr, cc))
check(T, "s11=0.2255=(30-17.647)/sqrt(170×17.647)", close(S[0, 0], 0.2255) and close(S[0, 0], (30 - Ex[0, 0]) / math.sqrt(nn * Ex[0, 0]), 1e-12), r(S[0, 0]))
check(T, "標準化残差行列 S（小数第4位）",
      allclose(S, [[0.2255, 0.0059, -0.2309], [-0.0196, 0.0430, -0.0196], [-0.1863, -0.0483, 0.2304]]), r(S))
U, g, Vt = np.linalg.svd(S)
check(T, "Σs²=Σγ²=χ²/n", close((S ** 2).sum(), chi2 / nn, 1e-12) and close((g ** 2).sum(), chi2 / nn, 1e-12), "")
check(T, "γ1=0.4393, γ2=0.0632, γ3=0（軸は min(I,J)-1=2）",
      close(g[0], 0.4393) and close(g[1], 0.0632) and g[2] < 1e-12, r(g))
check(T, "γ1²=0.1929, γ2²=0.0040, 寄与率 97.97%",
      close(g[0] ** 2, 0.1929) and close(g[1] ** 2, 0.0040) and close(g[0] ** 2 / (g ** 2).sum() * 100, 97.97, 5e-3),
      r(g[0] ** 2 / (g ** 2).sum() * 100))
F = (U[:, :2] / np.sqrt(rr)[:, None]) * g[:2]
G = (Vt.T[:, :2] / np.sqrt(cc)[:, None]) * g[:2]
cpts = chart(T, "points-2d")["params"]
Pc = np.array([[p["x"], p["y"]] for p in cpts["points"]])
for k in range(2):  # 行と列は同じ符号反転のみ
    if np.r_[F[:, k], G[:, k]] @ Pc[:, k] < 0:
        F[:, k] *= -1
        G[:, k] *= -1
check(T, "第1軸 行 -0.593,-0.009,0.503 列 -0.491,-0.070,0.549",
      allclose(F[:, 0], [-0.593, -0.009, 0.503], 5e-4) and allclose(G[:, 0], [-0.491, -0.070, 0.549], 5e-4), (r(F[:, 0], 3), r(G[:, 0], 3)))
check(T, "charts points-2d の主座標（行列同時の符号反転のみ、小数第4位）", allclose(np.vstack([F, G]), Pc, 5e-5), r(np.vstack([F, G])))
check(T, "charts 軸ラベルの寄与率 97.97%, 2.03%",
      "97.97%" in cpts["xlabel"] and "2.03%" in cpts["ylabel"] and close(g[1] ** 2 / (g ** 2).sum() * 100, 2.03, 5e-3), "")
check(T, "charts mosaic の表が N と一致", allclose(chart(T, "mosaic")["params"]["table"], N, 0), "")
prof = N[0] / N[0].sum()
check(T, "20代の行プロファイル (0.6,0.3,0.1)", allclose(prof, [0.6, 0.3, 0.1], 1e-12), "")
num = 0.6 * -0.491 + 0.3 * -0.070 + 0.1 * 0.549
check(T, "推移公式 -0.2606/0.4393=-0.593（主張値で計算）", close(num, -0.2606, 5e-4) and close(num / 0.4393, -0.593, 5e-4), r(num))
Ft = np.diag(1 / rr) @ Pm @ G / g[:2]
check(T, "推移公式 F=D_r^-1 P G Γ^-1（厳密）", allclose(Ft, F, 1e-12), "")
check(T, "Σr_iF_ik=0, Σr_iF_ik²=γ_k²", allclose(rr @ F, 0, 1e-12) and allclose(rr @ F ** 2, g[:2] ** 2, 1e-12), "")
chid = np.array([[((Pm[i] / rr[i] - Pm[j] / rr[j]) ** 2 / cc).sum() for j in range(3)] for i in range(3)])
check(T, "主座標のユークリッド距離²=χ²距離", allclose(((F[:, None] - F[None]) ** 2).sum(2), chid, 1e-12), r(chid))
U0, g0, V0 = np.linalg.svd(np.diag(rr ** -0.5) @ Pm @ np.diag(cc ** -0.5))
check(T, "自明解: D_r^-1/2 P D_c^-1/2 の最大特異値1, 特異ベクトル √r, √c",
      close(g0[0], 1, 1e-12) and allclose(np.abs(U0[:, 0]), np.sqrt(rr), 1e-12) and allclose(np.abs(V0[0]), np.sqrt(cc), 1e-12)
      and allclose(g0[1:], g[:2], 1e-12), r(g0))

# ================================================================ 数量化理論
T = "quantification-theory"
grp = {"新築": [10.0, 9.5, 8.0], "中古": [8.5, 7.0], "古い": [7.0, 5.0, 5.5]}
ally = np.concatenate(list(grp.values()))
means = np.array([np.mean(v) for v in grp.values()])
cnt = np.array([len(v) for v in grp.values()])
check(T, "例1 平均 9.1667, 7.7500, 5.8333, 全体 7.5625",
      allclose(means, [9.1667, 7.75, 5.8333]) and close(ally.mean(), 7.5625, 1e-12), r(means))
sc = means - ally.mean()
check(T, "例1 スコア 1.6042, 0.1875, -1.7292、重み付き和0",
      allclose(sc, [1.6042, 0.1875, -1.7292]) and close(cnt @ sc, 0, 1e-12), r(sc))
ssb = (cnt * sc ** 2).sum()
sst = ((ally - ally.mean()) ** 2).sum()
check(T, "例1 群間平方和 16.760, 全平方和 22.219, R²=0.754",
      close(ssb, 16.760, 5e-4) and close(sst, 22.219, 5e-4) and close(ssb / sst, 0.754, 5e-4), (r(ssb), r(sst), r(ssb / sst)))
Xo = np.column_stack([np.ones(8), np.repeat([0, 1, 0], cnt), np.repeat([0, 0, 1], cnt)])
res = ally - Xo @ np.linalg.lstsq(Xo, ally, rcond=None)[0]
check(T, "例1 最小二乗の予測値=カテゴリー平均（R² 一致）", close(1 - res @ res / sst, ssb / sst, 1e-12), "")
check(T, "例1 レンジ 3.333", close(sc.max() - sc.min(), 3.333, 5e-4), r(sc.max() - sc.min()))
check(T, "charts bar-data のスコア", allclose(chart(T, "bar-data")["params"]["values"], sc), "")
# 例2 II類
tab = np.array([[20, 10, 5], [10, 15, 20]], float)
prop = tab[0] / tab.sum(0)
check(T, "例2 購入割合 (0.667,0.400,0.200)", allclose(prop, [0.667, 0.4, 0.2], 5e-4), r(prop))
ng = tab.sum(1)
gm = tab @ prop / ng
om = tab.sum(0) @ prop / tab.sum()
check(T, "例2 群平均 0.5238, 0.3704, 全体 0.4375", allclose(gm, [0.5238, 0.3704]) and close(om, 0.4375), (r(gm), r(om)))


def eta2(x):
    x = np.asarray(x, float)
    tot = (tab.sum(0) * (x - (tab.sum(0) @ x) / tab.sum()) ** 2).sum()
    mu = tab @ x / ng
    m0 = tab.sum(0) @ x / tab.sum()
    return (ng * (mu - m0) ** 2).sum() / tot


e2 = eta2(prop)
check(T, "例2 η²=0.1534", close(e2, 0.1534), r(e2))
check(T, "例2 群間平方和の分子の式", close((35 * (gm[0] - om) ** 2 + 45 * (gm[1] - om) ** 2) / e2,
                                       (tab.sum(0) * (prop - om) ** 2).sum(), 1e-12), "")
best = min((optimize.minimize(lambda x: -eta2(x), rng.normal(size=3), method="Nelder-Mead",
                              options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 10000}) for _ in range(10)), key=lambda z: z.fun)
check(T, "例2 η² の最大値（数値最大化、許容1e-6）= 0.1534", close(-best.fun, e2, 1e-6), r(-best.fun, 6))
chi2b = stats.chi2_contingency(tab, correction=False)[0]
check(T, "例2 χ²=12.275, φ²=χ²/80=0.1534=η²", close(chi2b, 12.275, 5e-4) and close(chi2b / 80, e2, 1e-12), r(chi2b))
check(T, "例2 1次変換で η² 不変", close(eta2(3 * prop - 2), e2, 1e-12), "")

# ================================================================ 数量化I類
T = "quantification-type-1"
far = np.array([0, 0, 0, 0, 1, 1, 1, 1])
age = np.array([0, 1, 2, 0, 0, 1, 2, 2])
yv = np.array([10.0, 8.5, 7.0, 9.5, 8.0, 7.0, 5.0, 5.5])
Xq = np.column_stack([np.ones(8), far, age == 1, age == 2]).astype(float)
check(T, "正規方程式の X'X と X'y",
      allclose(Xq.T @ Xq, [[8, 4, 2, 3], [4, 4, 1, 2], [2, 1, 2, 0], [3, 2, 0, 3]], 0) and allclose(Xq.T @ yv, [60.5, 25.5, 15.5, 17.5], 1e-12), "")
bq = np.linalg.solve(Xq.T @ Xq, Xq.T @ yv)
check(T, "b=(9.7273,-1.6818,-1.1364,-2.7727)", allclose(bq, [9.7273, -1.6818, -1.1364, -2.7727]), r(bq))
s_st = np.array([0, bq[1]])
s_age = np.array([0, bq[2], bq[3]])
m_st = s_st @ [4, 4] / 8
m_age = s_age @ [3, 2, 3] / 8
st = s_st - m_st
ag = s_age - m_age
check(T, "駅距離 平均 -0.8409 → (0.8409,-0.8409)", close(m_st, -0.8409) and allclose(st, [0.8409, -0.8409]), r(st))
check(T, "築年 重み付き平均 -1.3239 → (1.3239,0.1875,-1.4489)", close(m_age, -1.3239) and allclose(ag, [1.3239, 0.1875, -1.4489]), r(ag))
b0 = bq[0] + m_st + m_age
check(T, "定数項 7.5625=ȳ", close(b0, 7.5625, 1e-9) and close(yv.mean(), 7.5625, 1e-12), r(b0))
check(T, "レンジ 1.682, 2.773", close(st.max() - st.min(), 1.682, 5e-4) and close(ag.max() - ag.min(), 2.773, 5e-4), "")
check(T, "予測 駅近・中古 8.591", close(b0 + st[0] + ag[1], 8.591, 5e-4) and close(Xq[1] @ bq, b0 + st[0] + ag[1], 1e-9), r(b0 + st[0] + ag[1]))
yh = Xq @ bq
R2q = 1 - ((yv - yh) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
check(T, "R²=0.988", close(R2q, 0.988, 5e-4), r(R2q))
check(T, "基準化しても予測値は不変", allclose(b0 + st[far] + ag[age], yh, 1e-9), "")
Xfull = np.column_stack([np.ones(8), far == 0, far == 1, age == 0, age == 1, age == 2]).astype(float)
check(T, "全ダミー+定数項は階数落ち（rank 4 < 6 列）", np.linalg.matrix_rank(Xfull) == 4, np.linalg.matrix_rank(Xfull))
check(T, "charts bar-data のスコア", allclose(chart(T, "bar-data")["params"]["values"], np.r_[st, ag]), "")
check(T, "charts caption のレンジ 2.773, 1.682", "2.773" in chart(T, "bar-data")["caption"] and "1.682" in chart(T, "bar-data")["caption"], "")
check(T, "単純平均の差は調整済みスコアと異なる（新築-古い: 3.333 vs 2.773）",
      abs((means[0] - means[2]) - (ag[0] - ag[2])) > 0.1, (r(means[0] - means[2]), r(ag[0] - ag[2])))

# ================================================================ 数量化III類
T = "quantification-type-3"
Fq = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], float)
rq, cq, Tq = Fq.sum(1), Fq.sum(0), Fq.sum()
check(T, "r=(2,3,2,2,1), c=(2,3,3,2), T=10", allclose(rq, [2, 3, 2, 2, 1], 0) and allclose(cq, [2, 3, 3, 2], 0) and Tq == 10, "")
H = np.diag(cq ** -0.5) @ Fq.T @ np.diag(1 / rq) @ Fq @ np.diag(cq ** -0.5)
check(T, "h11=(1/2+1/3)/2=0.4167", close(H[0, 0], 0.4167), r(H[0, 0]))
check(T, "H（小数第4位）", allclose(H, [[0.4167, 0.3402, 0.1361, 0], [0.3402, 0.4444, 0.2778, 0],
                                     [0.1361, 0.2778, 0.4444, 0.2041], [0, 0, 0.2041, 0.75]]), r(H))
lh, Vh = np.linalg.eigh(H)
o = np.argsort(lh)[::-1]
lh, Vh = lh[o], Vh[:, o]
check(T, "固有値 1, 0.7566, 0.2500, 0.0490", allclose(lh, [1, 0.7566, 0.25, 0.0490]), r(lh))
check(T, "最大固有値1の固有ベクトル ∝ √c（自明解）", allclose(np.abs(Vh[:, 0]), np.sqrt(cq / Tq), 1e-12), "")
check(T, "ρ1=0.870", close(math.sqrt(lh[1]), 0.870, 5e-4), r(math.sqrt(lh[1])))
cp = chart(T, "points-2d")["params"]
Pq = np.array([[p["x"], p["y"]] for p in cp["points"]])
Y = np.zeros((4, 2))
Xp = np.zeros((5, 2))
for k in range(2):
    yk = np.diag(cq ** -0.5) @ Vh[:, k + 1]
    yk = yk / np.sqrt(cq @ yk ** 2 / Tq)
    rk = math.sqrt(lh[k + 1])
    xk = (Fq @ yk) / rq / rk
    sgn = 1 if np.r_[yk, xk] @ Pq[:, k] >= 0 else -1  # 人と項目は同じ符号反転のみ
    Y[:, k], Xp[:, k] = sgn * yk, sgn * xk
check(T, "項目スコア 第1軸 (-0.859,-0.720,0.049,1.866)、Σc y²/T=1",
      allclose(Y[:, 0], [-0.859, -0.720, 0.049, 1.866], 5e-4) and close(cq @ Y[:, 0] ** 2 / Tq, 1, 1e-12), r(Y[:, 0], 3))
check(T, "A の計算 (1/0.870)(-0.859-0.720)/2=-0.908（丸めた入力の伝播で 1e-3 まで許容。厳密値は下で照合）",
      close((-0.859 - 0.720) / 2 / 0.870, -0.908, 1e-3), r((-0.859 - 0.720) / 2 / 0.870))
check(T, "人のスコア A -0.908, B -0.586, C -0.386, D 1.101, E 2.145",
      allclose(Xp[:, 0], [-0.908, -0.586, -0.386, 1.101, 2.145], 5e-4), r(Xp[:, 0], 3))
check(T, "Σr x=Σc y=0（自明解と直交）", close(rq @ Xp[:, 0], 0, 1e-9) and close(cq @ Y[:, 0], 0, 1e-9), "")
ii, jj = np.nonzero(Fq)
rc_ = np.corrcoef(Xp[ii, 0], Y[jj, 0])[0, 1]
check(T, "反応のある10組での相関 0.870 = ρ1", len(ii) == 10 and close(rc_, math.sqrt(lh[1]), 1e-9), r(rc_))
check(T, "相互平均 y=(1/ρ)(F'x)/c", allclose(Y[:, 0], (Fq.T @ Xp[:, 0]) / cq / math.sqrt(lh[1]), 1e-9), "")
check(T, "charts points-2d（人・項目、第1・第2軸、軸ごとの共通符号のみ、小数第4位）",
      allclose(np.vstack([Y, Xp]), Pq, 5e-5), r(np.vstack([Y, Xp])))
check(T, "charts 軸ラベル ρ²=0.757, 0.250", "0.757" in cp["xlabel"] and "0.250" in cp["ylabel"], "")
check(T, "グラフ説明: A・E が下、C が上（弓なり）",
      Pq[4, 1] < 0 and Pq[8, 1] < 0 and Pq[6, 1] > 0 and Pq[6, 1] == Pq[:, 1][4:].max(), "")
# CA との同値: F の標準化残差の特異値 = ρ
Pf = Fq / Tq
Sf = (Pf - np.outer(Pf.sum(1), Pf.sum(0))) / np.sqrt(np.outer(Pf.sum(1), Pf.sum(0)))
gf = np.linalg.svd(Sf, compute_uv=False)
check(T, "コレスポンデンス分析の特異値² = 1 を除いた固有値", allclose(gf[:3] ** 2, lh[1:], 1e-12), r(gf ** 2))

# ================================================================ 構造方程式モデリング
T = "structural-equation-modeling"
l1 = math.sqrt(0.48 * 0.42 / 0.56)
lam3 = np.array([l1, 0.48 / l1, 0.42 / l1])
check(T, "例1 λ1²=0.36, λ=(0.6,0.8,0.7)", close(l1 ** 2, 0.36, 1e-12) and allclose(lam3, [0.6, 0.8, 0.7], 1e-12), r(lam3))
check(T, "例1 ψ=(0.64,0.36,0.51)", allclose(1 - lam3 ** 2, [0.64, 0.36, 0.51], 1e-12), "")
S3 = np.array([[1, 0.48, 0.42], [0.48, 1, 0.56], [0.42, 0.56, 1]])
Sig3 = np.outer(lam3, lam3) + np.diag(1 - lam3 ** 2)
check(T, "例1 相関行列を完全に再現、df=6-6=0", allclose(Sig3, S3, 1e-12) and 3 * 4 // 2 - 6 == 0, "")
S4 = np.array([[1, 0.48, 0.42, 0.40], [0.48, 1, 0.56, 0.40], [0.42, 0.56, 1, 0.35], [0.40, 0.40, 0.35, 1]])


def fml(Sg, Sm):
    return np.linalg.slogdet(Sg)[1] + np.trace(Sm @ np.linalg.inv(Sg)) - np.linalg.slogdet(Sm)[1] - len(Sm)


def fobj(t):
    lam, lps = t[:4], t[4:]
    return fml(np.outer(lam, lam) + np.diag(np.exp(lps)), S4)


check(T, "F_ML(S)=0", close(fml(S4, S4), 0, 1e-12), "")
fits = [optimize.minimize(fobj, np.r_[rng.uniform(0.3, 0.9, 4), np.log(rng.uniform(0.2, 0.8, 4))], method="BFGS",
                          options={"gtol": 1e-11}) for _ in range(8)]
fit = min(fits, key=lambda z: z.fun)
lh4 = np.abs(fit.x[:4])
ps4 = np.exp(fit.x[4:])
check(T, "例2 df=10-8=2", 4 * 5 // 2 - 8 == 2, "")
check(T, "例2 λ̂=(0.630,0.779,0.695,0.536)", allclose(lh4, [0.630, 0.779, 0.695, 0.536], 5e-4), r(lh4))
check(T, "例2 ψ̂=(0.603,0.393,0.516,0.713)", allclose(ps4, [0.603, 0.393, 0.516, 0.713], 5e-4), r(ps4))
check(T, "例2 F_ML=0.01430", close(fit.fun, 0.01430, 5e-6), r(fit.fun, 6))
chis = 199 * fit.fun
check(T, "例2 χ²=199×F=2.845", close(chis, 2.845, 5e-4) and close(199 * 0.01430, 2.8457, 5e-4), r(chis))
pv = stats.chi2.sf(chis, 2)
check(T, "例2 χ²(2) 上側5%点 5.991, p=0.241", close(stats.chi2.ppf(0.95, 2), 5.991, 5e-4) and close(pv, 0.241, 5e-4), r(pv))
rmsea = math.sqrt(max(chis - 2, 0) / (2 * 199))
check(T, "例2 RMSEA=0.046", close(rmsea, 0.046, 5e-4), r(rmsea))
Sh = np.outer(lh4, lh4) + np.diag(ps4)
Res = S4 - Sh
iu = np.triu_indices(4, 1)
k = np.argmax(np.abs(Res[iu]))
check(T, "例2 最大残差は x1-x4 の 0.062", (iu[0][k], iu[1][k]) == (0, 3) and close(Res[0, 3], 0.062, 5e-4), r(Res[iu]))
check(T, "例2 Σ̂ の対角=1（標準化負荷量）", allclose(np.diag(Sh), 1, 1e-5), r(np.diag(Sh), 6))
check(T, "charts graph-diagram の weights（小数第2位）", allclose(chart(T, "graph-diagram")["params"]["weights"], lh4, 5e-3), "")

# ================================================================ パス解析
T = "path-analysis"
rxm, rxy, rmy = 0.5, 0.4, 0.6
cp_ = (rxy - rxm * rmy) / (1 - rxm ** 2)
bp = (rmy - rxm * rxy) / (1 - rxm ** 2)
check(T, "c'=0.1/0.75=0.133, b=0.4/0.75=0.533", close(cp_, 0.133, 5e-4) and close(bp, 0.533, 5e-4), (r(cp_), r(bp)))
sol = np.linalg.solve([[1, rxm], [rxm, 1]], [rxy, rmy])
check(T, "連立方程式 r_xy=c'+b r_xm, r_my=c' r_xm+b の解と一致", allclose(sol, [cp_, bp], 1e-12), "")
check(T, "間接効果 0.267, 総合効果 0.400=r_xy", close(rxm * bp, 0.267, 5e-4) and close(cp_ + rxm * bp, rxy, 1e-12), r(rxm * bp))
check(T, "間接効果の割合 約2/3", close(rxm * bp / rxy, 2 / 3, 1e-12), "")
R2p = cp_ * rxy + bp * rmy
check(T, "R²=0.373, 誤差パス 0.792, m の誤差パス 0.866",
      close(R2p, 0.373, 5e-4) and close(math.sqrt(1 - R2p), 0.792, 5e-4) and close(math.sqrt(0.75), 0.866, 5e-4), r(R2p))
check(T, "主張値 0.133×0.4+0.533×0.6=0.373", close(0.133 * 0.4 + 0.533 * 0.6, 0.373, 5e-4), "")
check(T, "例2 r̂_xy=0.30, ずれ 0.10", close(0.5 * 0.6, 0.30, 1e-12) and close(rxy - 0.3, 0.10, 1e-12), "")
check(T, "charts weights (0.500,0.533,0.133)", allclose(chart(T, "graph-diagram")["params"]["weights"], [rxm, bp, cp_], 5e-4), "")
# シミュレーション
nS = 200000
xs = rng.normal(size=nS)
ms = rxm * xs + math.sqrt(1 - rxm ** 2) * rng.normal(size=nS)
ys = cp_ * xs + bp * ms + math.sqrt(1 - R2p) * rng.normal(size=nS)
Cm = np.corrcoef([xs, ms, ys])
Zs = np.column_stack([(xs - xs.mean()) / xs.std(), (ms - ms.mean()) / ms.std()])
coef = np.linalg.lstsq(Zs, (ys - ys.mean()) / ys.std(), rcond=None)[0]
check(T, "シミュレーション 20万点: 相関 (0.5,0.4,0.6)・標準化偏回帰係数 (0.133,0.533)（許容0.01）",
      allclose([Cm[0, 1], Cm[0, 2], Cm[1, 2]], [0.5, 0.4, 0.6], 0.01) and allclose(coef, [cp_, bp], 0.01) and close(ys.var(), 1, 0.01),
      (r([Cm[0, 1], Cm[0, 2], Cm[1, 2]], 3), r(coef, 3)))

# ================================================================ 独立成分分析
T = "independent-component-analysis"
A = np.array([[1, 1], [0, 1.]])
Sx = A @ A.T
lx, Ex_ = np.linalg.eigh(Sx)
check(T, "Σx=AA'=[[2,1],[1,1]], 固有値 2.618, 0.382",
      allclose(Sx, [[2, 1], [1, 1]], 0) and allclose(sorted(lx, reverse=True), [2.618, 0.382], 5e-4), r(lx))
Sih = Ex_ @ np.diag(lx ** -0.5) @ Ex_.T
check(T, "Σx^-1/2=[[0.894,-0.447],[-0.447,1.342]]", allclose(Sih, [[0.894, -0.447], [-0.447, 1.342]], 5e-4), r(Sih, 3))
Rm = Sih @ A
check(T, "Σx^-1/2 A=[[0.894,0.447],[-0.447,0.894]] は直交", allclose(Rm, [[0.894, 0.447], [-0.447, 0.894]], 5e-4)
      and allclose(Rm @ Rm.T, np.eye(2), 1e-12), r(Rm, 3))
ang = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))
check(T, "回転角 θ≈-26.57°（cos=0.894, sin=-0.447）", close(ang, -26.57, 5e-3), r(ang))
ku = stats.uniform.stats(moments="k")
kl = stats.laplace.stats(moments="k")
check(T, "一様の超過尖度 -1.2, ラプラスの超過尖度 3", close(ku, -1.2, 1e-12) and close(kl, 3, 1e-12), (float(ku), float(kl)))
check(T, "κ(x1)=-1.2×(0.25+0.25)=-0.6", close(-1.2 * 2 * (1 / math.sqrt(2)) ** 4, -0.6, 1e-12), "")
check(T, "κ(z1)=-1.2×(0.8²+0.2²)=-0.816（z1 の係数² = 0.8, 0.2）",
      allclose(Rm[0] ** 2, [0.8, 0.2], 1e-12) and close(-1.2 * (Rm[0] ** 4).sum(), -0.816, 1e-12), r(Rm[0] ** 2))


def kth(t, k0):
    t = math.radians(t)
    return k0 * (math.cos(t) ** 4 + math.sin(t) ** 4)


check(T, "κ=-1.2(cos⁴+sin⁴)=-1.2(1-sin²2θ/2)（θ全域）",
      all(close(kth(t, -1.2), -1.2 * (1 - 0.5 * math.sin(math.radians(2 * t)) ** 2), 1e-12) for t in range(0, 181, 5)), "")
check(T, "θ=0°,22.5°,45° で -1.2, -0.9, -0.6",
      allclose([kth(0, -1.2), kth(22.5, -1.2), kth(45, -1.2)], [-1.2, -0.9, -0.6], 1e-12), "")
nS = 200000
s1 = rng.uniform(-math.sqrt(3), math.sqrt(3), nS)
s2 = rng.uniform(-math.sqrt(3), math.sqrt(3), nS)
sim = [exkurt(math.cos(math.radians(t)) * s1 + math.sin(math.radians(t)) * s2) for t in (0, 22.5, 45)]
check(T, "シミュレーション 20万点 -1.200,-0.898,-0.603 を再現（許容0.02）", allclose(sim, [-1.2, -0.898, -0.603], 0.02), r(sim, 3))
x1s = s1 + s2
check(T, "シミュレーション κ(x1)≈-0.6, κ(z1)≈-0.816（許容0.02）",
      close(exkurt(x1s), -0.6, 0.02) and close(exkurt(Rm[0, 0] * s1 + Rm[0, 1] * s2), -0.816, 0.02), "")
Xmix = np.column_stack([s1, s2]) @ A.T
Zw = (Xmix - Xmix.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(Xmix.T))).T
kk = [abs(exkurt(Zw @ [math.cos(t), math.sin(t)])) for t in np.radians(np.arange(0, 180, 0.5))]
tmax = np.arange(0, 180, 0.5)[int(np.argmax(kk))]
Wsep = np.array([math.cos(math.radians(tmax)), math.sin(math.radians(tmax))]) @ np.linalg.inv(np.linalg.cholesky(np.cov(Xmix.T)))
yy_ = (Xmix - Xmix.mean(0)) @ Wsep
cs = max(abs(np.corrcoef(yy_, s1)[0, 1]), abs(np.corrcoef(yy_, s2)[0, 1]))
check(T, "白色化後に |κ| 最大の方向が源信号の一つを復元（|相関|>0.99）", cs > 0.99, r(cs))
c = chart(T, "line-data")["params"]
th = c["x"]
ok = True
for sname, k0 in [("一様", -1.2), ("ラプラス", 3), ("正規", 0)]:
    sr = [s for s in c["series"] if s["label"].startswith(sname)][0]
    ok &= allclose(sr["y"], [kth(t, k0) for t in th], 5e-5)
check(T, "charts line-data の3系列（k0(cos⁴+sin⁴)、小数第4位）", ok, "")
g1 = rng.normal(size=nS)
g2 = rng.normal(size=nS)
check(T, "正規の源信号はどの方向も尖度0（許容0.03）",
      all(abs(exkurt(math.cos(t) * g1 + math.sin(t) * g2)) < 0.03 for t in np.radians([0, 30, 45, 60])), "")
la = rng.laplace(scale=1 / math.sqrt(2), size=nS)
check(T, "ラプラスと一様の混合でもキュムラントは加法的 κ=Σq⁴κ_j（45°、許容0.05）",
      close(exkurt((la + s1) / math.sqrt(2)), 0.25 * 3 + 0.25 * -1.2, 0.05), "")

# ================================================================ 重要度Cの用語
T = "nonmetric-mds"


def pava(y):
    """単調回帰（隣接違反の併合）。"""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * w for v, w in blocks])


def stress_1d(pos):
    A_, B_, C_, D_ = pos
    d = np.abs(np.array([A_ - B_, B_ - C_, A_ - C_, C_ - D_, B_ - D_, A_ - D_]))
    dh = pava(d)
    return d, dh, math.sqrt(((d - dh) ** 2).sum() / (d ** 2).sum())


dn, dhn, stn = stress_1d([0, 1, 1.8, 4])
check(T, "距離 (1, 0.8, 1.8, 2.2, 3, 4)", allclose(dn, [1, 0.8, 1.8, 2.2, 3, 4], 1e-12), r(dn))
check(T, "単調回帰のディスパリティ (0.9, 0.9, 1.8, 2.2, 3, 4)", allclose(dhn, [0.9, 0.9, 1.8, 2.2, 3, 4], 1e-12), r(dhn))
check(T, "分母 34.72、Stress=√(0.02/34.72)≈0.024", close((dn ** 2).sum(), 34.72, 1e-10) and close(((dn - dhn) ** 2).sum(), 0.02, 1e-12) and close(stn, 0.024, 5e-4), r(stn))
okiso = True
for _ in range(300):
    yv = rng.normal(size=6)
    fit = pava(yv)
    okiso &= bool(np.all(np.diff(fit) >= -1e-12))
    for _ in range(20):
        zv = np.sort(rng.normal(size=6))
        okiso &= ((yv - zv) ** 2).sum() >= ((yv - fit) ** 2).sum() - 1e-12
check(T, "PAVA の結果は単調で、ランダムな単調列より2乗誤差が小さい（乱数300×20）", okiso, "成立")
_, _, st0 = stress_1d([0, 0.85, 1.8, 4])
check(T, "B を A 側へ寄せて B と C を離す（B=0.85）と順序が完全に守られ Stress=0", st0 < 1e-12, st0)
_, _, stc = stress_1d([0, 1, 2.05, 4])
check(T, "（参考）D を固定して C だけ右へ動かすと順序は守れない（BC>AB と CD>AC が両立しない）", stc > 0, r(stc))
check(T, "ストレスは配置の拡大縮小で不変", close(stress_1d(3 * np.array([0, 1, 1.8, 4]))[2], stn, 1e-12), "成立")

T = "multiple-correspondence-analysis"
Nm = np.array([[30, 15, 5], [20, 20, 20], [10, 15, 35]])
Zi = np.array([np.r_[np.eye(3)[i], np.eye(3)[j]] for i in range(3) for j in range(3) for _ in range(Nm[i, j])])


def ca_svd(M):
    Pq = M / M.sum()
    rq, cq = Pq.sum(1), Pq.sum(0)
    Sq = (Pq - np.outer(rq, cq)) / np.sqrt(np.outer(rq, cq))
    Uq, sq, Vtq = np.linalg.svd(Sq, full_matrices=False)
    return rq, cq, Uq, sq, Vtq


check(T, "指示行列 170×6、各行の1は Q=2 個", Zi.shape == (170, 6) and np.all(Zi.sum(1) == 2), Zi.shape)
_, cz, Uz, sz, Vtz = ca_svd(Zi)
lz = sz ** 2
check(T, "固有値 0.7196, 0.5316, 0.4684, 0.2804, 0, 0、合計 2=(J−Q)/Q", allclose(lz, [0.7196, 0.5316, 0.4684, 0.2804, 0, 0], 5e-5) and close(lz.sum(), (6 - 2) / 2, 1e-10), r(lz))
_, _, _, gq, _ = ca_svd(Nm.astype(float))
check(T, "分割表の γ=(0.4393, 0.0632)、(1±γ)/2 が指示行列の固有値", allclose(gq[:2], [0.4393, 0.0632], 5e-5)
      and allclose(np.sort(lz[:4])[::-1], np.sort(np.r_[(1 + gq[:2]) / 2, (1 - gq[:2]) / 2])[::-1], 1e-10), r(gq[:2]))
check(T, "(1+0.4393)/2=0.7196、(1+0.0632)/2=0.5316（丸め値で計算）", close((1 + 0.4393) / 2, 0.7196, 1e-4) and close((1 + 0.0632) / 2, 0.5316, 1e-12), "")
check(T, "生の寄与率 0.7196/2 = 36.0%", close(lz[0] / lz.sum(), 0.360, 5e-4), r(lz[0] / lz.sum()))
Q = 2
lt = (Q / (Q - 1)) ** 2 * (lz[lz > 1 / Q] - 1 / Q) ** 2
check(T, "ベンゼクリ補正 0.1929, 0.0040 = γ_k²、補正後の第1軸 ≈98%", allclose(lt, [0.1929, 0.0040], 5e-5) and allclose(lt, gq[:2] ** 2, 1e-10)
      and close(lt[0] / lt.sum(), 0.98, 5e-3), (r(lt), r(lt[0] / lt.sum())))
_, _, _, sb, _ = ca_svd(Zi.T @ Zi)
check(T, "バート行列の固有値 = 指示行列の固有値の2乗", allclose(sb ** 2, lz ** 2, 1e-10), r(sb ** 2))
Qs = 3
Zr = np.column_stack([np.eye(J)[rng.integers(0, J, 50)] for J in (2, 3, 4)])
_, _, _, sr, _ = ca_svd(Zr)
check(T, "乱数の Q=3（J=9）でも全イナーシャ = (J−Q)/Q = 2", close((sr ** 2).sum(), (9 - Qs) / Qs, 1e-10), r((sr ** 2).sum()))
colc = Vtz[0] / np.sqrt(cz) * sz[0]
if colc[0] > 0:
    colc = -colc
check(T, "第1軸：20代とコーヒーが負、60代と緑茶が正（符号は連動して反転）", colc[0] < 0 and colc[3] < 0 and colc[2] > 0 and colc[5] > 0, r(colc, 3))

T = "goodness-of-fit-indices"
Sf = np.array([[1, .48, .42, .40], [.48, 1, .56, .40], [.42, .56, 1, .35], [.40, .40, .35, 1]])
nf, pf, dff = 200, 4, 2


def Ffit(th):
    Sig = np.outer(th[:4], th[:4]) + np.diag(np.exp(th[4:]))
    return np.linalg.slogdet(Sig)[1] + np.trace(Sf @ np.linalg.inv(Sig)) - np.linalg.slogdet(Sf)[1] - pf


of = optimize.minimize(Ffit, np.r_[[0.6] * 4, np.log([0.5] * 4)], method="BFGS", options={"gtol": 1e-12})
lf, psf = of.x[:4], np.exp(of.x[4:])
Sh = np.outer(lf, lf) + np.diag(psf)
chi = (nf - 1) * of.fun
check(T, "最尤推定 λ̂=(0.630,0.779,0.695,0.536)、ψ̂=(0.603,0.393,0.516,0.713)（SEM の項目と同じ）", allclose(np.abs(lf), [0.630, 0.779, 0.695, 0.536], 5e-4) and allclose(psf, [0.603, 0.393, 0.516, 0.713], 5e-4), (r(lf, 3), r(psf, 3)))
chi0 = -(nf - 1) * np.linalg.slogdet(Sf)[1]
check(T, "χ²=2.845、df=2、χ0²=199×(−log|S|)=187.105、df0=6、p=0.241", close(chi, 2.845, 5e-4) and dff == 10 - 8 and close(chi0, 187.105, 5e-4)
      and 4 * 3 // 2 == 6 and close(stats.chi2.sf(chi, dff), 0.241, 5e-4), (r(chi), r(chi0), r(stats.chi2.sf(chi, dff))))
rmsea = math.sqrt(max(chi - dff, 0) / (dff * (nf - 1)))
cfi = 1 - max(chi - dff, 0) / max(chi0 - 6, chi - dff, 0)
tli = (chi0 / 6 - chi / dff) / (chi0 / 6 - 1)
check(T, "RMSEA≈0.046、CFI=1−0.845/181.105≈0.995", close(rmsea, 0.046, 5e-4) and close(chi - dff, 0.845, 5e-4) and close(chi0 - 6, 181.105, 5e-4) and close(cfi, 0.995, 5e-4), (r(rmsea), r(cfi)))
check(T, "TLI=(31.184−1.423)/30.184≈0.986", close(chi0 / 6, 31.184, 5e-4) and close(chi / 2, 1.423, 5e-4) and close(chi0 / 6 - 1, 30.184, 5e-4) and close(tli, 0.986, 5e-4), r(tli))
Ag = np.linalg.inv(Sh) @ Sf
gfi = 1 - np.trace((Ag - np.eye(4)) @ (Ag - np.eye(4))) / np.trace(Ag @ Ag)
agfi = 1 - pf * (pf + 1) / (2 * dff) * (1 - gfi)
check(T, "GFI 0.993（0.99269）、AGFI=1−(20/4)(1−0.99269)≈0.963", close(gfi, 0.993, 5e-4) and close(gfi, 0.99269, 5e-5) and close(agfi, 0.963, 5e-4)
      and close(1 - 5 * (1 - 0.99269), 0.963, 5e-4), (r(gfi, 5), r(agfi)))
dsh = np.sqrt(np.diag(Sh))
res = Sf - Sh / np.outer(dsh, dsh)
srmr = math.sqrt(np.mean(res[np.tril_indices(4)] ** 2))
check(T, "SRMR 0.023（下三角10個、対角含む）", close(srmr, 0.023, 5e-4), r(srmr))
check(T, "目安：RMSEA≤0.05、CFI・TLI≥0.95、SRMR≤0.08、GFI≥0.9 をすべて満たす", rmsea <= 0.05 and cfi >= 0.95 and tli >= 0.95 and srmr <= 0.08 and gfi >= 0.9, "成立")
check(T, "χ²≤df なら RMSEA=0、CFI=1（max の働き）", math.sqrt(max(1.5 - 2, 0) / (2 * 199)) == 0 and 1 - max(1.5 - 2, 0) / max(chi0 - 6, 1.5 - 2, 0) == 1, "成立")

# ================================================================
ok_n = sum(results)
ng_n = len(results) - ok_n
print(f"\n合計: OK {ok_n} / NG {ng_n}")

# --- 重要度A への加筆: path-analysis ---
T = "path-analysis"
_R = np.array([[1, 0.3], [0.3, 1]]); _r = np.array([0.5, 0.4])
_p = np.linalg.solve(_R, _r)
check(T, "例3 p1=0.38/0.91≈0.4176, p2=0.25/0.91≈0.2747（正規方程式）", abs(_p[0] - 0.4176) < 5e-5 and abs(_p[1] - 0.2747) < 5e-5 and abs(_p[0] - 0.38 / 0.91) < 1e-12 and abs(_p[1] - 0.25 / 0.91) < 1e-12, np.round(_p, 5))
check(T, "例3 r1y = p1 + r12·p2 = 0.4176 + 0.0824 = 0.5", abs(0.3 * _p[1] - 0.0824) < 5e-5 and abs(_p[0] + 0.3 * _p[1] - 0.5) < 1e-12, round(0.3 * _p[1], 5))
_R2 = _p @ _r
check(T, "例3 R²≈0.3187, 誤差のパス係数≈0.8254", abs(_R2 - 0.3187) < 5e-5 and abs(math.sqrt(1 - _R2) - 0.8254) < 5e-5, (round(_R2, 5), round(math.sqrt(1 - _R2), 5)))
_C = np.array([[1, 0.3, 0.5], [0.3, 1, 0.4], [0.5, 0.4, 1]])
_Z = rng.multivariate_normal(np.zeros(3), _C, 400_000); _Z = (_Z - _Z.mean(0)) / _Z.std(0)
_b = np.linalg.lstsq(_Z[:, :2], _Z[:, 2], rcond=None)[0]
check(T, "例3 シミュ 標準化偏回帰係数（許容0.005）", np.all(np.abs(_b - _p) < 0.005), np.round(_b, 4))

# --- 統合候補からの論点の追記: multidimensional-scaling ---
_D = np.array([[0, 4, 5, 3], [4, 0, 3, 5], [5, 3, 0, 4], [3, 5, 4, 0]], float)
_J = np.eye(4) - 1 / 4; _B = -0.5 * _J @ (_D ** 2) @ _J
_ev = np.sort(np.linalg.eigvalsh(_B))[::-1]
check("multidimensional-scaling", "例1 tr B=Σ_{i<j}d²/n=100/4=25=16+9、固有値0を含む", abs(np.trace(_B) - 25) < 1e-12 and abs((_D ** 2).sum() / 2 - 100) < 1e-12 and np.allclose(_ev, [16, 9, 0, 0], atol=1e-10) and np.allclose(_B @ np.ones(4), 0), np.round(_ev, 6))
_Xc = np.linalg.eigh(_B)[1][:, ::-1][:, :2] * np.sqrt(_ev[:2])
check("multidimensional-scaling", "各次元の座標の2乗和＝固有値", np.allclose((_Xc ** 2).sum(0), _ev[:2]), (_Xc ** 2).sum(0))

ok_n = sum(results)
ng_n = len(results) - ok_n
print(f"\n合計（重要度A への加筆を含む）: OK {ok_n} / NG {ng_n}")
