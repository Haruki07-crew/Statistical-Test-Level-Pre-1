"""第14章 検算スクリプト（マルコフ連鎖）。
行列計算は numpy（浮動小数）と fractions / sympy（厳密な分数）の両方で行う。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5 など）。
charts の params（推移確率行列 P など）は各 md の front matter を yaml で読み、本文の行列と照合する。
markov-chain 図の「図の下の定常分布」は src/charts_custom.js と同じ遅延べき乗法（(πP+π)/2 を2000回）で再現する。
シミュレーションは seed=20261023、許容誤差は各チェックに明記。
"""
from fractions import Fraction as F
from math import comb, sqrt, pi as PI
from pathlib import Path
import numpy as np
import sympy as sp
import yaml

ROOT = Path(__file__).resolve().parent.parent
results = []


def check(term, what, ok, value):
    results.append(bool(ok))
    print(f"[{'OK' if ok else 'NG'}] {term} / {what} / 値: {value}")


def close(x, claim, tol=5e-5):
    return abs(float(x) - claim) <= tol


def front(term):
    txt = (ROOT / "content" / "ch14" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term):
    return front(term).get("charts", [])


def fr(M):
    return sp.Matrix([[sp.Rational(str(x)) for x in row] for row in M])


def stationary(P):
    """厳密な定常分布（既約を仮定）。sympy で πP=π, Σπ=1。"""
    P = sp.Matrix(P)
    K = P.shape[0]
    A = (P.T - sp.eye(K)).col_join(sp.ones(1, K))
    b = sp.zeros(K, 1).col_join(sp.Matrix([1]))
    sol = A.gauss_jordan_solve(b)[0]
    return list(sol)


def js_stationary(P):
    P = np.array(P, float)
    K = len(P)
    st = np.full(K, 1 / K)
    for _ in range(2000):
        st = 0.5 * (st @ P) + 0.5 * st
    return st


def dist_path(P, start, steps):
    P = np.array(P, float)
    x = np.zeros(len(P)); x[start] = 1
    out = [x]
    for _ in range(steps):
        x = x @ P; out.append(x)
    return np.array(out)


rng = np.random.default_rng(20261023)

W = fr([[0.7, 0.3], [0.4, 0.6]])          # 天気
B3 = fr([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])  # ブランド
SUB = fr([[0.5, 0.3, 0.2], [0, 0.9, 0.1], [0, 0, 1]])        # 試用・有料・解約

# ================= markov-chain =================
T = "markov-chain"
c = charts(T)[0]
check(T, "図 params の P が例2と一致", fr(c["params"]["P"]) == B3, c["params"]["P"])
pi0 = sp.Matrix([[sp.Rational(1, 5), sp.Rational(4, 5)]])
p1 = pi0 * W; p2 = p1 * W
check(T, "例1 π(1)=(0.46,0.54)", list(p1) == [sp.Rational(46, 100), sp.Rational(54, 100)], list(p1))
check(T, "例1 π(2)=(0.538,0.462)", list(p2) == [sp.Rational(538, 1000), sp.Rational(462, 1000)], list(p2))
check(T, "例1 経路 晴晴雨 0.2×0.7×0.3=0.042", sp.Rational(1, 5) * W[0, 0] * W[0, 1] == sp.Rational(42, 1000), 0.042)
s0 = sp.Matrix([[sp.Rational(1, 2), sp.Rational(3, 10), sp.Rational(1, 5)]])
s1 = s0 * B3; s2 = s1 * B3
check(T, "例2 来月 (0.48,0.30,0.22)", list(s1) == [sp.Rational(48, 100), sp.Rational(30, 100), sp.Rational(22, 100)], list(s1))
check(T, "例2 翌月 (0.466,0.302,0.232)", list(s2) == [sp.Rational(466, 1000), sp.Rational(302, 1000), sp.Rational(232, 1000)], list(s2))
stB = stationary(B3)
check(T, "例2 定常分布 (7/16,5/16,4/16)", stB == [sp.Rational(7, 16), sp.Rational(5, 16), sp.Rational(1, 4)], stB)
for st in range(3):
    h = dist_path(c["params"]["P"], st, 200)[-1]
    check(T, f"図 出発 {st} から 200 ステップで定常分布へ収束", np.allclose(h, [0.4375, 0.3125, 0.25], atol=1e-10), h.round(6))
check(T, "図 図の下の定常分布（JS と同じ計算）=(0.4375,0.3125,0.25)", np.allclose(js_stationary(c["params"]["P"]), [0.4375, 0.3125, 0.25], atol=5e-5), js_stationary(c["params"]["P"]).round(5))

# ================= markov-property =================
T = "markov-property"
c = charts(T)[0]
check(T, "図 params の P = 天気の行列", fr(c["params"]["P"]) == W, c["params"]["P"])
check(T, "例1 p晴晴 p晴雨 = 0.21", W[0, 0] * W[0, 1] == sp.Rational(21, 100), W[0, 0] * W[0, 1])
# 例2：Y=(昨日,今日) の 4 状態連鎖を組み立て
rain = {("雨", "雨"): F(7, 10), ("晴", "雨"): F(1, 2), ("雨", "晴"): F(2, 5), ("晴", "晴"): F(1, 5)}
states = list(rain)
PY = [[F(0)] * 4 for _ in range(4)]
for a, (y, t) in enumerate(states):
    for b, (y2, t2) in enumerate(states):
        if y2 == t:
            PY[a][b] = rain[(y, t)] if t2 == "雨" else 1 - rain[(y, t)]
check(T, "例2 拡大した4状態の推移行列の行和=1", all(sum(r) == 1 for r in PY), [sum(r) for r in PY])
i_sr, i_rr = states.index(("晴", "雨")), states.index(("雨", "雨"))
v = PY[i_sr][i_rr] * PY[i_rr][i_rr]
check(T, "例2 (晴,雨)→(雨,雨)→(雨,雨) = 0.5×0.7 = 0.35", v == F(35, 100), v)
# X だけではマルコフでないこと（定常状態で P(X2=雨|X1=雨,X0) が X0 で異なる）
check(T, "例2 X_n 単独ではマルコフ性が成り立たない（0.7≠0.5）", rain[("雨", "雨")] != rain[("晴", "雨")], "0.7 vs 0.5")

# ================= transition-probability-matrix =================
T = "transition-probability-matrix"
c = charts(T)[0]
check(T, "図 params の P が例2と一致", fr(c["params"]["P"]) == SUB, c["params"]["P"])
counts = np.array([[42, 18], [16, 24]])
Ph = counts / counts.sum(1, keepdims=True)
check(T, "例1 MLE n_ij/n_i· = [[0.7,0.3],[0.4,0.6]]", np.allclose(Ph, [[0.7, 0.3], [0.4, 0.6]]), Ph.tolist())
check(T, "例1 組の数 100（101日）", counts.sum() == 100, counts.sum())
# MLE を sympy で（ラグランジュ）
p = sp.symbols("p", positive=True)
ll = 42 * sp.log(p) + 18 * sp.log(1 - p)
check(T, "MLE 行ごとの多項尤度の最大点 = 42/60", sp.solve(sp.diff(ll, p), p) == [sp.Rational(7, 10)], sp.solve(sp.diff(ll, p), p))
check(T, "例2 行和=1", all(sum(SUB.row(i)) == 1 for i in range(3)), [sum(SUB.row(i)) for i in range(3)])
check(T, "例2 p12^(2)=0.42", (SUB ** 2)[0, 1] == sp.Rational(42, 100), (SUB ** 2)[0, 1])
path = dist_path(c["params"]["P"], 0, 60)[:, 1]
k = int(np.argmax(path))
check(T, "図 試用から出発すると有料の確率がいったん上がって下がる", 0 < k < 60 and path[-1] < path[k], f"最大 n={k}, {path[k]:.4f}")
check(T, "図 解約の確率が1に近づく", dist_path(c["params"]["P"], 0, 300)[-1, 2] > 0.9999, dist_path(c["params"]["P"], 0, 300)[-1, 2])
a_, b_ = sp.symbols("a b")
P2 = sp.Matrix([[1 - a_, a_], [b_, 1 - b_]])
ev = set(P2.eigenvals().keys())
check(T, "2状態の固有値 {1, 1-a-b}", ev == {1, 1 - a_ - b_}, ev)
check(T, "確率行列の積も確率行列（P·Q·1=1, 例）", np.allclose(np.array(B3 * SUB, float).sum(1), 1), np.array(B3 * SUB, float).sum(1))
check(T, "P1=1（右固有ベクトル1）", B3 * sp.ones(3, 1) == sp.ones(3, 1), "OK")
# 他の固有値の絶対値≦1（ランダム確率行列）
okev = True
for _ in range(200):
    M = rng.random((4, 4)); M /= M.sum(1, keepdims=True)
    e = np.linalg.eigvals(M)
    okev &= np.isclose(np.max(np.abs(e)), 1) and np.any(np.isclose(e, 1))
check(T, "ランダム確率行列200個で固有値1を持ち |λ|≦1", okev, "200/200")

# ================= chapman-kolmogorov-equation =================
T = "chapman-kolmogorov-equation"
c = charts(T)[0]
check(T, "図 params の P が例2と一致", fr(c["params"]["P"]) == B3, c["params"]["P"])
W2, W3, W5 = W ** 2, W ** 3, W ** 5
check(T, "例1 P^2=[[0.61,0.39],[0.52,0.48]]", W2 == fr([[0.61, 0.39], [0.52, 0.48]]), W2.tolist())
check(T, "例1 P^3=[[0.583,0.417],[0.556,0.444]]", W3 == fr([[0.583, 0.417], [0.556, 0.444]]), W3.tolist())
check(T, "例1 p11^(5)=0.57247（P^2 P^3 で厳密）", W5[0, 0] == sp.Rational(57247, 100000) and (W2 * W3)[0, 0] == W5[0, 0], W5[0, 0])
check(T, "例1 項 0.35563, 0.21684", W2[0, 0] * W3[0, 0] == sp.Rational(35563, 100000) and W2[0, 1] * W3[1, 0] == sp.Rational(21684, 100000), "OK")
check(T, "例1 p12^(5)=0.42753（0.25437+0.17316）", W5[0, 1] == sp.Rational(42753, 100000) and W2[0, 0] * W3[0, 1] == sp.Rational(25437, 100000) and W2[0, 1] * W3[1, 1] == sp.Rational(17316, 100000), W5[0, 1])
f5 = sp.Rational(4, 7) + sp.Rational(3, 7) * sp.Rational(3, 10) ** 5
check(T, "例1 公式 4/7+3/7·0.3^5 = 0.5714+0.0010 = 0.5725", f5 == W5[0, 0] and close(sp.Rational(3, 7) * sp.Rational(3, 10) ** 5, 0.0010) and close(f5, 0.5725), float(f5))
B2 = B3 ** 2
check(T, "例2 P^2", B2 == fr([[0.67, 0.17, 0.16], [0.31, 0.53, 0.16], [0.19, 0.29, 0.52]]), B2.tolist())
check(T, "例2 p11^(3)=0.586（0.536+0.034+0.016）", (B3 ** 3)[0, 0] == sp.Rational(586, 1000) and (B3 * B2)[0, 0] == (B2 * B3)[0, 0], (B3 ** 3)[0, 0])
okck = all(B3 ** (m + n) == B3 ** m * B3 ** n for m in range(5) for n in range(5))
check(T, "P^(m+n)=P^m P^n（m,n=0..4、厳密）", okck, "25通り")
h = dist_path(c["params"]["P"], 0, 10)
re = dist_path(c["params"]["P"], 0, 4)[-1] @ np.linalg.matrix_power(np.array(c["params"]["P"]), 6)
check(T, "図 m=4 からやり直した値 = 10 ステップ目", np.allclose(re, h[-1]), re.round(5))

# ================= n-step-transition-probability =================
T = "n-step-transition-probability"
c = charts(T)[0]
check(T, "図 params の P = 天気の行列", fr(c["params"]["P"]) == W, c["params"]["P"])
h = dist_path(c["params"]["P"], 0, 3)[:, 0]
check(T, "図 晴から 0.7, 0.61, 0.583", np.allclose(h[1:], [0.7, 0.61, 0.583]), h[1:])
n = sp.symbols("n", integer=True, nonnegative=True)
Pi = sp.Matrix([[b_, a_], [b_, a_]]) / (a_ + b_)
D = sp.Matrix([[a_, -a_], [-b_, b_]]) / (a_ + b_)
lam = 1 - a_ - b_
check(T, "公式 P=Π+λD", sp.simplify(Pi + lam * D - P2) == sp.zeros(2), "OK")
check(T, "公式 Π²=Π, D²=D, ΠD=DΠ=O", all(sp.simplify(x) == y for x, y in [(Pi ** 2 - Pi, sp.zeros(2)), (D ** 2 - D, sp.zeros(2)), (Pi * D, sp.zeros(2)), (D * Pi, sp.zeros(2))]), "OK")
okf = True
for (aa, bb) in [(0.3, 0.4), (0.9, 0.8), (1, 1), (0.1, 0.05)]:
    Pn = np.array([[1 - aa, aa], [bb, 1 - bb]])
    for k in range(0, 8):
        fpn = np.array(Pi.subs({a_: aa, b_: bb}), float) + (1 - aa - bb) ** k * np.array(D.subs({a_: aa, b_: bb}), float)
        okf &= np.allclose(np.linalg.matrix_power(Pn, k), fpn)
check(T, "2状態の P^n 公式（4組×n=0..7）", okf, "OK")
check(T, "例1 4/7+3/7·0.027 = 0.5714+0.0116 = 0.583", close(4 / 7, 0.5714) and close(3 / 7 * 0.027, 0.0116) and W3[0, 0] == sp.Rational(4, 7) + sp.Rational(3, 7) * sp.Rational(27, 1000), float(W3[0, 0]))
check(T, "例1 n=10 で 0.5714", close((W ** 10)[0, 0], 0.5714), float((W ** 10)[0, 0]))
check(T, "例2 P^2 第1行 0.67,0.17,0.16", list(B2.row(0)) == [sp.Rational(67, 100), sp.Rational(17, 100), sp.Rational(16, 100)], list(B2.row(0)))
check(T, "例2 P^2 の行和=1", all(sum(B2.row(i)) == 1 for i in range(3)), "OK")
v = sp.Rational(1, 2) * W3[0, 0] + sp.Rational(1, 2) * W3[1, 0]
check(T, "例3 P(X3=晴)=0.5695", v == sp.Rational(5695, 10000), v)
check(T, "例3 差 0.583-0.556=0.027=0.3^3", W3[0, 0] - W3[1, 0] == sp.Rational(3, 10) ** 3, W3[0, 0] - W3[1, 0])
dd = [float((W ** k)[0, 0] - sp.Rational(4, 7)) for k in range(1, 6)]
check(T, "図 4/7 との差が毎回 0.3 倍", np.allclose(np.array(dd[1:]) / np.array(dd[:-1]), 0.3), dd)

# ================= stationary-distribution =================
T = "stationary-distribution"
cs = charts(T)
check(T, "図1 params の P が例2と一致", fr(cs[0]["params"]["P"]) == B3, cs[0]["params"]["P"])
check(T, "図2 params の P が例1と一致", fr(cs[1]["params"]["P"]) == W, cs[1]["params"]["P"])
stW = stationary(W)
check(T, "例1 π=(4/7,3/7)", stW == [sp.Rational(4, 7), sp.Rational(3, 7)], stW)
check(T, "例1 0.5714, 0.4286", close(4 / 7, 0.5714) and close(3 / 7, 0.4286), (4 / 7, 3 / 7))
check(T, "例1 検算 4/7×0.7+3/7×0.4=4/7", sp.Rational(4, 7) * W[0, 0] + sp.Rational(3, 7) * W[1, 0] == sp.Rational(4, 7), "OK")
check(T, "例2 π=(7/16,5/16,4/16)", stB == [sp.Rational(7, 16), sp.Rational(5, 16), sp.Rational(1, 4)], stB)
check(T, "例2 捨てた(2)で検算 5/16", sp.Rational(1, 10) * sp.Rational(7, 16) + sp.Rational(7, 10) * sp.Rational(5, 16) + sp.Rational(1, 5) * sp.Rational(1, 4) == sp.Rational(5, 16), "OK")
st2 = stationary(P2)
check(T, "2状態 π=(b/(a+b), a/(a+b))", [sp.simplify(x - y) for x, y in zip(st2, [b_ / (a_ + b_), a_ / (a_ + b_)])] == [0, 0], st2)
ev_, evec = np.linalg.eig(np.array(B3, float).T)
lv = np.real(evec[:, np.argmin(abs(ev_ - 1))]); lv /= lv.sum()
check(T, "例2 P^T の固有値1の固有ベクトル（numpy）", np.allclose(lv, [0.4375, 0.3125, 0.25]), lv.round(6))
check(T, "平均再帰時間 7/4=1.75 日（晴）", 1 / stW[0] == sp.Rational(7, 4), 1 / stW[0])
check(T, "図1 図の下の定常分布 (0.4375,0.3125,0.25)", np.allclose(js_stationary(cs[0]["params"]["P"]), [0.4375, 0.3125, 0.25], atol=5e-5), js_stationary(cs[0]["params"]["P"]).round(4))
dd = [dist_path(cs[1]["params"]["P"], 1, 6)[k, 0] - 4 / 7 for k in range(6)]
check(T, "図2 差が毎回 0.3 倍（雨から出発）", np.allclose(np.array(dd[1:]) / np.array(dd[:-1]), 0.3), np.round(dd, 5))
Pp = sp.Matrix([[0, 1], [1, 0]])
check(T, "周期2の例 π=(1/2,1/2) は定常、分布は振動", stationary(Pp) == [sp.Rational(1, 2)] * 2 and (Pp ** 2)[0, 0] == 1 and (Pp ** 3)[0, 0] == 0, "OK")
Abs2 = sp.Matrix([[1, 0, 0], [sp.Rational(1, 2), 0, sp.Rational(1, 2)], [0, 0, 1]])
check(T, "吸収状態2つ：(1,0,0) と (0,0,1) がどちらも定常", sp.Matrix([[1, 0, 0]]) * Abs2 == sp.Matrix([[1, 0, 0]]) and sp.Matrix([[0, 0, 1]]) * Abs2 == sp.Matrix([[0, 0, 1]]), "OK")

# ================= detailed-balance =================
T = "detailed-balance"
cs = charts(T)
E1 = fr([[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]])
MH = sp.Matrix([[0, sp.Rational(1, 2), sp.Rational(1, 2)], [sp.Rational(1, 3), sp.Rational(1, 6), sp.Rational(1, 2)], [sp.Rational(1, 5), sp.Rational(3, 10), sp.Rational(1, 2)]])
check(T, "図1 params の P が例3の行列（1/3,1/6 は4桁丸め）", np.allclose(np.array(cs[0]["params"]["P"]), np.array(MH, float), atol=5e-5), cs[0]["params"]["P"])
check(T, "図1 params の各行和=1（丸め込み）", np.allclose(np.array(cs[0]["params"]["P"]).sum(1), 1, atol=1e-12), np.array(cs[0]["params"]["P"]).sum(1))
check(T, "図2 params の P が例1と一致", fr(cs[1]["params"]["P"]) == E1, cs[1]["params"]["P"])


def db_holds(P, pi):
    K = P.shape[0]
    return all(pi[i] * P[i, j] == pi[j] * P[j, i] for i in range(K) for j in range(K))


pE1 = stationary(E1)
check(T, "例1 π=(0.25,0.5,0.25)", pE1 == [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)], pE1)
check(T, "例1 詳細つり合いが全組で成立", db_holds(E1, pE1), "OK")
check(T, "例2 詳細つり合い不成立", not db_holds(B3, stB), "NG（期待どおり）")
check(T, "例2 π1p12=0.04375, π2p21=0.0625", stB[0] * B3[0, 1] == sp.Rational(4375, 100000) and stB[1] * B3[1, 0] == sp.Rational(625, 10000), (stB[0] * B3[0, 1], stB[1] * B3[1, 0]))
c1 = B3[0, 1] * B3[1, 2] * B3[2, 0]; c2 = B3[0, 2] * B3[2, 1] * B3[1, 0]
check(T, "例2 コルモゴロフ 0.001 vs 0.004", c1 == sp.Rational(1, 1000) and c2 == sp.Rational(4, 1000), (c1, c2))
# 循環の向き：正味の流れ 1→3, 3→2, 2→1 が正
net = lambda i, j: stB[i] * B3[i, j] - stB[j] * B3[j, i]
check(T, "例2 1→3→2→1 向きの正味の流れが正", net(0, 2) > 0 and net(2, 1) > 0 and net(1, 0) > 0, (net(0, 2), net(2, 1), net(1, 0)))
# 例3 メトロポリス
tgt = [sp.Rational(1, 5), sp.Rational(3, 10), sp.Rational(1, 2)]
M = sp.zeros(3)
for i in range(3):
    for j in range(3):
        if i != j:
            M[i, j] = sp.Rational(1, 2) * sp.Min(1, tgt[j] / tgt[i])
    M[i, i] = 1 - sum(M[i, j] for j in range(3) if j != i)
check(T, "例3 メトロポリスの推移行列", M == MH, M.tolist())
check(T, "例3 詳細つり合い成立（0.1, 0.1, 0.15）", db_holds(M, tgt) and tgt[0] * M[0, 1] == sp.Rational(1, 10) and tgt[1] * M[1, 2] == sp.Rational(15, 100), "OK")
check(T, "例3 定常分布 = (0.2,0.3,0.5)", stationary(M) == tgt, stationary(M))
check(T, "図1 図の下の定常分布 ≈ (0.2,0.3,0.5)", np.allclose(js_stationary(cs[0]["params"]["P"]), [0.2, 0.3, 0.5], atol=5e-5), js_stationary(cs[0]["params"]["P"]).round(5))
check(T, "図2 図の下の定常分布 = (0.25,0.5,0.25)", np.allclose(js_stationary(cs[1]["params"]["P"]), [0.25, 0.5, 0.25], atol=5e-5), js_stationary(cs[1]["params"]["P"]).round(5))
# 一般証明（記号）：詳細つり合い ⇒ 定常
pis = sp.symbols("q1:4", positive=True)
okg = True
for _ in range(50):  # ランダム可逆連鎖（対称重み w_ij / π_i）で数値確認
    w = rng.random((4, 4)); w = w + w.T
    pi_ = rng.random(4); pi_ /= pi_.sum()
    Pm = w / w.sum(1, keepdims=True)
    piR = w.sum(1) / w.sum()
    okg &= np.allclose(piR @ Pm, piR) and np.allclose(piR[:, None] * Pm, (piR[:, None] * Pm).T)
check(T, "詳細つり合い ⇒ 定常（ランダム可逆連鎖50個）", okg, "50/50")
# MH 一般：q 非対称・ π 任意で詳細つり合い（記号）
pi_i, pi_j, q_ij, q_ji = sp.symbols("pi_i pi_j q_ij q_ji", positive=True)
r = (pi_j * q_ji) / (pi_i * q_ij)
lhs1 = pi_i * q_ij * 1; rhs1 = pi_j * q_ji * (1 / r)        # r>=1 の場合
lhs2 = pi_i * q_ij * r; rhs2 = pi_j * q_ji * 1             # r<1 の場合
check(T, "MH の受容確率 min(1,πj q_ji/(πi q_ij)) で詳細つり合い（記号）", sp.simplify(lhs1 - rhs1) == 0 and sp.simplify(lhs2 - rhs2) == 0, "OK")
# 例1：隣接型は可逆（ランダム出生死滅連鎖で）
okbd = True
for _ in range(50):
    K = 5; Pm = np.zeros((K, K))
    for i in range(K):
        up = rng.random() * 0.5 if i < K - 1 else 0; dn = rng.random() * 0.5 if i > 0 else 0
        if i < K - 1: Pm[i, i + 1] = up
        if i > 0: Pm[i, i - 1] = dn
        Pm[i, i] = 1 - up - dn
    e, V = np.linalg.eig(Pm.T); s = np.real(V[:, np.argmin(abs(e - 1))]); s /= s.sum()
    okbd &= np.allclose(s[:, None] * Pm, (s[:, None] * Pm).T)
check(T, "隣どうしにしか移らない連鎖は詳細つり合い（ランダム50個）", okbd, "50/50")

# ================= limiting-distribution =================
T = "limiting-distribution"
cs = charts(T)
check(T, "図1 params の P = 天気", fr(cs[0]["params"]["P"]) == W, cs[0]["params"]["P"])
check(T, "図2 params の P = [[0,1],[1,0]]", fr(cs[1]["params"]["P"]) == Pp, cs[1]["params"]["P"])
okl = all((W ** k)[0, 0] == sp.Rational(4, 7) + sp.Rational(3, 7) * sp.Rational(3, 10) ** k and (W ** k)[1, 0] == sp.Rational(4, 7) - sp.Rational(4, 7) * sp.Rational(3, 10) ** k for k in range(12))
check(T, "例1 p11^(n), p21^(n) の式（n=0..11）", okl, "OK")
check(T, "例1 n=5 で 0.5725, 0.5700, 差 0.0024", close(W5[0, 0], 0.5725) and close(W5[1, 0], 0.5700) and close(W5[0, 0] - W5[1, 0], 0.0024), (float(W5[0, 0]), float(W5[1, 0])))
R4 = sp.Matrix([[1, 0, 0, 0], [sp.Rational(3, 5), 0, sp.Rational(2, 5), 0], [0, sp.Rational(3, 5), 0, sp.Rational(2, 5)], [0, 0, 0, 1]])
Pn = np.linalg.matrix_power(np.array(R4, float), 400)
check(T, "例2 P^n 第2行 → (0.7895,0,0,0.2105)", np.allclose(Pn[1], [0.7895, 0, 0, 0.2105], atol=5e-5), Pn[1].round(5))
check(T, "例2 P^n 第3行 → (0.4737,0,0,0.5263)", np.allclose(Pn[2], [0.4737, 0, 0, 0.5263], atol=5e-5), Pn[2].round(5))
check(T, "図2 図の下の定常分布 (0.5,0.5)", np.allclose(js_stationary(cs[1]["params"]["P"]), [0.5, 0.5]), js_stationary(cs[1]["params"]["P"]))
h = dist_path(cs[1]["params"]["P"], 0, 15)[:, 0]
check(T, "図2 確率が 1 と 0 を往復", np.allclose(h, [1 - (k % 2) for k in range(16)]), h[:6])
ca = np.mean([np.linalg.matrix_power(np.array(Pp, float), k) for k in range(1, 1001)], axis=0)
check(T, "周期的でも時間平均 (1/n)Σp^(k) → π（n=1000）", np.allclose(ca, 0.5, atol=1e-3), ca.round(4).tolist())

# ================= ergodicity =================
T = "ergodicity"
cs = charts(T)
ld = cs[0]["params"]
xs = np.array(ld["x"]); ys = np.array(ld["series"][0]["y"], float)
check(T, "図2 params の P = 天気", fr(cs[1]["params"]["P"]) == W, cs[1]["params"]["P"])
check(T, "図1 定常確率の系列 = 4/7 の4桁丸め", all(close(y, 4 / 7) and round(4 / 7, 4) == y for y in ld["series"][1]["y"]), ld["series"][1]["y"][0])
cnt = ys * xs
check(T, "図1（乱数の一例）晴の日数 y·n が整数", np.allclose(cnt, np.round(cnt), atol=0.5 + 1e-9) and all(abs(y * n - round(y * n)) / n < 5e-5 + 1e-12 for y, n in zip(ys, xs)), np.round(cnt).astype(int).tolist())
inc = np.diff(np.round(cnt)); dx = np.diff(xs)
check(T, "図1（乱数の一例）累積日数が単調で増分≦日数の増分", np.all(inc >= 0) and np.all(inc <= dx), inc.astype(int).tolist())
# 漸近分散 σ²=π1π2(1+λ)/(1-λ)、|y-4/7| ≦ 4σ/√n（n≧50）
lam_ = 0.3; s2a = (4 / 7) * (3 / 7) * (1 + lam_) / (1 - lam_)
z = (ys - 4 / 7) / np.sqrt(s2a / xs)
check(T, "図1（乱数の一例）n≧50 で |z|≦4（漸近分散 0.4548）", np.all(np.abs(z[xs >= 50]) <= 4), np.round(z, 2).tolist())
check(T, "例2 表（0.300,0.560,0.570,0.575）と図の値が3桁で一致", [round(float(ys[list(xs).index(k)]), 3) for k in (10, 100, 1000, 10000)] == [0.3, 0.56, 0.57, 0.575], [ys[list(xs).index(k)] for k in (10, 100, 1000, 10000)])
# 例2 の雨から10日で晴3日の確率が極端でない
Pw = np.array(W, float)
dp = {0: np.array([0.0, 1.0])}
# 10日間の晴日数分布（雨から出発、X1..X10）
dist = {(1, 0): 1.0}
for _ in range(10):
    nd = {}
    for (s, c_), pr in dist.items():
        for t in (0, 1):
            key = (t, c_ + (t == 0)); nd[key] = nd.get(key, 0) + pr * Pw[s, t]
    dist = nd
pc3 = sum(pr for (s, c_), pr in dist.items() if c_ <= 3)
check(T, "図1（乱数の一例）n=10 で晴3日以下の確率が 1% 以上（妥当）", pc3 > 0.01, round(pc3, 4))
check(T, "例1 長期平均売上 58/7≈8.286", sp.Rational(10) * sp.Rational(4, 7) + 6 * sp.Rational(3, 7) == sp.Rational(58, 7) and close(58 / 7, 8.286, 5e-4), 58 / 7)
# シミュレーションでエルゴード定理（seed 固定、20万日。漸近SD 0.0015 に対し割合は許容 0.01、売上は 0.04）
N = 200000
u = rng.random(N); x = np.empty(N, int); s = 1
for k in range(N):
    s = 0 if u[k] < Pw[s, 0] else 1; x[k] = s
fs = np.where(x == 0, 10, 6).mean()
check(T, "シミュレーション 晴の割合 → 4/7（20万日、許容 0.01）", abs((x == 0).mean() - 4 / 7) < 0.01, (x == 0).mean())
check(T, "シミュレーション 平均売上 → 58/7（20万日、許容 0.04）", abs(fs - 58 / 7) < 0.04, fs)
check(T, "例3 周期2：時間平均 1/2、p11^(n) は 1,0,1,0", np.allclose(ca, 0.5, atol=1e-3) and [int((Pp ** k)[0, 0]) for k in range(1, 5)] == [0, 1, 0, 1], "p11^(1..4)=0,1,0,1")

# ================= periodicity =================
T = "periodicity"
cs = charts(T)
P1 = fr([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])
Qz = fr([[0.5, 0.5, 0], [0.25, 0.5, 0.25], [0, 0.5, 0.5]])
check(T, "図1 params の P が例1と一致", fr(cs[0]["params"]["P"]) == P1, cs[0]["params"]["P"])
check(T, "図2 params の P が例2 Q と一致", fr(cs[1]["params"]["P"]) == Qz, cs[1]["params"]["P"])
from math import gcd
from functools import reduce
ret = [k for k in range(1, 41) if (P1 ** k)[0, 0] > 0]
check(T, "例1 d(1)=gcd{2,4,6,...}=2", reduce(gcd, ret) == 2 and ret[:3] == [2, 4, 6], ret[:5])
check(T, "例1 P^2", P1 ** 2 == fr([[0.5, 0, 0.5], [0, 1, 0], [0.5, 0, 0.5]]), (P1 ** 2).tolist())
check(T, "例1 P^3 = P", P1 ** 3 == P1, "OK")
check(T, "例1 π=(0.25,0.5,0.25)", stationary(P1) == [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)], stationary(P1))
h = dist_path(cs[0]["params"]["P"], 0, 20)[:, 1]
check(T, "例1 状態2の確率 0,1,0,1…（出発点1）、平均 0.5", np.allclose(h, [k % 2 for k in range(21)]) and close(h[1:].mean(), 0.5), h[:6])
check(T, "図1 図の下の定常分布 (0.25,0.5,0.25)", np.allclose(js_stationary(cs[0]["params"]["P"]), [0.25, 0.5, 0.25], atol=5e-5), js_stationary(cs[0]["params"]["P"]).round(5))
check(T, "例2 Q=(I+P)/2", Qz == (sp.eye(3) + P1) / 2, "OK")
check(T, "例2 Q の定常分布も (0.25,0.5,0.25)", stationary(Qz) == [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)], stationary(Qz))
e0 = sp.Matrix([[1, 0, 0]])
q1, q2, q10 = e0 * Qz, e0 * Qz ** 2, e0 * Qz ** 10
check(T, "例2 (0.5,0.5,0) → (0.375,0.5,0.125)", list(q1) == [sp.Rational(1, 2), sp.Rational(1, 2), 0] and list(q2) == [sp.Rational(3, 8), sp.Rational(1, 2), sp.Rational(1, 8)], (list(q1), list(q2)))
check(T, "例2 n=10 で (0.2505,0.5,0.2495)", np.allclose(np.array(q10, float).ravel(), [0.2505, 0.5, 0.2495], atol=5e-5), np.array(q10, float).ravel().round(6))
check(T, "2状態の周期2の固有値 {1,-1}", set(Pp.eigenvals()) == {1, -1}, set(Pp.eigenvals()))
# 長さ2と3の閉路があれば周期1
C23 = fr([[0, 1, 0], [0.5, 0, 0.5], [1, 0, 0]])  # 1→2→1（長さ2）, 1→2→3→1（長さ3）
ret = [k for k in range(1, 30) if (C23 ** k)[0, 0] > 0]
check(T, "長さ2と3の閉路で周期1（最短は2）", reduce(gcd, ret) == 1 and ret[0] == 2, ret[:6])

# ================= irreducibility =================
T = "irreducibility"
cs = charts(T)
I1 = fr([[0, 1, 0], [0, 0, 1], [0.5, 0, 0.5]])
I2 = fr([[0.6, 0.4, 0, 0], [0.3, 0.7, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.2, 0.8]])
check(T, "図1 params の P が例1と一致", fr(cs[0]["params"]["P"]) == I1, cs[0]["params"]["P"])
check(T, "図2 params の P が例2と一致", fr(cs[1]["params"]["P"]) == I2, cs[1]["params"]["P"])


def reach(P):
    A = (np.array(P, float) > 0).astype(int); K = len(A)
    Rm = np.eye(K, dtype=int)
    for _ in range(K):
        Rm = ((Rm + Rm @ A) > 0).astype(int)
    return Rm


check(T, "例1 既約（全組で到達可能）", reach(I1).all(), reach(I1).tolist())
check(T, "例1 p13=0 だが p13^(2)=1", I1[0, 2] == 0 and (I1 ** 2)[0, 2] == 1, (I1 ** 2)[0, 2])
check(T, "例1 π=(0.25,0.25,0.5)", stationary(I1) == [sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 2)], stationary(I1))
h = dist_path(cs[0]["params"]["P"], 1, 400)[-1]
check(T, "図1 どこから出発しても (0.25,0.25,0.5) に収束", all(np.allclose(dist_path(cs[0]["params"]["P"], s, 400)[-1], [0.25, 0.25, 0.5], atol=1e-8) for s in range(3)), h.round(6))
check(T, "例2 既約でない（1→3 不可）", reach(I2)[0, 2] == 0, reach(I2).tolist())
wsym = sp.symbols("w")
pw = sp.Matrix([[3 * wsym / 7, 4 * wsym / 7, 2 * (1 - wsym) / 7, 5 * (1 - wsym) / 7]])
check(T, "例2 π_w が任意の w で定常", sp.simplify(pw * I2 - pw) == sp.zeros(1, 4) and sp.simplify(sum(pw) - 1) == 0, "OK")
check(T, "例2 状態1から (0.429,0.571,0,0)、状態3から (0,0,0.286,0.714)", np.allclose(dist_path(cs[1]["params"]["P"], 0, 300)[-1], [0.429, 0.571, 0, 0], atol=5e-4) and np.allclose(dist_path(cs[1]["params"]["P"], 2, 300)[-1], [0, 0, 0.286, 0.714], atol=5e-4), "OK")

# ================= communicating-class =================
T = "communicating-class"
cs = charts(T)
CC = fr([[0.5, 0.5, 0, 0], [0.6, 0.4, 0, 0], [0.25, 0, 0.5, 0.25], [0, 0, 0, 1]])
check(T, "図 params の P が例と一致", fr(cs[0]["params"]["P"]) == CC, cs[0]["params"]["P"])
Rm = reach(CC)
cls = sorted({tuple(int(j) + 1 for j in range(4) if Rm[i, j] and Rm[j, i]) for i in range(4)})
check(T, "例 類 {1,2},{3},{4}", cls == [(1, 2), (3,), (4,)], cls)
closed = {c_: all(np.array(CC, float)[i - 1, j] == 0 for i in c_ for j in range(4) if j + 1 not in c_) for c_ in cls}
check(T, "例 閉じた類は {1,2},{4}、{3} は閉じていない", closed == {(1, 2): True, (3,): False, (4,): True}, closed)
hh = sp.symbols("h"); hs = sp.solve(sp.Eq(hh, sp.Rational(1, 4) + sp.Rational(1, 2) * hh), hh)[0]
check(T, "例 h=0.5（{1,2}へ入る確率）", hs == sp.Rational(1, 2), hs)
lim = dist_path(cs[0]["params"]["P"], 2, 400)[-1]
check(T, "図 状態3から {1,2} 合計 0.5、4 に 0.5", close(lim[0] + lim[1], 0.5, 1e-8) and close(lim[3], 0.5, 1e-8), lim.round(5))
check(T, "例 {1,2} 内の定常分布 (0.6/1.1,0.5/1.1)≈(0.545,0.455)", stationary(CC[:2, :2]) == [sp.Rational(6, 11), sp.Rational(5, 11)] and np.allclose(lim[:2] / 0.5, [0.545, 0.455], atol=5e-4), (lim[:2] / 0.5).round(4))
h3 = dist_path(cs[0]["params"]["P"], 2, 6)[:, 2]
check(T, "図 状態3の確率が 0.5 倍ずつ減る", np.allclose(h3, 0.5 ** np.arange(7)), h3)

# ================= recurrence-and-transience =================
T = "recurrence-and-transience"
cs = charts(T)
check(T, "図2 params の P が例1と一致", fr(cs[1]["params"]["P"]) == SUB, cs[1]["params"]["P"])
check(T, "図1 random-walk params（p=0.5, 400 ステップ, 5 本）", cs[0]["params"] == {"p": 0.5, "steps": 400, "paths": 5, "start": 0}, cs[0]["params"])
Q = SUB[:2, :2]
Nm = (sp.eye(2) - Q).inv()
check(T, "例1 f11=p11=0.5、期待滞在 2 か月 = N_11", Nm[0, 0] == 2 and sum((Q ** k)[0, 0] for k in range(200)) - 2 < sp.Rational(1, 10 ** 50), Nm[0, 0])
check(T, "例1 有料 1/(1-0.9)=10 = N_22", Nm[1, 1] == 10, Nm[1, 1])
# 訪問回数の幾何分布：シミュレーション（試用、10万人、許容 0.02）
vis = rng.geometric(0.5, 100000)
check(T, "訪問回数の期待値 1/(1-f)=2（幾何分布、10万回、許容 0.02）", abs(vis.mean() - 2) < 0.02, vis.mean())
# ランダムウォーク
okst = all(abs(comb(2 * k, k) * 0.25 ** k * sqrt(PI * k) - 1) < 0.13 / k + 1e-12 for k in range(1, 200))
check(T, "例2 p=1/2: C(2n,n)/4^n ≈ 1/√(πn)（相対誤差 < 1/(8n)）", okst, comb(200, 100) * 0.25 ** 100 * sqrt(PI * 100))
def csum(x, N_):
    t, tot = 1.0, 0.0
    for k in range(N_):
        tot += t; t *= (2 * k + 1) * (2 * k + 2) / (k + 1) ** 2 * x
    return tot


ss = csum(0.24, 2000)
check(T, "例2 p=0.6: Σp00^(2n)=1/|p-q|=5、f00=1-1/5=0.8", close(ss, 5, 1e-8) and close(1 - 1 / ss, 0.8, 1e-8), ss)
# シミュレーションで f00（p=0.6、2万本、5000 ステップで打ち切り、許容 0.01）
M_ = 20000; ret0 = 0
steps = np.where(rng.random((M_, 3000)) < 0.6, 1, -1).cumsum(1)
ret0 = (steps == 0).any(1).mean()
check(T, "シミュレーション p=0.6 の原点回帰確率 ≈ 0.8（2万本、許容 0.01）", abs(ret0 - 0.8) < 0.01, ret0)
psum = [csum(0.25, N_) for N_ in (100, 400, 1600)]
check(T, "例2 p=1/2: 部分和が √N 的に発散（100→400→1600 でほぼ2倍ずつ）", psum[1] / psum[0] > 1.9 and psum[2] / psum[1] > 1.95, np.round(psum, 3))

# ================= first-passage-time =================
T = "first-passage-time"
cs = charts(T)
vals = cs[0]["params"]["values"]
exact = [0.4 * 0.6 ** (k - 1) for k in range(1, 11)]
check(T, "図 bar-data = 0.4×0.6^(n-1) の4桁丸め", all(close(v_, e_) for v_, e_ in zip(vals, exact)), [round(e_, 4) for e_ in exact])
check(T, "例1 m21=2.5（幾何分布の平均、1+0.6m）", sp.solve(sp.Eq(hh, 1 + sp.Rational(3, 5) * hh), hh)[0] == sp.Rational(5, 2), 2.5)
check(T, "例1 3日以内 1-0.6^3=0.784", close(1 - 0.6 ** 3, 0.784, 1e-12), 1 - 0.6 ** 3)
check(T, "例1 m12=1/0.3≈3.333", close(1 / 0.3, 3.333, 5e-4), 1 / 0.3)
check(T, "例1 m1=1+0.3×2.5=7/4=1/π1", 1 + sp.Rational(3, 10) * sp.Rational(5, 2) == sp.Rational(7, 4) == 1 / stW[0], "7/4")
check(T, "例1 m2=1+0.4×10/3=7/3=1/π2", 1 + sp.Rational(2, 5) * sp.Rational(10, 3) == sp.Rational(7, 3) == 1 / stW[1], "7/3")
check(T, "よくある間違い 1/0.7≈1.43", close(1 / 0.7, 1.43, 5e-3), 1 / 0.7)
m2, m3 = sp.symbols("m2 m3")
sol = sp.solve([sp.Eq(m2, 1 + sp.Rational(7, 10) * m2 + sp.Rational(1, 10) * m3), sp.Eq(m3, 1 + sp.Rational(1, 5) * m2 + sp.Rational(7, 10) * m3)], [m2, m3])
check(T, "例2 m2=40/7≈5.714, m3=50/7≈7.143", sol[m2] == sp.Rational(40, 7) and sol[m3] == sp.Rational(50, 7) and close(40 / 7, 5.714, 5e-4) and close(50 / 7, 7.143, 5e-4), sol)
check(T, "例2 途中式 0.07m2=0.4", sp.Rational(9, 100) * sol[m2] == sp.Rational(4, 10) + sp.Rational(2, 100) * sol[m2], "OK")
m1 = 1 + sp.Rational(1, 10) * sol[m2] + sp.Rational(1, 10) * sol[m3]
check(T, "例2 m1=16/7≈2.286=1/π1", m1 == sp.Rational(16, 7) == 1 / stB[0] and close(16 / 7, 2.286, 5e-4), m1)
# 一般式：全ペアで m_ij を解き、m_jj=1/π_j（ブランド）
Bf = np.array(B3, float); okm = True
for j in range(3):
    idx = [k for k in range(3) if k != j]
    mm = np.linalg.solve(np.eye(2) - Bf[np.ix_(idx, idx)], np.ones(2))
    mjj = 1 + Bf[j, idx] @ mm
    okm &= np.isclose(mjj, 1 / float(stB[j]))
check(T, "全状態で m_j=1/π_j（ブランド、16/7, 16/5, 4）", okm, [16 / 7, 16 / 5, 4])
# シミュレーション：A の再帰時間の平均（seed 固定、5万回、許容 0.05）
rt = []
cum = Bf.cumsum(1)
for _ in range(50000):
    s, t = 0, 0
    while True:
        s = int(np.searchsorted(cum[s], rng.random())); t += 1
        if s == 0: break
    rt.append(t)
check(T, "シミュレーション A の平均再帰時間 ≈ 16/7（5万回、許容 0.05）", abs(np.mean(rt) - 16 / 7) < 0.05, np.mean(rt))

# ================= absorbing-state =================
T = "absorbing-state"
cs = charts(T)
check(T, "図1 params の P が例1と一致", fr(cs[0]["params"]["P"]) == SUB, cs[0]["params"]["P"])
check(T, "図2 random-walk params（p=0.4, 壁 0 と 5, 出発 2）", cs[1]["params"]["lower"] == 0 and cs[1]["params"]["upper"] == 5 and cs[1]["params"]["p"] == 0.4, cs[1]["params"])
t1, t2 = sp.symbols("t1 t2")
sol = sp.solve([sp.Eq(t1, 1 + sp.Rational(1, 2) * t1 + sp.Rational(3, 10) * t2), sp.Eq(t2, 1 + sp.Rational(9, 10) * t2)], [t1, t2])
check(T, "例1 t2=10, t1=8", sol[t1] == 8 and sol[t2] == 10, sol)
check(T, "例1 I-Q, det=0.05", (sp.eye(2) - Q) == fr([[0.5, -0.3], [0, 0.1]]) and (sp.eye(2) - Q).det() == sp.Rational(1, 20), (sp.eye(2) - Q).det())
check(T, "例1 N=[[2,6],[0,10]]", Nm == sp.Matrix([[2, 6], [0, 10]]), Nm.tolist())
check(T, "例1 N1 = (8,10)", Nm * sp.ones(2, 1) == sp.Matrix([8, 10]), list(Nm * sp.ones(2, 1)))
check(T, "例1 B=NR=(1,1)（解約のみ）", Nm * SUB[:2, 2:] == sp.ones(2, 1), list(Nm * SUB[:2, 2:]))
Nser = sum((np.linalg.matrix_power(np.array(Q, float), k) for k in range(500)), np.zeros((2, 2)))
check(T, "N = I+Q+Q²+…（500項）", np.allclose(Nser, np.array(Nm, float)), Nser.round(6).tolist())
hh2 = sp.solve(sp.Eq(hh, sp.Rational(3, 10) + sp.Rational(1, 2) * hh), hh)[0]
check(T, "例2 承認 h=0.6、却下 0.4、期間 t=2", hh2 == sp.Rational(3, 5) and sp.solve(sp.Eq(hh, sp.Rational(2, 10) + sp.Rational(1, 2) * hh), hh)[0] == sp.Rational(2, 5) and sp.solve(sp.Eq(hh, 1 + sp.Rational(1, 2) * hh), hh)[0] == 2, "0.6, 0.4, 2")
# シミュレーション：試用からの吸収時間（seed 固定、5万人、許容 0.15）
Sf = np.array(SUB, float); cumS = Sf.cumsum(1); tt = []
for _ in range(50000):
    s, t = 0, 0
    while s != 2:
        s = int(np.searchsorted(cumS[s], rng.random())); t += 1
    tt.append(t)
check(T, "シミュレーション 試用からの平均吸収時間 ≈ 8（5万人、許容 0.15）", abs(np.mean(tt) - 8) < 0.15, np.mean(tt))

# ================= absorption-probability =================
T = "absorption-probability"
cs = charts(T)
pr_ = cs[0]["params"]
check(T, "図 random-walk params（p=0.4, 出発 1, 壁 0 と 3）", pr_["p"] == 0.4 and pr_["start"] == 1 and pr_["lower"] == 0 and pr_["upper"] == 3, pr_)
h1, h2 = sp.symbols("h1 h2")
sol = sp.solve([sp.Eq(h1, sp.Rational(2, 5) * h2), sp.Eq(h2, sp.Rational(3, 5) * h1 + sp.Rational(2, 5))], [h1, h2])
check(T, "(1) h2=10/19≈0.5263, h1=4/19≈0.2105", sol[h2] == sp.Rational(10, 19) and sol[h1] == sp.Rational(4, 19) and close(10 / 19, 0.5263) and close(4 / 19, 0.2105), sol)
check(T, "(1) 破産確率 15/19≈0.7895", close(15 / 19, 0.7895), 15 / 19)
check(T, "(1) 一般解 (1-1.5)/(1-1.5^3)=-0.5/-2.375=0.2105", close((1 - 1.5) / (1 - 1.5 ** 3), 0.2105) and 1.5 ** 3 - 1 == 2.375, (1 - 1.5) / (1 - 1.5 ** 3))
check(T, "キャプション 4/19≈0.21", close(4 / 19, 0.21, 5e-3), 4 / 19)
sol = sp.solve([sp.Eq(t1, 1 + sp.Rational(2, 5) * t2), sp.Eq(t2, 1 + sp.Rational(3, 5) * t1)], [t1, t2])
check(T, "(2) t1=35/19≈1.8421, t2=40/19≈2.1053", sol[t1] == sp.Rational(35, 19) and sol[t2] == sp.Rational(40, 19) and close(35 / 19, 1.8421) and close(40 / 19, 2.1053), sol)
Qr = sp.Matrix([[0, sp.Rational(2, 5)], [sp.Rational(3, 5), 0]]); Rr = sp.Matrix([[sp.Rational(3, 5), 0], [0, sp.Rational(2, 5)]])
Nr = (sp.eye(2) - Qr).inv()
check(T, "(3) det(I-Q)=0.76", (sp.eye(2) - Qr).det() == sp.Rational(76, 100), (sp.eye(2) - Qr).det())
check(T, "(3) N≈[[1.3158,0.5263],[0.7895,1.3158]]", np.allclose(np.array(Nr, float), [[1.3158, 0.5263], [0.7895, 1.3158]], atol=5e-5), np.array(Nr, float).round(5).tolist())
check(T, "(3) N1=(1.8421,2.1053)", Nr * sp.ones(2, 1) == sp.Matrix([sp.Rational(35, 19), sp.Rational(40, 19)]), list(Nr * sp.ones(2, 1)))
BB = Nr * Rr
check(T, "(3) B=NR=(1/0.76)[[0.6,0.16],[0.36,0.4]]", BB * sp.Rational(76, 100) == fr([[0.6, 0.16], [0.36, 0.4]]) and np.allclose(np.array(BB, float), [[0.7895, 0.2105], [0.4737, 0.5263]], atol=5e-5), np.array(BB, float).round(4).tolist())
check(T, "(3) B の行和=1、第2列=(h1,h2)", BB * sp.ones(2, 1) == sp.ones(2, 1) and list(BB.col(1)) == [sp.Rational(4, 19), sp.Rational(10, 19)], "OK")
# 一般解（記号）が一次方程式を満たす
# r=q/p とおくと p=1/(1+r), q=r/(1+r)（sympy の簡約を通すため r で表す）
rr, ii, KK = sp.symbols("r i K", positive=True)
pp = 1 / (1 + rr)
hgen = lambda i: (1 - rr ** i) / (1 - rr ** KK)
resid = sp.simplify(sp.powsimp(sp.expand(pp * hgen(ii + 1) + (1 - pp) * hgen(ii - 1) - hgen(ii)), force=True))
check(T, "一般解が h_i=p h_{i+1}+q h_{i-1} と境界条件を満たす（記号、r=q/p）", resid == 0 and sp.simplify(hgen(0)) == 0 and sp.simplify(hgen(KK) - 1) == 0, resid)


def ruin(p, K):
    Qm = np.zeros((K - 1, K - 1)); Rm = np.zeros((K - 1, 2))
    for i in range(1, K):
        if i + 1 < K: Qm[i - 1, i] = p
        else: Rm[i - 1, 1] = p
        if i - 1 > 0: Qm[i - 1, i - 2] = 1 - p
        else: Rm[i - 1, 0] = 1 - p
    Nn = np.linalg.inv(np.eye(K - 1) - Qm)
    return Nn @ Rm, Nn @ np.ones(K - 1)


okr = True
for p_ in (0.3, 0.4, 0.5, 0.7):
    for K_ in (3, 5, 8):
        Bm, tm = ruin(p_, K_)
        for i in range(1, K_):
            th = i / K_ if p_ == 0.5 else (1 - ((1 - p_) / p_) ** i) / (1 - ((1 - p_) / p_) ** K_)
            okr &= np.isclose(Bm[i - 1, 1], th)
            if p_ == 0.5:
                okr &= np.isclose(tm[i - 1], i * (K_ - i))
check(T, "破産公式と対称時の平均終了時間 i(K-i)（NR・N1 と照合）", okr, "12組")
check(T, "p=0.5, K=3 で所持金1から 1/3", np.isclose(ruin(0.5, 3)[0][0, 1], 1 / 3), ruin(0.5, 3)[0][0, 1])
# シミュレーション（seed 固定、10万本、許容 0.006 ≈ SE 0.0013 の約4.5倍）
M_ = 100000; pos_ = np.ones(M_, int); alive = np.ones(M_, bool); tcount = np.zeros(M_, int)
while alive.any():
    stp = np.where(rng.random(alive.sum()) < 0.4, 1, -1)
    pos_[alive] += stp; tcount[alive] += 1
    alive = (pos_ > 0) & (pos_ < 3)
check(T, "シミュレーション 上の壁到達率 ≈ 4/19（10万本、許容 0.006）", abs((pos_ == 3).mean() - 4 / 19) < 0.006, (pos_ == 3).mean())
check(T, "シミュレーション 平均終了回数 ≈ 35/19（10万本、許容 0.02）", abs(tcount.mean() - 35 / 19) < 0.02, tcount.mean())

# ================= 重要度Cの用語 =================
print("\n--- 重要度Cの用語 ---")
from scipy.linalg import expm as _expm
rngC = np.random.default_rng(20261023)

T = "positive-recurrence"
pp, qq = .4, .6
pin = [(1 / 3) * (2 / 3) ** n for n in range(200)]
check(T, "例 π_n=(1/3)(2/3)^n: π0=0.3333, π1=0.2222, π2=0.1481, Σ=1", close(pin[0], .3333) and close(pin[1], .2222) and close(pin[2], .1481) and close(sum(pin), 1, 1e-12), pin[:3])
check(T, "釣り合い π_n p=π_{n+1} q と 0 での釣り合い π0=qπ0+qπ1", all(abs(pin[n] * pp - pin[n + 1] * qq) < 1e-15 for n in range(50)) and abs(pin[0] - (qq * pin[0] + qq * pin[1])) < 1e-15, "")
# 有限打ち切りの推移行列で定常分布
K = 200
Pm = np.zeros((K, K))
for i in range(K):
    Pm[i, min(i + 1, K - 1)] += pp
    Pm[i, max(i - 1, 0)] += qq
w, v = np.linalg.eig(Pm.T)
st = np.real(v[:, np.argmin(abs(w - 1))]); st = st / st.sum()
check(T, "打ち切り行列（200状態）の定常分布が π_n と一致", np.allclose(st[:10], pin[:10], atol=1e-10), st[:3])
nret = 100000
x = np.zeros(nret, int); tt = np.zeros(nret, int); alive = np.ones(nret, bool)
while alive.any():
    idx = np.where(alive)[0]
    step = np.where(rngC.random(idx.size) < pp, 1, -1)
    x[idx] = np.maximum(x[idx] + step, 0)
    tt[idx] += 1
    alive[idx] = x[idx] != 0
mret = tt.mean()
check(T, "sim 状態0の平均再帰時間 ≈ 3（10万回, 許容 0.05）", abs(mret - 3) < .05, mret)
check(T, "m0=1/π0=3", close(1 / pin[0], 3, 1e-12), 3)

T = "continuous-time-markov-chain"
la, mu = 0.2, 1.0
Q = np.array([[-la, la], [mu, -mu]])
check(T, "Q の各行の和 0", np.allclose(Q.sum(1), 0), Q.sum(1))
pi0 = mu / (la + mu)
check(T, "π0=0.8333, π1=0.1667, πQ=0", close(pi0, .8333) and close(1 - pi0, .1667) and np.allclose(np.array([pi0, 1 - pi0]) @ Q, 0), pi0)
P1 = _expm(Q)
check(T, "e^{-1.2}=0.3012, p00(1)=0.8835（式と expm(Q) の両方）", close(np.exp(-1.2), .3012) and close(pi0 + la / (la + mu) * np.exp(-1.2), .8835) and close(P1[0, 0], .8835), (np.exp(-1.2), P1[0, 0]))
ts, l_, m_ = sp.symbols("t lambda mu", positive=True)
Qs = sp.Matrix([[-l_, l_], [m_, -m_]])
p00 = m_ / (l_ + m_) + l_ / (l_ + m_) * sp.exp(-(l_ + m_) * ts)
Pt = sp.Matrix([[p00, 1 - p00], [1 - (l_ / (l_ + m_) + m_ / (l_ + m_) * sp.exp(-(l_ + m_) * ts)), l_ / (l_ + m_) + m_ / (l_ + m_) * sp.exp(-(l_ + m_) * ts)]])
check(T, "sympy P'(t)=P(t)Q=QP(t), P(0)=I", sp.simplify(Pt.diff(ts) - Pt * Qs) == sp.zeros(2) and sp.simplify(Pt.diff(ts) - Qs * Pt) == sp.zeros(2) and sp.simplify(Pt.subs(ts, 0)) == sp.eye(2), "一致")
check(T, "P(s+t)=P(s)P(t)（s=0.7, t=1.3）", np.allclose(_expm(2.0 * Q), _expm(.7 * Q) @ _expm(1.3 * Q)), "")
check(T, "t 大で p00→0.8333", close(_expm(50 * Q)[0, 0], pi0, 1e-10), _expm(50 * Q)[0, 0])
# sim: 滞在時間が指数分布の交代過程で時刻1の状態
ns = 200000
state = np.zeros(ns, int); tcur = np.zeros(ns)
alive = np.ones(ns, bool)
while alive.any():
    idx = np.where(alive)[0]
    rate = np.where(state[idx] == 0, la, mu)
    tcur[idx] += rngC.exponential(1 / rate)
    done = tcur[idx] > 1
    alive[idx[done]] = False
    state[idx[~done]] = 1 - state[idx[~done]]
p00s = (state == 0).mean()
check(T, "sim p00(1)≈0.8835（20万本, 許容 0.004）", abs(p00s - .8835) < .004, p00s)

# --- 統合候補からの論点の追記: recurrence-and-transience ---
_Pr = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]])
_w, _v = np.linalg.eig(_Pr.T)
_pi = np.real(_v[:, np.argmin(abs(_w - 1))]); _pi = _pi / _pi.sum()
_mret = []
for _i in range(3):
    _o = [j for j in range(3) if j != _i]
    _h = np.linalg.solve(np.eye(2) - _Pr[np.ix_(_o, _o)], np.ones(2))  # 他の状態から i に初めて着く平均時間
    _mret.append(1 + _Pr[_i, _o] @ _h)
check("recurrence-and-transience", "既約・正再帰の連鎖で π_i=1/m_i（3状態の例、平均再帰時間を連立方程式で計算）", np.allclose(_pi, 1 / np.array(_mret)), (_pi, _mret))

# ================= まとめ =================
ok = sum(results); ng = len(results) - ok
print(f"\n合計: OK {ok} / NG {ng}")
