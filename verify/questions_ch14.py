"""第14章（マルコフ連鎖）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import numpy as np
import sympy as sp
from math import gcd
from functools import reduce
from scipy.linalg import expm

q = Q(14)

P = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
pi0 = np.array([0.5, 0.3, 0.2])


def stationary(M):
    w, v = np.linalg.eig(M.T)
    s = np.real(v[:, np.argmin(abs(w - 1))])
    return s / s.sum()


def reach(M):
    A = (np.asarray(M) > 0).astype(int) | np.eye(len(M), dtype=int)
    R = A.copy()
    for _ in range(len(M)):
        R = ((R @ A) > 0).astype(int)
    return R


# --- マルコフ性：2段前と2段後が独立でない例 ---
P2s = np.array([[0.9, 0.1], [0.1, 0.9]])
joint = np.zeros((2, 2))  # X_{n-1}, X_{n+1}、X_{n-1} は一様
for a in range(2):
    for c in range(2):
        joint[a, c] = 0.5 * (P2s @ P2s)[a, c]
q.check("q14-markov-property-1", not np.allclose(joint, np.outer(joint.sum(1), joint.sum(0))),
        "条件をはずすと X_{n-1} と X_{n+1} は独立でない＝誤り")

# --- マルコフ連鎖 ---
q.num("q14-markov-chain-1", 0.5 * P[0, 1] * P[1, 1])
q.num("q14-markov-chain-2", (pi0 @ P)[2])

# --- 推移確率行列 ---
cands = [np.array([[0.2, 0.8], [0.5, 0.5]]), np.array([[0.2, 0.8], [0.6, 0.6]]),
         np.array([[0.3, 0.5], [0.7, 0.5]]), np.array([[1.2, -0.2], [0.4, 0.6]])]
valid = [bool((M >= 0).all() and np.allclose(M.sum(1), 1)) for M in cands]
q.check("q14-transition-probability-matrix-1", valid == [True, False, False, False], "1つ目だけが確率行列")
q.check("q14-transition-probability-matrix-2", not np.allclose(P.sum(0), 1) and np.allclose(P.sum(0), [0.8, 1.2, 1.0]),
        "列和は 0.8,1.2,1.0（誤り）")

# --- n 段階 ---
P2 = P @ P
q.num("q14-n-step-transition-probability-1", P2[1, 0])
q.num("q14-n-step-transition-probability-2", np.linalg.matrix_power(np.array([[0.7, 0.3], [0.1, 0.9]]), 5)[0, 0])

# --- チャップマン・コルモゴロフ ---
P2given = np.array([[0.33, 0.39, 0.28], [0.24, 0.48, 0.28], [0.17, 0.39, 0.44]])
q.check("q14-chapman-kolmogorov-equation-1", np.allclose(P2given, P2), "与えた P^(2) は P² と一致")
q.num("q14-chapman-kolmogorov-equation-1", (P2given @ P2given)[0, 1])
rng = np.random.default_rng(0)
okk = True
for _ in range(200):
    M = rng.random((4, 4)); M /= M.sum(1, keepdims=True)
    m, n = rng.integers(0, 4, 2)
    Pm, Pn = np.linalg.matrix_power(M, m), np.linalg.matrix_power(M, n)
    Pmn = Pm @ Pn
    okk &= bool((Pmn[:, None, :] >= Pm[:, :, None] * Pn[None, :, :] - 1e-12).all())
q.check("q14-chapman-kolmogorov-equation-2", okk, "ランダムな行列で不等式が成立（正しい）")

# --- 定常分布 ---
q.num("q14-stationary-distribution-1", stationary(np.array([[0.8, 0.2], [0.3, 0.7]]))[0])
s = stationary(P)
q.num("q14-stationary-distribution-2", s[1])
q.check("q14-stationary-distribution-2", np.allclose(s, np.array([5, 9, 7]) / 21), "π=(5,9,7)/21")
Pab = np.array([[1, 0, 0], [0.5, 0, 0.5], [0, 0, 1]])
okc = all(np.allclose(np.array([c, 0, 1 - c]) @ Pab, [c, 0, 1 - c]) for c in np.linspace(0, 1, 11))
q.check("q14-stationary-distribution-3", okc and not np.allclose(np.ones(3) / 3 @ Pab, np.ones(3) / 3),
        "(c,0,1-c) はすべて定常、(1/3,1/3,1/3) は定常でない")

# --- 既約性 ---
Ms = [np.array([[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]), np.array([[1, 0, 0], [0.3, 0.4, 0.3], [0, 0.5, 0.5]]),
      np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]]), np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1]])]
irr = [bool(reach(M).all()) for M in Ms]
q.check("q14-irreducibility-1", irr == [True, False, False, False], "1つ目だけが既約")
Pp = np.array([[0, 1], [1, 0]])
q.check("q14-irreducibility-2", reach(Pp).all() and not np.allclose(np.linalg.matrix_power(Pp, 100), np.linalg.matrix_power(Pp, 101)),
        "既約でも周期2で P^n は収束しない（誤り）")

# --- 類 ---
P4 = np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0.25, 0.25, 0.25, 0.25], [0, 0, 0, 1]])
Rr = reach(P4)
comm = Rr & Rr.T
classes = {tuple(np.nonzero(row)[0]) for row in comm}
closed = [c for c in classes if all(P4[i, j] == 0 for i in c for j in range(4) if j not in c)]
q.check("q14-communicating-class-1", len(classes) == 3 and len(closed) == 2, "類3個・閉じた類2個")

# --- 周期 ---
Pc = np.zeros((4, 4)); Pc[0, 1] = 1; Pc[1, 0] = 0.5; Pc[1, 2] = 0.5; Pc[2, 3] = 1; Pc[3, 0] = 1
ret = [n for n in range(1, 40) if np.linalg.matrix_power(Pc, n)[0, 0] > 0]
q.num("q14-periodicity-1", reduce(gcd, ret))
Pq = np.array([[0.3, 0.7, 0], [0, 0, 1], [1, 0, 0]])  # p11>0 で既約
ret2 = [reduce(gcd, [n for n in range(1, 40) if np.linalg.matrix_power(Pq, n)[i, i] > 0]) for i in range(3)]
q.check("q14-periodicity-2", reach(Pq).all() and ret2 == [1, 1, 1], "1状態で p_ii>0 なら全状態の周期1（正しい）")

# --- 再帰と過渡 ---
f = 0.6
q.num("q14-recurrence-and-transience-1", sum(f ** (k - 1) * (1 - f) * k for k in range(1, 400)))
q.check("q14-recurrence-and-transience-1", abs(1 / (1 - f) - 2.5) < 1e-12, "1/(1-f)=2.5")
# 有限連鎖：ランダムな部分確率行列（過渡のみ）なら行和<1 となり確率行列にならない → 少なくとも1つ再帰
okr = True
for _ in range(100):
    M = rng.random((5, 5)) * (rng.random((5, 5)) < 0.5); M[np.arange(5), rng.integers(0, 5, 5)] += 0.1
    M /= M.sum(1, keepdims=True)
    Mn = np.linalg.matrix_power(M, 200)
    okr &= bool((Mn.sum(0) > 1e-6).any() and np.allclose(Mn.sum(1), 1))
q.check("q14-recurrence-and-transience-2", okr, "有限の確率行列では確率が消えず、ある状態に残り続ける＝全部過渡はない（誤り）")

# --- 吸収 ---
Qm = np.array([[0, 0.5], [0.5, 0]])
N = np.linalg.inv(np.eye(2) - Qm)
q.num("q14-absorbing-state-1", N.sum(1)[0])
Nser = sum(np.linalg.matrix_power(Qm, k) for k in range(200))
q.check("q14-absorbing-state-2", np.allclose(N, Nser), "N = Σ Q^n（訪問回数の期待値）")

# --- エルゴード ---
q.num("q14-ergodicity-1", s @ np.array([10, 20, 30]))
q.check("q14-ergodicity-1", abs(s @ np.array([10, 20, 30]) - 440 / 21) < 1e-9, "440/21")
path = np.array([0, 1] * 500)
q.check("q14-ergodicity-2", abs((path == 0).mean() - 0.5) < 1e-12 and np.allclose(stationary(Pp.astype(float)), [0.5, 0.5]),
        "周期的でも時間平均は 1/2 に収束（誤り）")

# --- 詳細つり合い ---
q.check("q14-detailed-balance-1", abs(P[0, 1] * P[1, 2] * P[2, 0] - 0.006) < 1e-12
        and abs(P[0, 2] * P[2, 1] * P[1, 0] - 0.012) < 1e-12 and not np.isclose(s[0] * P[0, 1], s[1] * P[1, 0]),
        "閉路の積が異なり可逆でない（誤り）")
Pbd = np.array([[0.6, 0.4, 0], [0.3, 0.3, 0.4], [0, 0.6, 0.4]])
sb = stationary(Pbd)
q.num("q14-detailed-balance-2", sb[2])
q.check("q14-detailed-balance-2", abs(sb[2] - 8 / 29) < 1e-12 and np.isclose(sb[0] * 0.4, sb[1] * 0.3), "8/29、詳細つり合い成立")

# --- 極限分布 ---
q.num("q14-limiting-distribution-1", np.linalg.matrix_power(np.array([[0.5, 0.5], [0.25, 0.75]]), 200)[0, 1])

# --- 吸収確率・時間 ---
Q2 = np.array([[0, 0.6], [0.4, 0]]); R2 = np.array([[0.4, 0], [0, 0.6]])
N2 = np.linalg.inv(np.eye(2) - Q2)
q.num("q14-absorption-probability-1", (N2 @ R2)[0, 1])
q.check("q14-absorption-probability-1", abs((N2 @ R2)[0, 1] - 9 / 19) < 1e-12, "9/19")
q.num("q14-absorption-probability-2", N2.sum(1)[0])

# --- 初到達・再帰時間 ---
P2s2 = np.array([[0.8, 0.2], [0.3, 0.7]])
m1 = 1 + 0.2 * (1 / 0.3)
q.num("q14-first-passage-time-1", m1)
q.check("q14-first-passage-time-1", abs(m1 - 1 / stationary(P2s2)[0]) < 1e-12, "1/π1 と一致")

# --- 零再帰 ---
# 初めて0に戻る確率 f_{2k} = C(2k,k)/((2k-1)4^k)。和は1（再帰）、平均は発散（零再帰）。数値で確認
u = np.ones(3001)  # u_k = C(2k,k)/4^k を漸化式で
for kk in range(1, 3001):
    u[kk] = u[kk - 1] * (2 * kk - 1) / (2 * kk)
fk = lambda kk: u[kk] / (2 * kk - 1)
part = sum(fk(kk) for kk in range(1, 3000))
mean_part = [sum(2 * kk * fk(kk) for kk in range(1, K)) for K in (100, 1000, 3000)]
q.check("q14-positive-recurrence-1", part > 0.985 and mean_part[2] > mean_part[1] * 1.5 > mean_part[0] * 1.5,
        "戻る確率→1、平均再帰時間の部分和は発散（零再帰で定常分布なし＝誤り）")

# --- 連続時間 ---
G = np.array([[-2, 2], [3, -3]])
q.num("q14-continuous-time-markov-chain-1", expm(0.5 * G)[0, 0])

# --- 導出問題（追加） ---
def _deriv14():
    import numpy as _N
    import sympy as _S
    a, b = _S.symbols("a b", positive=True); n = _S.symbols("n", integer=True, nonnegative=True)
    pn = b / (a + b) + a / (a + b) * (1 - a - b) ** n
    ok = _S.simplify(pn.subs(n, 0) - 1) == 0
    ok &= _S.simplify(pn.subs(n, n + 1) - (b + (1 - a - b) * pn)) == 0
    P = _S.Matrix([[1 - a, a], [b, 1 - b]])
    for k in (1, 2, 5):
        ok &= _S.simplify((P ** k)[0, 0] - pn.subs(n, k)) == 0
    pi = _S.Matrix([[b / (a + b), a / (a + b)]])
    ok &= _S.simplify(pi * P - pi) == _S.zeros(1, 2)
    Pn = _N.linalg.matrix_power(_N.array([[0.7, 0.3], [0.1, 0.9]]), 200)
    ok &= abs(Pn[0, 0] - 0.1 / 0.4) < 1e-12
    q.check("q14-n-step-transition-probability-3", bool(ok), "sympy: 漸化式と初期値を満たし (P^k)_11 と一致、極限 b/(a+b) が定常分布")
_deriv14()

q.done()
