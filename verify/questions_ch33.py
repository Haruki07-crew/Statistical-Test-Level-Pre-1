"""第33章（統計的因果推論）の問題の検算。"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qcheck import Q

import itertools
import numpy as np

q = Q(33)
rng = np.random.default_rng(33)


# --- d分離の判定（道を列挙して遮断を調べる） ---
def dsep(edges, x, y, Z):
    Z = set(Z)
    nodes = {a for e in edges for a in e}
    ch = {v: {b for a, b in edges if a == v} for v in nodes}
    pa = {v: {a for a, b in edges if b == v} for v in nodes}
    def desc(v):
        out, st = set(), [v]
        while st:
            u = st.pop()
            for w in ch[u]:
                if w not in out:
                    out.add(w); st.append(w)
        return out
    def paths(cur, path):
        if cur == y:
            yield path; return
        for nb in ch[cur] | pa[cur]:
            if nb not in path:
                yield from paths(nb, path + [nb])
    for p in paths(x, [x]):
        blocked = False
        for i in range(1, len(p) - 1):
            a, m, b = p[i - 1], p[i], p[i + 1]
            collider = (a in pa[m]) and (b in pa[m])
            if collider:
                if m not in Z and not (desc(m) & Z):
                    blocked = True; break
            elif m in Z:
                blocked = True; break
        if not blocked:
            return False
    return True


# --- 因果効果 ---
q.concept("q33-causal-effect-1", "因果推論の根本問題（数値なし）")
Y1 = np.array([5, 4, 6, 3.0]); Y0 = np.array([3, 4, 2, 3.0]); T = np.array([1, 0, 1, 0])
q.num("q33-causal-effect-2", np.mean(Y1 - Y0))
naive = Y1[T == 1].mean() - Y0[T == 0].mean()
att = np.mean((Y1 - Y0)[T == 1])
sb = Y0[T == 1].mean() - Y0[T == 0].mean()
q.num("q33-causal-effect-3", sb)
q.check("q33-causal-effect-3", np.isclose(naive, att + sb) and np.isclose(naive, 2.0) and np.isclose(att, 3.0), "群間差 2 = ATT 3 + 選択バイアス -1")

# --- バックドア ---
pY = {(1, 0): 0.3, (1, 1): 0.6, (0, 0): 0.2, (0, 1): 0.5}
pz = 0.4
do = lambda x: (1 - pz) * pY[(x, 0)] + pz * pY[(x, 1)]
q.num("q33-backdoor-criterion-1", do(1) - do(0))
pzx = {1: 0.8, 0: 0.2}; px1 = 1 / 3
obs = lambda x: (1 - pzx[x]) * pY[(x, 0)] + pzx[x] * pY[(x, 1)]
q.num("q33-backdoor-criterion-2", obs(1) - obs(0))
# 同時分布を作って、シミュレーションでも do と条件付けの違いを確認
n = 1_000_000
X = (rng.uniform(size=n) < px1).astype(int)
Zs = (rng.uniform(size=n) < np.where(X == 1, 0.8, 0.2)).astype(int)
q.check("q33-backdoor-criterion-2", abs(Zs.mean() - 0.4) < 0.002 and abs(px1 * 0.8 + (1 - px1) * 0.2 - 0.4) < 1e-12, "P(Z=1)=0.4 と整合")

E3 = [("Z", "X"), ("Z", "Y"), ("X", "M"), ("M", "Y"), ("X", "C"), ("Y", "C")]
def backdoor_ok(edges, x, y, Zset):
    # 条件1：Z に X の子孫がない／条件2：X から出る矢印を除いたグラフで X と Y が Z で d分離
    nodes = {a for e in edges for a in e}
    ch = {v: {b for a, b in edges if a == v} for v in nodes}
    des, st = set(), [x]
    while st:
        u = st.pop()
        for w in ch[u]:
            if w not in des:
                des.add(w); st.append(w)
    if set(Zset) & des:
        return False
    return dsep([(a, b) for a, b in edges if a != x], x, y, Zset)
res = [backdoor_ok(E3, "X", "Y", s) for s in [{"Z"}, {"Z", "M"}, {"Z", "C"}, set()]]
q.check("q33-backdoor-criterion-3", res == [True, False, False, False], "バックドア基準を満たすのは {Z} だけ")

# --- d分離 ---
E1 = [("X", "M"), ("Y", "M")]
q.check("q33-d-separation-1", not dsep(E1, "X", "Y", {"M"}) and dsep(E1, "X", "Y", set()), "合流点で条件付けると d連結（誤り）")
E2 = [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C"), ("C", "E")]
res = [dsep(E2, "A", "C", {"B", "D"}), dsep(E2, "A", "C", {"B"}), dsep(E2, "B", "D", {"C"}), dsep(E2, "B", "D", {"E"})]
q.check("q33-d-separation-2", res == [True, False, False, False], "正しいのは1つめだけ")
S = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 3.0]])  # (X, Y, M)
P = np.linalg.inv(S)
pc = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
q.num("q33-d-separation-3", pc)
Xs, Ys = rng.normal(size=n), rng.normal(size=n); Ms = Xs + Ys + rng.normal(size=n)
rx = Xs - np.polyval(np.polyfit(Ms, Xs, 1), Ms); ry = Ys - np.polyval(np.polyfit(Ms, Ys, 1), Ms)
q.check("q33-d-separation-3", abs(np.corrcoef(rx, ry)[0, 1] + 0.5) < 0.003, "シミュレーションの偏相関 ≈ -0.5")

# --- フロントドア ---
pM = {1: 0.8, 0: 0.2}
pYxm = {(0, 0): 0.1, (0, 1): 0.5, (1, 0): 0.2, (1, 1): 0.6}
doM = {m: 0.5 * pYxm[(0, m)] + 0.5 * pYxm[(1, m)] for m in (0, 1)}
fd = lambda x: (1 - pM[x]) * doM[0] + pM[x] * doM[1]
q.num("q33-frontdoor-criterion-1", fd(1))
q.check("q33-frontdoor-criterion-1", np.isclose(fd(0), 0.23) and np.isclose(fd(1) - fd(0), 0.24), "do(X=0)=0.23、効果 0.24")
# 公式の妥当性：未観測 U をもつランダムな SCM で、フロントドア公式＝真の介入効果
ok = True
for _ in range(50):
    pu = rng.uniform(0.2, 0.8); px_u = rng.uniform(0.1, 0.9, 2); pm_x = rng.uniform(0.1, 0.9, 2); py_mu = rng.uniform(0.05, 0.95, (2, 2))
    joint = {}
    for u, x, m, y in itertools.product((0, 1), repeat=4):
        p = (pu if u else 1 - pu) * (px_u[u] if x else 1 - px_u[u]) * (pm_x[x] if m else 1 - pm_x[x]) * (py_mu[m, u] if y else 1 - py_mu[m, u])
        joint[(u, x, m, y)] = p
    P_ = lambda **kw: sum(v for k, v in joint.items() if all(k["uxmy".index(a)] == b for a, b in kw.items()))
    for x in (0, 1):
        true = sum((pu if u else 1 - pu) * (pm_x[x] if m else 1 - pm_x[x]) * py_mu[m, u] for u in (0, 1) for m in (0, 1))
        est = sum(P_(x=x, m=m) / P_(x=x) * sum(P_(x=x2, m=m, y=1) / P_(x=x2, m=m) * P_(x=x2) for x2 in (0, 1)) for m in (0, 1))
        ok &= abs(true - est) < 1e-10
q.check("q33-frontdoor-criterion-1", bool(ok), "ランダムな SCM 50個でフロントドア公式が真の介入効果と一致")
# 直接の矢印があると一致しない例
pu, px_u, pm_x = 0.5, np.array([0.3, 0.7]), np.array([0.2, 0.8])
py = lambda x, m, u: 0.1 + 0.3 * m + 0.2 * u + 0.3 * x
joint = {}
for u, x, m, y in itertools.product((0, 1), repeat=4):
    pyv = py(x, m, u)
    joint[(u, x, m, y)] = 0.5 * (px_u[u] if x else 1 - px_u[u]) * (pm_x[x] if m else 1 - pm_x[x]) * (pyv if y else 1 - pyv)
P_ = lambda **kw: sum(v for k, v in joint.items() if all(k["uxmy".index(a)] == b for a, b in kw.items()))
true1 = sum(0.5 * (pm_x[1] if m else 1 - pm_x[1]) * py(1, m, u) for u in (0, 1) for m in (0, 1))
est1 = sum(P_(x=1, m=m) / P_(x=1) * sum(P_(x=x2, m=m, y=1) / P_(x=x2, m=m) * P_(x=x2) for x2 in (0, 1)) for m in (0, 1))
q.check("q33-frontdoor-criterion-2", abs(true1 - est1) > 0.01, "直接の矢印があると公式は真の値と一致しない（誤り）")

# --- 導出問題（追加）---
_pz = {0: 0.7, 1: 0.3}
_py = {(1, 0): 0.5, (1, 1): 0.7, (0, 0): 0.3, (0, 1): 0.6}
_do = {x: sum(_py[(x, z)] * _pz[z] for z in (0, 1)) for x in (0, 1)}
# 逆確率重み付け：P(X|Z) を任意に与えても（正値性のもとで）同じ値になる
_ok = True
for _px1 in [(0.2, 0.6), (0.5, 0.5), (0.9, 0.1)]:
    _pxz = {(1, 0): _px1[0], (1, 1): _px1[1], (0, 0): 1 - _px1[0], (0, 1): 1 - _px1[1]}
    for _x in (0, 1):
        _ipw = sum(_pz[z] * _pxz[(_x, z)] * _py[(_x, z)] / _pxz[(_x, z)] for z in (0, 1))
        _ok &= abs(_ipw - _do[_x]) < 1e-12
# 切断された分解：同時分布から P(x|z) を除いて和をとったものと一致
_ok &= all(abs(sum(_pz[z] * _py[(x, z)] for z in (0, 1)) - _do[x]) < 1e-12 for x in (0, 1))
_ace = _do[1] - _do[0]
q.check("q33-backdoor-criterion-4", _ok and abs(_do[1] - 0.56) < 1e-12 and abs(_do[0] - 0.39) < 1e-12 and round(_ace, 2) == 0.17
        and abs((0.2 + 0.1) / 2 - 0.17) > 0.01, f"調整化公式＝逆確率重み付け、ACE={_ace:.4f}、誤答 0.15 とは異なる")

# --- 4択の追加（ATE・フロントドア基準の状況判断）---
_Y0 = np.array([2, 3, 1, 4, 2, 5]); _Y1 = np.array([5, 4, 4, 4, 3, 7]); _T = np.array([1, 0, 1, 0, 0, 1])
_ate = (_Y1 - _Y0).mean()
_naive = _Y1[_T == 1].mean() - _Y0[_T == 0].mean()
_att = (_Y1 - _Y0)[_T == 1].mean(); _atc = (_Y1 - _Y0)[_T == 0].mean()
q.check("q33-causal-effect-4", round(_ate, 2) == 1.67 and [round(v, 2) for v in (_naive, _att, _atc)] == [2.33, 2.67, 0.67],
        f"ATE {_ate:.4f}、群間差 {_naive:.4f}・ATT {_att:.4f}・ATC {_atc:.4f}")
# U→X, U→Y, X→M→Y の数値例
_pu = 0.4; _px = {0: 0.2, 1: 0.8}; _pm = {0: 0.1, 1: 0.7}
_pyf = lambda m, u: 0.1 + 0.5 * m + 0.3 * u
_J = {}
for u, x, m, y in itertools.product((0, 1), repeat=4):
    pu = _pu if u else 1 - _pu; px = _px[u] if x else 1 - _px[u]
    pm = _pm[x] if m else 1 - _pm[x]; py = _pyf(m, u) if y else 1 - _pyf(m, u)
    _J[(u, x, m, y)] = pu * px * pm * py
_Pr = lambda **kw: sum(v for k, v in _J.items() if all(k["uxmy".index(a)] == b for a, b in kw.items()))
_true = sum((_pu if u else 1 - _pu) * (_pm[1] if m else 1 - _pm[1]) * _pyf(m, u) for u in (0, 1) for m in (0, 1))
_fd = sum(_Pr(x=1, m=m) / _Pr(x=1) * sum(_Pr(x=x2, m=m, y=1) / _Pr(x=x2, m=m) * _Pr(x=x2) for x2 in (0, 1)) for m in (0, 1))
_bd_m = sum(_Pr(x=1, m=m, y=1) / _Pr(x=1, m=m) * _Pr(m=m) for m in (0, 1))   # M で調整（誤り）
_cond = _Pr(x=1, y=1) / _Pr(x=1)
_ok = abs(_fd - _true) < 1e-12 and abs(_bd_m - _true) > 0.01 and abs(_cond - _true) > 0.01
_ok &= dsep([("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")], "M", "U", {"X"})
q.check("q33-frontdoor-criterion-3", _ok, f"真の値 {_true:.4f}＝フロントドア {_fd:.4f}、M で調整 {_bd_m:.4f}・P(y|x) {_cond:.4f} は不一致")

q.done()
