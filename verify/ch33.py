"""第33章 検算スクリプト（統計的因果推論）。
潜在結果の表（ATE・ATT・選択バイアス・無作為化20通りの平均）、2値の4タイプ、
バックドア調整（調整化公式・単純な条件付き確率）、DAG での基準の判定（d分離はモラルグラフ法で実装）、
線形モデルの欠落変数バイアス（理論値と乱数による回帰）、合流点バイアス、d分離の判定表と線形ガウスでの偏相関、
フロントドア調整（未観測 U を含む真のモデルとの一致）を再計算し、本文と照合する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4。各チェックに明記）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
乱数（seed=20261023）を使うチェックは、各チェックに許容誤差を明記する。
"""
from pathlib import Path
import itertools
import math
import numpy as np
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
    return xs.shape == claims.shape and bool(np.all(np.abs(xs - claims) <= tol + 1e-12))


def front(term):
    txt = (ROOT / "content" / "ch33" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def charts(term, cid):
    return [c for c in front(term).get("charts", []) if c["id"] == cid]


def b(p, v):
    return p if v == 1 else 1 - p


# ---------------------------------------------------------------- グラフの道具
def parents(G, v):
    return {i for i, j in G if j == v}


def children(G, v):
    return {j for i, j in G if i == v}


def ancestors(G, S):
    out, stack = set(S), list(S)
    while stack:
        v = stack.pop()
        for p in parents(G, v):
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


def descendants(G, v):
    out, stack = set(), [v]
    while stack:
        u = stack.pop()
        for c in children(G, u):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def dsep(G, X, Y, Z):
    """モラルグラフ法：X∪Y∪Z の祖先集合に制限してモラル化し、Z を除いて X と Y がつながらなければ d分離。"""
    X, Y, Z = set(X), set(Y), set(Z)
    A = ancestors(G, X | Y | Z)
    E = set()
    for i, j in G:
        if i in A and j in A:
            E.add(frozenset((i, j)))
    for v in A:
        ps = [p for p in parents(G, v) if p in A]
        for p, q in itertools.combinations(ps, 2):
            E.add(frozenset((p, q)))
    seen, stack = set(X), list(X)
    while stack:
        u = stack.pop()
        for e in E:
            if u in e:
                (w,) = e - {u}
                if w not in Z and w not in seen:
                    seen.add(w)
                    stack.append(w)
    return not (seen & Y)


def backdoor_ok(G, X, Y, Z):
    if any(z in descendants(G, X) for z in Z):
        return False
    Gx = [(i, j) for i, j in G if i != X]  # X から出る矢印を除く
    return dsep(Gx, {X}, {Y}, set(Z))


# ================================================================ causal-effect
t = "causal-effect"
y0 = np.array([50, 55, 60, 70, 75, 80])
y1 = np.array([58, 62, 66, 74, 78, 82])
tau = y1 - y0
check(t, "例1 効果 8,7,6,4,3,2", tau.tolist() == [8, 7, 6, 4, 3, 2], tau.tolist())
ate = tau.mean()
check(t, "例1 ATE=5", close(ate, 5, 1e-12), ate)
T = np.array([1, 1, 1, 0, 0, 0])
m1, m0 = y1[T == 1].mean(), y0[T == 0].mean()
check(t, "例1 観測平均 62 と 75、単純な差 −13", close(m1, 62, 1e-12) and close(m0, 75, 1e-12) and close(m1 - m0, -13, 1e-12), (m1, m0))
att = tau[T == 1].mean()
sel = y0[T == 1].mean() - y0[T == 0].mean()
check(t, "例1 ATT=7、選択バイアス 55−75=−20、和=−13", close(att, 7, 1e-12) and close(y0[T == 1].mean(), 55, 1e-12) and close(sel, -20, 1e-12) and close(att + sel, m1 - m0, 1e-12), (att, sel))
diffs = []
for s in itertools.combinations(range(6), 3):
    tt = np.zeros(6, int)
    tt[list(s)] = 1
    diffs.append(y1[tt == 1].mean() - y0[tt == 0].mean())
diffs = np.array(diffs)
check(t, "例2 C(6,3)=20 通り、平均 5、最小 −13、最大 23", len(diffs) == 20 == math.comb(6, 3) and close(diffs.mean(), 5, 1e-12)
      and close(diffs.min(), -13, 1e-12) and close(diffs.max(), 23, 1e-12), (len(diffs), diffs.mean(), diffs.min(), diffs.max()))
types = {"help": (0, 1, 0.30), "always": (1, 1, 0.20), "never": (0, 0, 0.45), "hurt": (1, 0, 0.05)}
check(t, "例3 割合の合計 1", close(sum(v[2] for v in types.values()), 1, 1e-12), "1")
p1 = sum(w for a, c, w in types.values() if c == 1)
p0 = sum(w for a, c, w in types.values() if a == 1)
check(t, "例3 P(Y(1)=1)=0.50、P(Y(0)=1)=0.25、ATE=0.25", close(p1, 0.5, 1e-12) and close(p0, 0.25, 1e-12) and close(p1 - p0, 0.25, 1e-12), (p1, p0))
# 回復する人の割合 = ATE + 悪化する人の割合（回復率だけでは決まらない）
ok = True
for hurt in [0, 0.05, 0.1, 0.2]:
    # 周辺 P(Y1=1)=0.5, P(Y0=1)=0.25 を保つ4タイプの割合
    helpp = 0.25 + hurt
    always = 0.5 - helpp
    never = 1 - helpp - always - hurt
    ok &= min(helpp, always, never) >= -1e-12 and close(always + helpp, 0.5, 1e-12) and close(always + hurt, 0.25, 1e-12)
check(t, "例3 悪化 0 なら回復型 0.25、悪化 >0 なら 0.25 より大きくても周辺は同じ（0, 0.05, 0.1, 0.2 で確認）", ok, "成立")
ld = charts(t, "line-data")[0]["params"]
check(t, "charts line-data の Y(0), Y(1) が表と一致", ld["series"][0]["y"] == y0.tolist() and ld["series"][1]["y"] == y1.tolist(), "一致")
bd = charts(t, "bar-data")[0]["params"]
check(t, "charts bar-data [5, −13, 7, −20]", allclose(bd["values"], [ate, m1 - m0, att, sel], 1e-12), bd["values"])

# ================================================================ backdoor-criterion
t = "backdoor-criterion"
pz = {1: 0.4, 0: 0.6}
pxz = {1: 0.75, 0: 0.25}
py = {(1, 0): 0.9, (0, 0): 0.8, (1, 1): 0.5, (0, 1): 0.4}
do = {x: sum(pz[z] * py[(x, z)] for z in (0, 1)) for x in (0, 1)}
check(t, "例1 P(Y=1|do(X=1))=0.74、do(X=0)=0.64、効果 0.10", close(do[1], 0.74, 1e-12) and close(do[0], 0.64, 1e-12) and close(do[1] - do[0], 0.10, 1e-12), do)
PX1 = sum(pz[z] * pxz[z] for z in (0, 1))
w1 = pz[1] * pxz[1] / PX1
w0 = pz[1] * (1 - pxz[1]) / (1 - PX1)
check(t, "例1 P(X=1)=0.45、P(Z=1|X=1)=2/3、P(Z=1|X=0)=0.10/0.55≈0.1818", close(PX1, 0.45, 1e-12) and close(w1, 2 / 3, 1e-12) and close(w0, 0.1818, 5e-5)
      and close(pz[1] * (1 - pxz[1]), 0.10, 1e-12), (PX1, w1, w0))
c1 = (1 - w1) * py[(1, 0)] + w1 * py[(1, 1)]
c0 = (1 - w0) * py[(0, 0)] + w0 * py[(0, 1)]
check(t, "例1 P(Y=1|X=1)≈0.6333、P(Y=1|X=0)≈0.7273、差≈−0.0939", close(c1, 0.6333, 5e-5) and close(c0, 0.7273, 5e-5) and close(c1 - c0, -0.0939, 5e-5), (c1, c0, c1 - c0))
# 同時分布から直接（全確率）でも確認
J = {(z, x, y): pz[z] * b(pxz[z], x) * b(py[(x, z)], y) for z in (0, 1) for x in (0, 1) for y in (0, 1)}
c1b = sum(J[(z, 1, 1)] for z in (0, 1)) / sum(J[(z, 1, y)] for z in (0, 1) for y in (0, 1))
check(t, "例1 同時分布からの P(Y=1|X=1) と一致", close(c1b, c1, 1e-12), c1b)
check(t, "例1 逆確率重み付け Σ P(z,x,y)/P(x|z) でも 0.74", close(sum(J[(z, 1, 1)] / pxz[z] for z in (0, 1)), 0.74, 1e-12), "0.74")
G1 = [("Z", "X"), ("Z", "Y"), ("X", "Y")]
check(t, "例1 {Z} は基準を満たし、空集合は満たさない", backdoor_ok(G1, "X", "Y", {"Z"}) and not backdoor_ok(G1, "X", "Y", set()), "成立")
G2 = [("A", "X"), ("A", "Y"), ("X", "M"), ("M", "Y"), ("X", "C"), ("Y", "C")]
res = {k: backdoor_ok(G2, "X", "Y", s) for k, s in {"∅": set(), "{A}": {"A"}, "{A,M}": {"A", "M"}, "{A,C}": {"A", "C"}}.items()}
check(t, "例2 ∅×、{A}○、{A,M}×、{A,C}×", res == {"∅": False, "{A}": True, "{A,M}": False, "{A,C}": False}, res)
check(t, "例2 M と C は X の子孫", {"M", "C"} <= descendants(G2, "X"), sorted(descendants(G2, "X")))
# 線形モデル
a, bx, g = 0.8, 0.5, 0.6
VX = a ** 2 + 0.36
covXY = bx * VX + g * a
check(t, "例3 V[X]=1、単回帰係数 0.5+0.6×0.8=0.98", close(VX, 1, 1e-12) and close(covXY / VX, 0.98, 1e-12), covXY / VX)
n = 200000
Z = rng.normal(size=n)
X = a * Z + rng.normal(scale=0.6, size=n)
Y = bx * X + g * Z + rng.normal(size=n)
s1 = np.polyfit(X, Y, 1)[0]
coef = np.linalg.lstsq(np.column_stack([np.ones(n), X, Z]), Y, rcond=None)[0]
check(t, "例3 乱数 n=200000：単回帰 ≈0.98、Z を加えた重回帰 ≈0.5（許容 0.01）", abs(s1 - 0.98) < 0.01 and abs(coef[1] - 0.5) < 0.01, (round(s1, 4), round(coef[1], 4)))
bars = charts(t, "bar-data")[0]["params"]["values"]
check(t, "charts bar-data [0.6333, 0.7273, 0.74, 0.64]", allclose(bars, [c1, c0, do[1], do[0]], 5e-5), bars)
gn = charts(t, "graph-network")
g1 = gn[0]["params"]
e1 = {(g1["nodes"][i].split()[-1], g1["nodes"][j].split()[-1]) for i, j in g1["edges"]}
g2 = gn[1]["params"]
e2 = {(g2["nodes"][i].split()[-1], g2["nodes"][j].split()[-1]) for i, j in g2["edges"]}
check(t, "charts graph-network 2つの辺が例1・例2の DAG と一致", e1 == set(G1) and e2 == set(G2) and g1["directed"] == 1 and g2["directed"] == 1, (sorted(e1), sorted(e2)))

# ================================================================ d-separation
t = "d-separation"
ps = {(0, 0): 0.1, (1, 0): 0.5, (0, 1): 0.5, (1, 1): 0.9}
J = {(tt, e, s): 0.25 * b(ps[(tt, e)], s) for tt in (0, 1) for e in (0, 1) for s in (0, 1)}
P = lambda f: sum(v for k, v in J.items() if f(*k))
PS = P(lambda tt, e, s: s == 1)
check(t, "例1 P(S=1)=0.5", close(PS, 0.5, 1e-12), PS)
q = {(tt, e): J[(tt, e, 1)] / PS for tt in (0, 1) for e in (0, 1)}
check(t, "例1 合格者内の同時確率 0.45, 0.25, 0.25, 0.05", allclose([q[(1, 1)], q[(1, 0)], q[(0, 1)], q[(0, 0)]], [0.45, 0.25, 0.25, 0.05], 1e-12), q)
pt_s = q[(1, 1)] + q[(1, 0)]
pe1 = q[(1, 1)] / (q[(1, 1)] + q[(0, 1)])
pe0 = q[(1, 0)] / (q[(1, 0)] + q[(0, 0)])
check(t, "例1 P(T=1|S=1)=0.7、P(T=1|S=1,E=1)≈0.6429、P(T=1|S=1,E=0)≈0.8333", close(pt_s, 0.7, 1e-12) and close(pe1, 0.6429, 5e-5) and close(pe0, 0.8333, 5e-5)
      and close(pe1, 0.9 / 1.4, 1e-12) and close(pe0, 0.5 / 0.6, 1e-12), (pt_s, pe1, pe0))
a_, b_, c_, d_ = (P(lambda tt, e, s, T_=T_, E_=E_: tt == T_ and e == E_ and s == 1) for T_, E_ in [(1, 1), (1, 0), (0, 1), (0, 0)])
orr = a_ * d_ / (b_ * c_)
check(t, "例1 合格者内のオッズ比 0.225×0.025/(0.125×0.125)=0.36", allclose([a_, b_, c_, d_], [0.225, 0.125, 0.125, 0.025], 1e-12) and close(orr, 0.36, 1e-12), orr)
check(t, "例1 全体では T と E は独立（P(T=1,E=1)=0.25=0.5×0.5）", close(P(lambda tt, e, s: tt == 1 and e == 1), 0.25, 1e-12), "成立")
G = [("A", "C"), ("B", "C"), ("C", "D"), ("B", "E")]
table = [({"A"}, {"B"}, set(), True), ({"A"}, {"B"}, {"D"}, False), ({"A"}, {"E"}, set(), True),
         ({"A"}, {"E"}, {"C"}, False), ({"A"}, {"E"}, {"C", "B"}, True)]
check(t, "例2 判定表（○, ×, ○, ×, ○）をモラルグラフ法で確認", all(dsep(G, x, y, z) == ans for x, y, z, ans in table), [dsep(G, x, y, z) for x, y, z, _ in table])


def pcorr_lin(G, coefs, order, x, y, zs):
    """線形ガウス SEM（誤差分散1）の共分散を作り、偏相関を返す。"""
    idx = {v: i for i, v in enumerate(order)}
    p = len(order)
    B = np.zeros((p, p))
    for (i, j), cf in zip(G, coefs):
        B[idx[j], idx[i]] = cf
    IB = np.linalg.inv(np.eye(p) - B)
    S = IB @ IB.T
    sel = [idx[x], idx[y]] + [idx[z] for z in zs]
    Om = np.linalg.inv(S[np.ix_(sel, sel)])
    return -Om[0, 1] / math.sqrt(Om[0, 0] * Om[1, 1])


order = ["A", "B", "C", "D", "E"]
ok = True
for _ in range(50):
    cf = rng.uniform(0.3, 1.5, size=len(G)) * rng.choice([-1, 1], size=len(G))
    for x, y, z, ans in table:
        r_ = pcorr_lin(G, cf, order, next(iter(x)), next(iter(y)), sorted(z))
        ok &= (abs(r_) < 1e-10) if ans else (abs(r_) > 1e-6)
check(t, "例2 線形ガウスの偏相関：d分離なら 0、そうでなければ 0 でない（乱数の係数50組）", ok, "成立")
gp = charts(t, "graph-network")[0]["params"]
ge = {(gp["nodes"][i], gp["nodes"][j]) for i, j in gp["edges"]}
check(t, "charts graph-network の辺が例2と一致", ge == set(G), sorted(ge))
check(t, "charts bar-data [0.5, 0.7, 0.6429, 0.8333]", allclose(charts(t, "bar-data")[0]["params"]["values"], [0.5, pt_s, pe1, pe0], 5e-5), "一致")

# ================================================================ frontdoor-criterion
t = "frontdoor-criterion"
pu = {1: 0.3, 0: 0.7}
pxu = {1: 0.8, 0: 0.2}
pmx = {1: 0.9, 0: 0.1}
pymu = {(0, 0): 0.1, (1, 0): 0.3, (0, 1): 0.4, (1, 1): 0.6}
J = {(u, x, m, y): pu[u] * b(pxu[u], x) * b(pmx[x], m) * b(pymu[(m, u)], y) for u, x, m, y in itertools.product((0, 1), repeat=4)}
P = lambda f: sum(v for k, v in J.items() if f(*k))
Px = {x: P(lambda u, X, m, y, x=x: X == x) for x in (0, 1)}
Pm = {x: P(lambda u, X, m, y, x=x: X == x and m == 1) / Px[x] for x in (0, 1)}
check(t, "観測 P(X=1)=0.38、P(M=1|X=1)=0.9、P(M=1|X=0)=0.1", close(Px[1], 0.38, 1e-12) and close(Pm[1], 0.9, 1e-12) and close(Pm[0], 0.1, 1e-12), (Px[1], Pm))
Pyxm = {(x, mm): P(lambda u, X, m, y, x=x, mm=mm: X == x and m == mm and y == 1) / P(lambda u, X, m, y, x=x, mm=mm: X == x and m == mm) for x in (0, 1) for mm in (0, 1)}
check(t, "観測 P(Y=1|x,m)=0.1290, 0.3290, 0.2895, 0.4895（第4位）", allclose([Pyxm[(0, 0)], Pyxm[(0, 1)], Pyxm[(1, 0)], Pyxm[(1, 1)]], [0.1290, 0.3290, 0.2895, 0.4895], 5e-5), Pyxm)
inner = {mm: sum(Pyxm[(x2, mm)] * Px[x2] for x2 in (0, 1)) for mm in (0, 1)}
check(t, "内側の和 0.19（0.08+0.11）と 0.39（0.204+0.186）", close(inner[0], 0.19, 1e-12) and close(inner[1], 0.39, 1e-12)
      and allclose([Pyxm[(0, 0)] * 0.62, Pyxm[(1, 0)] * 0.38, Pyxm[(0, 1)] * 0.62, Pyxm[(1, 1)] * 0.38], [0.08, 0.11, 0.204, 0.186], 1e-12), inner)
fd = {x: sum(b(Pm[x], mm) * inner[mm] for mm in (0, 1)) for x in (0, 1)}
check(t, "フロントドア公式 do(X=1)=0.37、do(X=0)=0.21、効果 0.16", close(fd[1], 0.37, 1e-12) and close(fd[0], 0.21, 1e-12) and close(fd[1] - fd[0], 0.16, 1e-12), fd)
true = {x: sum(pu[u] * b(pmx[x], mm) * pymu[(mm, u)] for u in (0, 1) for mm in (0, 1)) for x in (0, 1)}
check(t, "U を使った真の介入効果 0.16 と一致", close(true[1] - true[0], 0.16, 1e-12) and allclose([true[0], true[1]], [fd[0], fd[1]], 1e-12), true)
nv = {x: P(lambda u, X, m, y, x=x: X == x and y == 1) / Px[x] for x in (0, 1)}
check(t, "単純な比較 0.4695−0.1490≈0.3204（第4位）、約2倍", close(nv[1], 0.4695, 5e-5) and close(nv[0], 0.1490, 5e-5) and close(nv[1] - nv[0], 0.3204, 5e-5)
      and 1.9 < (nv[1] - nv[0]) / 0.16 < 2.1, (nv, (nv[1] - nv[0]) / 0.16))
GF = [("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")]
def dpaths(G, a, z, path=None):
    path = (path or []) + [a]
    if a == z:
        return [path]
    return [q for c in children(G, a) for q in dpaths(G, c, z, path)]


c1_ = all("M" in q for q in dpaths(GF, "X", "Y")) and len(dpaths(GF, "X", "Y")) == 1  # 条件1：X→Y の有向道はすべて M を通る
c2_ = backdoor_ok(GF, "X", "M", set())
c3_ = backdoor_ok(GF, "M", "Y", {"X"})
check(t, "基準の確認：条件2（X→M は空集合でバックドア基準）、条件3（M→Y は {X}）、バックドアで X→Y は {U} が必要", c1_ and c2_ and c3_
      and not backdoor_ok(GF, "X", "Y", set()) and backdoor_ok(GF, "X", "Y", {"U"}), (c2_, c3_))
check(t, "U で調整したバックドア公式でも 0.37 / 0.21", allclose([sum(pu[u] * P(lambda U_, X, m, y, u=u, x=x: U_ == u and X == x and y == 1) / P(lambda U_, X, m, y, u=u, x=x: U_ == u and X == x) for u in (0, 1)) for x in (1, 0)], [0.37, 0.21], 1e-12), "成立")
gp = charts(t, "graph-network")[0]["params"]
ge = {(gp["nodes"][i].split()[-1].replace("（未観測）", ""), gp["nodes"][j].split()[-1].replace("（未観測）", "")) for i, j in gp["edges"]}
check(t, "charts graph-network の辺が DAG と一致", ge == set(GF), sorted(ge))

# ================================================================ 集計
okn = sum(results)
ngn = len(results) - okn
print(f"\n合計: OK {okn} / NG {ngn}")
