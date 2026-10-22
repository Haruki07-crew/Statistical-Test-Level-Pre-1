"""第24章 検算スクリプト（クラスター分析）。
5店舗の距離行列での最短・最長・群平均法の併合過程、ウォード法の Δ、重心法の逆転、
ランス・ウィリアムズの係数表（7手法）、k-means の反復と大域最適性、コーフェン相関、
エルボー法・シルエット係数（9点の全分割を総当たり）、距離・類似度の数値例を
scipy.cluster.hierarchy と numpy で再計算し、本文と照合する。
主張値との比較は表示桁の丸め幅（小数第4位なら 5e-5、第3位なら 5e-4 など。各チェックに明記）。
charts の params は各 md の front matter を yaml で読み、本文と照合する。
乱数を使うチェック（更新式と直接計算の一致、図の JS アルゴリズムの移植と scipy の一致）は
seed=20261023 で、許容誤差 1e-9（決定的な等式なので数値誤差のみ）。
"""
from pathlib import Path
from itertools import combinations
import math
import numpy as np
import sympy as sp
import yaml
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform, pdist
from scipy.sparse.csgraph import minimum_spanning_tree

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
    txt = (ROOT / "content" / "ch24" / f"{term}.md").read_text(encoding="utf-8")
    return yaml.safe_load(txt.split("---")[1])


def chart(term, cid, idx=0):
    return [c for c in front(term).get("charts", []) if c["id"] == cid][idx]


def r(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


# ---------------------------------------------------------------- 5店舗の距離行列
L = "ABCDE"
D = np.array([[0, 3, 7, 11, 12],
              [3, 0, 4, 8, 9],
              [7, 4, 0, 5, 5],
              [11, 8, 5, 0, 2],
              [12, 9, 5, 2, 0]], float)
dv = squareform(D)


def lw_coef(method, ni, nj, nk):
    """ランス・ウィリアムズの係数 (alpha_i, alpha_j, beta, gamma)（本文の表どおり）"""
    if method == "single":
        return 0.5, 0.5, 0.0, -0.5
    if method == "complete":
        return 0.5, 0.5, 0.0, 0.5
    if method == "average":
        return ni / (ni + nj), nj / (ni + nj), 0.0, 0.0
    if method == "weighted":
        return 0.5, 0.5, 0.0, 0.0
    if method == "centroid":
        return ni / (ni + nj), nj / (ni + nj), -ni * nj / (ni + nj) ** 2, 0.0
    if method == "median":
        return 0.5, 0.5, -0.25, 0.0
    if method == "ward":
        s = ni + nj + nk
        return (ni + nk) / s, (nj + nk) / s, -nk / s, 0.0
    raise ValueError(method)


def lw_cluster(Dm, method, names=None):
    """更新式だけで凝集型クラスタリングを行い、各段の (併合した組, 高さ, 距離表) を返す"""
    n = len(Dm)
    names = names or [str(i) for i in range(n)]
    Dm = Dm.astype(float).copy()
    clusters = {i: (names[i], 1) for i in range(n)}
    steps = []
    while len(clusters) > 1:
        act = sorted(clusters)
        best, bi, bj = math.inf, None, None
        for a, b in combinations(act, 2):
            if Dm[a, b] < best - 1e-12:
                best, bi, bj = Dm[a, b], a, b
        ni, nj = clusters[bi][1], clusters[bj][1]
        for k in act:
            if k in (bi, bj):
                continue
            nk = clusters[k][1]
            ai, aj, be, ga = lw_coef(method, ni, nj, nk)
            v = ai * Dm[k, bi] + aj * Dm[k, bj] + be * best + ga * abs(Dm[k, bi] - Dm[k, bj])
            Dm[k, bi] = Dm[bi, k] = v
        newname = "".join(sorted(clusters[bi][0] + clusters[bj][0]))
        steps.append((clusters[bi][0], clusters[bj][0], best))
        clusters[bi] = (newname, ni + nj)
        del clusters[bj]
        table = {(clusters[a][0], clusters[b][0]): Dm[a, b] for a, b in combinations(sorted(clusters), 2)}
        steps[-1] = steps[-1] + (table,)
    return steps


def tget(table, x, y):
    x, y = "".join(sorted(x)), "".join(sorted(y))
    return table.get((x, y), table.get((y, x)))


def two_clusters(Z):
    lab = fcluster(Z, 2, criterion="maxclust")
    groups = {}
    for i, l in enumerate(lab):
        groups.setdefault(l, []).append(L[i])
    return sorted("".join(g) for g in groups.values())


Zs = linkage(dv, "single")
Zc = linkage(dv, "complete")
Za = linkage(dv, "average")
Zw = linkage(dv, "weighted")
Ss = lw_cluster(D, "single", list(L))
Sc = lw_cluster(D, "complete", list(L))
Sa = lw_cluster(D, "average", list(L))
Sw = lw_cluster(D, "weighted", list(L))

# ---------------------------------------------------------------- hierarchical-clustering
T = "hierarchical-clustering"
check(T, "距離行列は対称・対角0・非負", np.allclose(D, D.T) and np.all(np.diag(D) == 0) and np.all(D >= 0), "OK")
check(T, "例1 最短距離法の併合順 DE→AB→ABC→全体", [("".join(sorted(a + b))) for a, b, *_ in Ss] == ["DE", "AB", "ABC", "ABCDE"],
      [("".join(sorted(a + b))) for a, b, *_ in Ss])
check(T, "例1 段2の距離 A–B=3, B–C=4, C–DE=5, A–C=7, B–DE=8, A–DE=11",
      [tget(Ss[0][3], *p) for p in [("A", "B"), ("B", "C"), ("C", "DE"), ("A", "C"), ("B", "DE"), ("A", "DE")]] == [3, 4, 5, 7, 8, 11],
      [tget(Ss[0][3], *p) for p in [("A", "B"), ("B", "C"), ("C", "DE"), ("A", "C"), ("B", "DE"), ("A", "DE")]])
check(T, "例1 段3の距離 C–AB=4, C–DE=5, AB–DE=8",
      [tget(Ss[1][3], *p) for p in [("C", "AB"), ("C", "DE"), ("AB", "DE")]] == [4, 5, 8],
      [tget(Ss[1][3], *p) for p in [("C", "AB"), ("C", "DE"), ("AB", "DE")]])
check(T, "例1 段4 ABC–DE=5", tget(Ss[2][3], "ABC", "DE") == 5, tget(Ss[2][3], "ABC", "DE"))
for name, Z, S, h, cut in [("最短距離法", Zs, Ss, [2, 3, 4, 5], ["ABC", "DE"]),
                           ("最長距離法", Zc, Sc, [2, 3, 5, 12], ["AB", "CDE"]),
                           ("群平均法", Za, Sa, [2, 3, 5, 8.5], ["AB", "CDE"])]:
    check(T, f"例2 {name}の高さ {h}（scipy）", allclose(Z[:, 2], h, 1e-12), r(Z[:, 2]))
    check(T, f"例2 {name}の高さ（更新式での手計算と一致）", allclose([s[2] for s in S], h, 1e-12), [s[2] for s in S])
    check(T, f"例2 {name}の2クラスター {cut}", two_clusters(Z) == cut, two_clusters(Z))
check(T, "例2 3段目の併合: 最短は C+AB、最長・群平均は C+DE",
      "".join(sorted(Ss[2][0] + Ss[2][1])) == "ABC" and "".join(sorted(Sc[2][0] + Sc[2][1])) == "CDE"
      and "".join(sorted(Sa[2][0] + Sa[2][1])) == "CDE", "OK")
check(T, "併合は n−1=4 回", len(Zs) == 4, len(Zs))

# ---------------------------------------------------------------- single-linkage
T = "single-linkage"
x_, y_ = sp.symbols("x y", real=True)
expr = sp.Rational(1, 2) * x_ + sp.Rational(1, 2) * y_ - sp.Rational(1, 2) * sp.Abs(x_ - y_)
check(T, "½d_ki+½d_kj−½|d_ki−d_kj| = min（sympy、x<y と x>y の両場合）",
      sp.simplify(expr.subs(x_, y_ + 1) - sp.Min(y_ + 1, y_)) == 0 and sp.simplify(expr.subs(x_, y_ - 1) - sp.Min(y_ - 1, y_)) == 0, "OK")
t1 = Ss[0][3]
check(T, "例1 段1後 d(A,DE)=11, d(B,DE)=8, d(C,DE)=5", [tget(t1, "A", "DE"), tget(t1, "B", "DE"), tget(t1, "C", "DE")] == [11, 8, 5],
      [tget(t1, "A", "DE"), tget(t1, "B", "DE"), tget(t1, "C", "DE")])
t2 = Ss[1][3]
check(T, "例1 段2後 d(C,AB)=4, d(DE,AB)=8", [tget(t2, "C", "AB"), tget(t2, "DE", "AB")] == [4, 8], [tget(t2, "C", "AB"), tget(t2, "DE", "AB")])
check(T, "例1 段3後 d(DE,ABC)=5", tget(Ss[2][3], "DE", "ABC") == 5, 5)
check(T, "例1 高さ 2→3→4→5、2クラスター {A,B,C},{D,E}", allclose(Zs[:, 2], [2, 3, 4, 5], 0) and two_clusters(Zs) == ["ABC", "DE"], r(Zs[:, 2]))
mst = minimum_spanning_tree(D).toarray()
w = sorted(mst[mst > 0])
check(T, "例2 最小全域木の辺の重み 2,3,4,5、合計 14", w == [2, 3, 4, 5] and sum(w) == 14, w)
check(T, "例2 MST の重み = 併合の高さ", allclose(w, Zs[:, 2], 0), "一致")
check(T, "距離を2乗しても併合順が同じ（単調変換）",
      np.array_equal(linkage(squareform(D ** 2), "single")[:, :2], Zs[:, :2]), "一致")
# MST 一致をランダムデータでも確認
okm = True
for _ in range(50):
    P = rng.normal(size=(12, 2))
    dd = pdist(P)
    m = minimum_spanning_tree(squareform(dd)).toarray()
    okm &= np.allclose(sorted(m[m > 0]), linkage(dd, "single")[:, 2])
check(T, "乱数 12点×50回で MST の辺の重み = 最短距離法の高さ（許容 1e-9）", okm, "一致")

# ---------------------------------------------------------------- complete-linkage
T = "complete-linkage"
expr = sp.Rational(1, 2) * x_ + sp.Rational(1, 2) * y_ + sp.Rational(1, 2) * sp.Abs(x_ - y_)
check(T, "½d_ki+½d_kj+½|d_ki−d_kj| = max（sympy）",
      sp.simplify(expr.subs(x_, y_ + 1) - (y_ + 1)) == 0 and sp.simplify(expr.subs(x_, y_ - 1) - y_) == 0, "OK")
t1 = Sc[0][3]
check(T, "例1 段1後 d(A,DE)=12, d(B,DE)=9, d(C,DE)=5", [tget(t1, "A", "DE"), tget(t1, "B", "DE"), tget(t1, "C", "DE")] == [12, 9, 5],
      [tget(t1, "A", "DE"), tget(t1, "B", "DE"), tget(t1, "C", "DE")])
t2 = Sc[1][3]
check(T, "例1 段2後 d(C,AB)=7, d(DE,AB)=12", [tget(t2, "C", "AB"), tget(t2, "DE", "AB")] == [7, 12], [tget(t2, "C", "AB"), tget(t2, "DE", "AB")])
check(T, "例1 段3後 d(AB,CDE)=12", tget(Sc[2][3], "AB", "CDE") == 12, 12)
idx = {c: i for i, c in enumerate(L)}


def diam(s):
    return max([D[idx[a], idx[b]] for a, b in combinations(s, 2)] or [0])


check(T, "例2 直径 {C,D,E}=5, {A,B}=3, 最短の {A,B,C}=7", [diam("CDE"), diam("AB"), diam("ABC")] == [5, 3, 7],
      [diam("CDE"), diam("AB"), diam("ABC")])
# 直径 ≤ 高さ をランダムデータで
okd = True
for _ in range(30):
    P = rng.normal(size=(10, 2))
    dd = pdist(P)
    Z = linkage(dd, "complete")
    Dq = squareform(dd)
    members = {i: [i] for i in range(10)}
    for t, (a, b, h, _) in enumerate(Z):
        mem = members[int(a)] + members[int(b)]
        members[10 + t] = mem
        okd &= max(Dq[i, j] for i, j in combinations(mem, 2)) <= h + 1e-12
check(T, "乱数 10点×30回でクラスター直径 ≤ 併合の高さ（最長距離法）", okd, "成立")

# ---------------------------------------------------------------- average-linkage
T = "average-linkage"
t1 = Sa[0][3]
check(T, "例1 段1後 d(A,DE)=11.5, d(B,DE)=8.5, d(C,DE)=5", [tget(t1, "A", "DE"), tget(t1, "B", "DE"), tget(t1, "C", "DE")] == [11.5, 8.5, 5],
      [tget(t1, "A", "DE"), tget(t1, "B", "DE"), tget(t1, "C", "DE")])
t2 = Sa[1][3]
check(T, "例1 段2後 d(C,AB)=5.5, d(DE,AB)=10", [tget(t2, "C", "AB"), tget(t2, "DE", "AB")] == [5.5, 10], [tget(t2, "C", "AB"), tget(t2, "DE", "AB")])
check(T, "例1 段3後 d(AB,CDE)=(5.5+20)/3=8.5", close(tget(Sa[2][3], "AB", "CDE"), 8.5, 1e-12), tget(Sa[2][3], "AB", "CDE"))
allpairs = [D[idx[a], idx[b]] for a in "AB" for b in "CDE"]
check(T, "例1 全6組の平均 51/6=8.5", sum(allpairs) == 51 and close(np.mean(allpairs), 8.5, 1e-12), allpairs)
check(T, "例2 WPGMA の d(AB,CDE)=7.75（更新式・scipy weighted）",
      close(tget(Sw[2][3], "AB", "CDE"), 7.75, 1e-12) and close(Zw[-1, 2], 7.75, 1e-12), r(Zw[:, 2]))
check(T, "群平均は min と max の間（例: 段1後 A–DE）", 11 <= 11.5 <= 12, "11 ≤ 11.5 ≤ 12")
oka = True
for _ in range(30):
    P = rng.normal(size=(10, 2))
    dd = pdist(P)
    Dq = squareform(dd)
    Z = linkage(dd, "average")
    members = {i: [i] for i in range(10)}
    for t, (a, b, h, _) in enumerate(Z):
        A_, B_ = members[int(a)], members[int(b)]
        oka &= abs(np.mean([Dq[i, j] for i in A_ for j in B_]) - h) < 1e-9
        members[10 + t] = A_ + B_
check(T, "乱数 10点×30回で、群平均法の高さ = 全ペア距離の平均（定義どおり、許容 1e-9）", oka, "一致")

# ---------------------------------------------------------------- ward-method
T = "ward-method"
Pw = np.array([[0, 0], [2, 0], [1, 3], [6, 4], [7, 6]], float)
d2 = squareform(pdist(Pw, "sqeuclidean"))
pairs = ["AB", "AC", "BC", "CD", "BD", "CE", "AD", "BE", "AE", "DE"]
d2c = [4, 10, 10, 26, 32, 45, 52, 61, 85, 5]
check(T, "例1 段1の d² 表", [d2[idx[p[0]], idx[p[1]]] for p in pairs] == d2c, [d2[idx[p[0]], idx[p[1]]] for p in pairs])
check(T, "例1 段1の Δ=d²/2 表", allclose([d2[idx[p[0]], idx[p[1]]] / 2 for p in pairs], [2, 5, 5, 13, 16, 22.5, 26, 30.5, 42.5, 2.5], 0), "一致")


def ess(S):
    X = Pw[[idx[c] for c in S]]
    return float(((X - X.mean(0)) ** 2).sum())


def delta(a, b):
    return ess(a + b) - ess(a) - ess(b)


def delta_formula(a, b):
    Xa, Xb = Pw[[idx[c] for c in a]], Pw[[idx[c] for c in b]]
    return len(a) * len(b) / (len(a) + len(b)) * float(((Xa.mean(0) - Xb.mean(0)) ** 2).sum())


check(T, "例1 段2 Δ: C–AB=6, D–AB=27.333, E–AB=48",
      allclose([delta("C", "AB"), delta("D", "AB"), delta("E", "AB")], [6, 27.333, 48], 5e-4), r([delta("C", "AB"), delta("D", "AB"), delta("E", "AB")], 3))
check(T, "例1 段3 Δ: C–DE=22.833, AB–DE=55.25", allclose([delta("C", "DE"), delta("AB", "DE")], [22.833, 55.25], 5e-4),
      r([delta("C", "DE"), delta("AB", "DE")], 3))
check(T, "例1 段4 Δ(ABC,DE)=1.2×46.25=55.5", close(delta("ABC", "DE"), 55.5, 1e-9), delta("ABC", "DE"))
check(T, "Δ の公式 n_in_j/(n_i+n_j)‖差‖² = ESS の差（全組合せ）",
      all(abs(delta(a, b) - delta_formula(a, b)) < 1e-9 for a, b in [("C", "AB"), ("AB", "DE"), ("ABC", "DE"), ("C", "DE"), ("D", "AB")]), "一致")
check(T, "重心 AB=(1,0), DE=(6.5,5), ABC=(1,1)",
      allclose(np.r_[Pw[[0, 1]].mean(0), Pw[[3, 4]].mean(0), Pw[[0, 1, 2]].mean(0)], [1, 0, 6.5, 5, 1, 1], 1e-12), "一致")
Wseq = [0, ess("AB"), ess("AB") + ess("DE"), ess("ABC") + ess("DE"), ess("ABCDE")]
check(T, "例1 W の推移 0→2→4.5→10.5→66", allclose(Wseq, [0, 2, 4.5, 10.5, 66], 1e-9), Wseq)
dev = Pw - Pw.mean(0)
check(T, "全体平均 (3.2,2.6)、平方和 38.8+27.2=66",
      allclose(Pw.mean(0), [3.2, 2.6], 1e-12) and allclose((dev ** 2).sum(0), [38.8, 27.2], 1e-9), r((dev ** 2).sum(0)))
Sward = lw_cluster(d2, "ward", list(L))
check(T, "更新式（2乗距離）で併合順 AB→DE→ABC→全体、値=2Δ (4,5,12,111)",
      ["".join(sorted(a + b)) for a, b, *_ in Sward] == ["AB", "DE", "ABC", "ABCDE"] and allclose([s[2] for s in Sward], [4, 5, 12, 111], 1e-9),
      [s[2] for s in Sward])
check(T, "例2 更新式 d(C,AB)=(2/3)10+(2/3)10−(1/3)4=12=2×6", close(2 / 3 * 10 + 2 / 3 * 10 - 4 / 3, 12, 1e-12) and close(tget(Sward[0][3], "C", "AB"), 12, 1e-9), 12)
Zward = linkage(Pw, "ward")
check(T, "例2 scipy ward の高さ √(2Δ) = 2, 2.236, 3.464, 10.536", allclose(Zward[:, 2], [2, 2.236, 3.464, 10.536], 5e-4), r(Zward[:, 2], 3))
check(T, "1点同士は Δ=d²/2", close(delta("A", "B"), d2[0, 1] / 2, 1e-12), delta("A", "B"))
cw = chart(T, "line-data")["params"]
check(T, "charts line-data の W [0,2,4.5,10.5,66]", cw["x"] == [0, 1, 2, 3, 4] and allclose(cw["series"][0]["y"], Wseq, 1e-9), cw["series"][0]["y"])
cd = chart(T, "dendrogram")["params"]
check(T, "charts dendrogram method=ward、n=12 がスライダー範囲内", cd["method"] == "ward" and 4 <= cd["n"] <= 16, cd)
# ESS 分解の sympy 確認
ni, nj = sp.symbols("n_i n_j", positive=True)
a_, b_ = sp.symbols("a b", real=True)
xb = (ni * a_ + nj * b_) / (ni + nj)
check(T, "Δ の導出 n_i(a−x̄)²+n_j(b−x̄)² = n_in_j/(n_i+n_j)(a−b)²（sympy）",
      sp.simplify(ni * (a_ - xb) ** 2 + nj * (b_ - xb) ** 2 - ni * nj / (ni + nj) * (a_ - b_) ** 2) == 0, "OK")

# ---------------------------------------------------------------- centroid-method
T = "centroid-method"
Pc = np.array([[0, 0], [2, 0], [1, 1.8], [6, 4], [7, 6]], float)
d2c_ = squareform(pdist(Pc, "sqeuclidean"))
check(T, "例1 段1 AB=4, AC=BC=4.24, DE=5", allclose([d2c_[0, 1], d2c_[0, 2], d2c_[1, 2], d2c_[3, 4]], [4, 4.24, 4.24, 5], 1e-9),
      r([d2c_[0, 1], d2c_[0, 2], d2c_[1, 2], d2c_[3, 4]]))
check(T, "例1 段1の最小は AB", np.argmin(squareform(d2c_)) == 0, "AB")
cAB = Pc[[0, 1]].mean(0)
vals = [((Pc[2] - cAB) ** 2).sum(), ((Pc[3] - cAB) ** 2).sum(), ((Pc[4] - cAB) ** 2).sum(), d2c_[2, 3], d2c_[2, 4], d2c_[3, 4]]
check(T, "例1 段2の表 3.24, 41, 72, 29.84, 53.64, 5", allclose(vals, [3.24, 41, 72, 29.84, 53.64, 5], 1e-9), r(vals))
check(T, "例1 段2 更新式 ½4.24+½4.24−¼4=3.24", close(0.5 * 4.24 + 0.5 * 4.24 - 0.25 * 4, 3.24, 1e-12), 3.24)
cABC = Pc[[0, 1, 2]].mean(0)
check(T, "ABC の重心 (1,0.6)", allclose(cABC, [1, 0.6], 1e-12), cABC.tolist())
v3 = [((Pc[3] - cABC) ** 2).sum(), ((Pc[4] - cABC) ** 2).sum()]
check(T, "例1 段3 D–ABC=36.56, E–ABC=65.16", allclose(v3, [36.56, 65.16], 1e-9), r(v3))
cDE = Pc[[3, 4]].mean(0)
check(T, "例1 段4 高さ 5.5²+4.4²=49.61", close(((cABC - cDE) ** 2).sum(), 49.61, 1e-9), ((cABC - cDE) ** 2).sum())
Scen = lw_cluster(d2c_, "centroid", list(L))
check(T, "更新式で併合順 AB→ABC→DE→全体、高さ 4,3.24,5,49.61",
      ["".join(sorted(a + b)) for a, b, *_ in Scen] == ["AB", "ABC", "DE", "ABCDE"] and allclose([s[2] for s in Scen], [4, 3.24, 5, 49.61], 1e-9),
      r([s[2] for s in Scen]))
Zcen = linkage(Pc, "centroid")
check(T, "scipy centroid の高さ 2, 1.8, 2.236, 7.043", allclose(Zcen[:, 2], [2, 1.8, 2.236, 7.043], 5e-4), r(Zcen[:, 2]))
check(T, "逆転（2段目 < 1段目）", Zcen[1, 2] < Zcen[0, 2], r(Zcen[:, 2]))
cc = chart(T, "line-data")["params"]
check(T, "charts line-data y=[2,1.8,2.2361,7.0434]", allclose(cc["series"][0]["y"], Zcen[:, 2], 5e-5), r(Zcen[:, 2]))
Pw2 = Pc


def ess2(S):
    X = Pw2[[idx[c] for c in S]]
    return float(((X - X.mean(0)) ** 2).sum())


dCAB = ess2("ABC") - ess2("AB")
check(T, "同じ5点のウォード法: 段2 C–AB の Δ=2/3·3.24=2.16 < DE の 2.5、段1 Δ=2 より大", close(dCAB, 2.16, 1e-9) and dCAB < 2.5 and dCAB > 2, dCAB)
Zw2 = linkage(Pc, "ward")
check(T, "同じ5点のウォード法は単調（逆転なし）、2段目 = C+AB", np.all(np.diff(Zw2[:, 2]) >= 0) and set(map(int, Zw2[1, :2])) == {2, 5}, r(Zw2[:, 2]))
check(T, "重心法の α_i+α_j+β = 1 − n_in_j/(n_i+n_j)² < 1（n_i=n_j=1 で 0.75）", sum(lw_coef("centroid", 1, 1, 1)[:3]) == 0.75, 0.75)
check(T, "メディアン法の係数 ½,½,−¼,0", lw_coef("median", 2, 1, 1) == (0.5, 0.5, -0.25, 0.0), "OK")
# スチュワートの定理（重心法の更新式）の sympy 確認（1次元成分ごとで十分）
k_, ai_, aj_ = sp.symbols("k a_i a_j", real=True)
mid = (ni * ai_ + nj * aj_) / (ni + nj)
check(T, "更新式 = 内分点までの2乗距離（sympy）",
      sp.simplify(ni / (ni + nj) * (k_ - ai_) ** 2 + nj / (ni + nj) * (k_ - aj_) ** 2 - ni * nj / (ni + nj) ** 2 * (ai_ - aj_) ** 2 - (k_ - mid) ** 2) == 0, "OK")

# ---------------------------------------------------------------- lance-williams-formula
T = "lance-williams-formula"


def direct_dist(method, P, A_, B_):
    """クラスター間距離の直接計算（中心系は2乗ユークリッド）"""
    XA, XB = P[A_], P[B_]
    if method in ("centroid",):
        return float(((XA.mean(0) - XB.mean(0)) ** 2).sum())
    if method == "ward":
        na, nb = len(A_), len(B_)
        return 2 * na * nb / (na + nb) * float(((XA.mean(0) - XB.mean(0)) ** 2).sum())
    Dq = squareform(pdist(P))
    vals = [Dq[i, j] for i in A_ for j in B_]
    return {"single": min(vals), "complete": max(vals), "average": float(np.mean(vals))}[method]


ok_lw = {m: True for m in ["single", "complete", "average", "weighted", "centroid", "median", "ward"]}
for rep in range(40):
    n = 9
    P = rng.normal(size=(n, 2)) * rng.uniform(0.5, 2)
    for m in ok_lw:
        sq = m in ("centroid", "median", "ward")
        base = squareform(pdist(P, "sqeuclidean" if sq else "euclidean"))
        # 更新式でクラスタリングしながら、各段のすべての距離を直接計算と比べる
        Dm = base.copy()
        clusters = {i: [i] for i in range(n)}
        reps = {i: P[i].copy() for i in range(n)}  # WPGMA/WPGMC 用の代表点（中点）
        wdist = {}  # WPGMA の直接計算用（再帰的平均）
        while len(clusters) > 1:
            act = sorted(clusters)
            a, b = min(combinations(act, 2), key=lambda p: Dm[p])
            h = Dm[a, b]
            ni_, nj_ = len(clusters[a]), len(clusters[b])
            for k in act:
                if k in (a, b):
                    continue
                al, aj, be, ga = lw_coef(m, ni_, nj_, len(clusters[k]))
                v = al * Dm[k, a] + aj * Dm[k, b] + be * h + ga * abs(Dm[k, a] - Dm[k, b])
                Dm[k, a] = Dm[a, k] = v
            clusters[a] = clusters[a] + clusters[b]
            reps[a] = (reps[a] + reps[b]) / 2
            del clusters[b], reps[b]
            for k in clusters:
                if k == a:
                    continue
                if m in ("single", "complete", "average", "centroid", "ward"):
                    dirv = direct_dist(m, P, clusters[k], clusters[a])
                elif m == "median":
                    dirv = float(((reps[k] - reps[a]) ** 2).sum())
                else:
                    continue
                ok_lw[m] &= abs(dirv - Dm[k, a]) < 1e-9
        if m in ("weighted", "median"):
            # scipy の weighted / median（median は √ の高さ）と高さを比較
            Zm = linkage(pdist(P), m) if m == "weighted" else linkage(P, "median")
            Smy = lw_cluster(base, m)
            hs = np.array([s[2] for s in Smy])
            ok_lw[m] &= np.allclose(np.sort(Zm[:, 2]), np.sort(hs if m == "weighted" else np.sqrt(hs)), atol=1e-9)
labels7 = {"single": "最短距離法", "complete": "最長距離法", "average": "群平均法", "weighted": "WPGMA",
           "centroid": "重心法", "median": "メディアン法（中点の2乗距離）", "ward": "ウォード法（=2Δ）"}
for m, ok in ok_lw.items():
    extra = "・scipy の高さ" if m in ("weighted", "median") else ""
    check(T, f"係数表 {labels7[m]}: 乱数 9点×40回の全段で更新式 = 直接計算{extra}（許容 1e-9）", ok, "一致" if ok else "不一致")
# 単調性の十分条件
conds = {}
for m in ok_lw:
    good = True
    for ni_, nj_, nk_ in [(1, 1, 1), (2, 3, 1), (5, 1, 4), (3, 3, 3)]:
        al, aj, be, ga = lw_coef(m, ni_, nj_, nk_)
        good &= al >= 0 and aj >= 0 and al + aj + be >= 1 - 1e-12 and ga >= -min(al, aj) - 1e-12
    conds[m] = good
check(T, "単調性の十分条件: 最短・最長・群平均・WPGMA・ウォードは満たし、重心・メディアンは満たさない",
      all(conds[m] for m in ["single", "complete", "average", "weighted", "ward"]) and not conds["centroid"] and not conds["median"], conds)
check(T, "ウォード法の α_i+α_j+β = 1", all(abs(sum(lw_coef("ward", *t)[:3]) - 1) < 1e-12 for t in [(1, 1, 1), (2, 3, 4), (7, 1, 2)]), "1")
# 逆転の有無を乱数で（単調手法は常に単調）
mono = {m: True for m in ["single", "complete", "average", "weighted", "ward"]}
inv = {"centroid": False, "median": False}
for _ in range(200):
    P = rng.normal(size=(8, 2))
    for m in mono:
        Z = linkage(P, m) if m == "ward" else linkage(pdist(P), m)
        mono[m] &= bool(np.all(np.diff(Z[:, 2]) >= -1e-12))
    for m in inv:
        Z = linkage(P, m)
        inv[m] |= bool(np.any(np.diff(Z[:, 2]) < -1e-12))
check(T, "乱数 8点×200回: 最短・最長・群平均・WPGMA・ウォードは逆転なし", all(mono.values()), mono)
check(T, "乱数 8点×200回: 重心法・メディアン法では逆転が起こる", all(inv.values()), inv)
# 例1
ex = [0.5 * 11 + 0.5 * 12 - 0.5 * 1, 0.5 * 11 + 0.5 * 12 + 0.5 * 1, 0.5 * 11 + 0.5 * 12]
check(T, "例1 d(A,DE)=11, 12, 11.5", ex == [11, 12, 11.5] and [tget(Ss[0][3], "A", "DE"), tget(Sc[0][3], "A", "DE"), tget(Sa[0][3], "A", "DE")] == ex, ex)
check(T, "例1 高さの列（最短 2,3,4,5／群平均 2,3,5,8.5／最長 2,3,5,12）",
      allclose(Zs[:, 2], [2, 3, 4, 5], 0) and allclose(Za[:, 2], [2, 3, 5, 8.5], 0) and allclose(Zc[:, 2], [2, 3, 5, 12], 0), "一致")
check(T, "例1 群平均の段3 1/3·5.5+2/3·10=8.5", close(5.5 / 3 + 20 / 3, 8.5, 1e-12), 8.5)
check(T, "例2 重心法 ½10+½10−¼4=9 = (1,0)–(1,3) の2乗距離", close(0.5 * 10 + 0.5 * 10 - 1, 9, 0) and ((Pw[2] - Pw[:2].mean(0)) ** 2).sum() == 9, 9)
check(T, "例2 ウォード法 (2/3)10+(2/3)10−(1/3)4=12 = 2Δ（Δ=2/3·9=6）", close(2 / 3 * 20 - 4 / 3, 12, 1e-12) and close(2 * delta("C", "AB"), 12, 1e-9), 12)
cl = chart(T, "line-data")["params"]
ys = {s["label"]: s["y"] for s in cl["series"]}
check(T, "charts line-data（最短/群平均/最長の高さ）",
      cl["x"] == [1, 2, 3, 4] and allclose(ys["最短距離法"], Zs[:, 2], 0) and allclose(ys["群平均法"], Za[:, 2], 0) and allclose(ys["最長距離法"], Zc[:, 2], 0), ys)

# ---------------------------------------------------------------- dendrogram
T = "dendrogram"
for hcut, k, groups in [(1, 5, ["A", "B", "C", "D", "E"]), (2.5, 4, ["A", "B", "C", "DE"]), (4, 3, ["AB", "C", "DE"]), (8, 2, ["AB", "CDE"])]:
    lab = fcluster(Zc, hcut, criterion="distance")
    g = {}
    for i, l in enumerate(lab):
        g.setdefault(l, []).append(L[i])
    gs = sorted("".join(v) for v in g.values())
    check(T, f"例1 高さ {hcut} で切ると {k} クラスター {groups}", len(gs) == k and gs == sorted(groups), gs)
coph_c = squareform(cophenet(Zc))
check(T, "例2 c_AC=12, c_CD=5, c_DE=2", [coph_c[0, 2], coph_c[2, 3], coph_c[3, 4]] == [12, 5, 2], [coph_c[0, 2], coph_c[2, 3], coph_c[3, 4]])
for name, Z, rc in [("最短距離法", Zs, 0.7214), ("最長距離法", Zc, 0.7638), ("群平均法", Za, 0.7712)]:
    c_, _ = cophenet(Z, dv)
    direct = np.corrcoef(dv, cophenet(Z))[0, 1]
    check(T, f"例2 {name}のコーフェン相関 {rc}（scipy・相関の直接計算、小数第4位）", close(c_, rc) and close(direct, rc), round(c_, 6))
check(T, "例2 群平均法が最大", cophenet(Za, dv)[0] > max(cophenet(Zs, dv)[0], cophenet(Zc, dv)[0]), "OK")
check(T, "組の数 10", len(dv) == 10, len(dv))
okum = True
for Z in (Zs, Zc, Za):
    C = squareform(cophenet(Z))
    for i in range(5):
        for j in range(5):
            for k in range(5):
                okum &= C[i, j] <= max(C[i, k], C[k, j]) + 1e-12
check(T, "単調な木のコーフェン距離は超距離 c_ij ≤ max(c_ik, c_kj)", okum, "成立")
check(T, "葉の並べ方 2^{n−1}（n=5 で 16）: 併合点 n−1 個の左右反転", 2 ** (len(Zc)) == 16, 16)
cdn = chart(T, "dendrogram")["params"]
check(T, "charts dendrogram method=complete、n=12 がスライダー範囲内", cdn["method"] == "complete" and 4 <= cdn["n"] <= 16, cdn)

# ---------------------------------------------------------------- 図の JS アルゴリズム（dendrogram）の移植と scipy の一致
T = "dendrogram（図）"


def js_dendro(P, meth):
    n = len(P)
    D_ = squareform(pdist(P, "sqeuclidean")) if meth == "ward" else squareform(pdist(P))
    active, size, merges = list(range(n)), [1] * n, []
    while len(active) > 1:
        best, bi, bj = math.inf, -1, -1
        for i in active:
            for j in active:
                if i < j and D_[i, j] < best:
                    best, bi, bj = D_[i, j], i, j
        ni_, nj_ = size[bi], size[bj]
        for k in active:
            if k in (bi, bj):
                continue
            nk, dik, djk = size[k], D_[bi, k], D_[bj, k]
            if meth == "single":
                v = min(dik, djk)
            elif meth == "complete":
                v = max(dik, djk)
            elif meth == "average":
                v = (ni_ * dik + nj_ * djk) / (ni_ + nj_)
            else:
                v = ((ni_ + nk) * dik + (nj_ + nk) * djk - nk * best) / (ni_ + nj_ + nk)
            D_[bi, k] = D_[k, bi] = v
        merges.append(math.sqrt(best) if meth == "ward" else best)
        size[bi] = ni_ + nj_
        active = [k for k in active if k != bj]
    return np.array(merges)


okjs = True
for _ in range(50):
    P = rng.normal(size=(rng.integers(4, 17), 2))
    for m in ["single", "complete", "average", "ward"]:
        Z = linkage(P, "ward") if m == "ward" else linkage(pdist(P), m)
        okjs &= np.allclose(np.sort(js_dendro(P, m)), np.sort(Z[:, 2]), atol=1e-9)
check(T, "src の dendrogram 描画アルゴリズム（4手法、ward は √(2Δ)）を移植し乱数 50組で scipy と高さ一致（許容 1e-9）", okjs, "一致")
check(T, "5店舗相当（ward 例の5点）で JS 移植の高さ = 2, 2.236, 3.464, 10.536", allclose(js_dendro(Pw, "ward"), [2, 2.236, 3.464, 10.536], 5e-4), r(js_dendro(Pw, "ward"), 3))
for term, meth in [("hierarchical-clustering", "average"), ("single-linkage", "single"), ("complete-linkage", "complete"), ("average-linkage", "average")]:
    pp = chart(term, "dendrogram")["params"]
    sl = chart(term, "dendrogram")["sliders"][0]
    check(T, f"{term} の charts dendrogram method={meth}、n がスライダー範囲内", pp["method"] == meth and sl["min"] <= pp["n"] <= sl["max"], pp)

# ---------------------------------------------------------------- k-means-method
T = "k-means-method"
K = np.array([[2, 3], [3, 2], [3, 6], [4, 3], [5, 0], [6, 7]], float)
KN = "ABCDEF"


def sqd(X, M):
    return ((X[:, None, :] - M[None, :, :]) ** 2).sum(-1)


def W_of(X, lab, k):
    return sum(((X[lab == j] - X[lab == j].mean(0)) ** 2).sum() for j in range(k) if np.any(lab == j))


M = np.array([[0, 3], [7, 5]], float)
claims = [
    ([[4, 10, 18, 16, 34, 52], [29, 25, 17, 13, 29, 5]], [0, 0, 1, 1, 1, 1], [[2.5, 2.5], [4.5, 4]], 36, 1e-9),
    ([[0.5, 0.5, 12.5, 2.5, 12.5, 32.5], [7.25, 6.25, 6.25, 1.25, 16.25, 11.25]], [0, 0, 1, 1, 0, 1], [[10 / 3, 5 / 3], [13 / 3, 16 / 3]], 22.667, 5e-4),
    ([[3.556, 0.222, 18.889, 2.222, 5.556, 35.556], [10.889, 12.889, 2.222, 5.556, 28.889, 5.556]], [0, 0, 1, 0, 0, 1], [[3.5, 2], [4.5, 6.5]], 16, 5e-4),
]
for it, (dist_c, lab_c, cen_c, W_c, tol) in enumerate(claims, 1):
    dd = sqd(K, M).T
    lab = np.argmin(dd, axis=0)
    check(T, f"例1 反復{it} 2乗距離の表（小数第3位）", allclose(dd, dist_c, 5e-4), r(dd, 3))
    check(T, f"例1 反復{it} 割り当て {lab_c}", lab.tolist() == lab_c, lab.tolist())
    M = np.array([K[lab == j].mean(0) for j in range(2)])
    check(T, f"例1 反復{it} 新しい中心 {r(cen_c, 3)}", allclose(M, cen_c, 1e-9), r(M, 3))
    check(T, f"例1 反復{it} W={W_c}", close(W_of(K, lab, 2), W_c, tol), round(W_of(K, lab, 2), 4))
dd = sqd(K, M).T
check(T, "例1 反復4 2乗距離 3.25/18.5, 0.25/22.5, 16.25/2.5, 1.25/12.5, 6.25/42.5, 31.25/2.5",
      allclose(dd, [[3.25, 0.25, 16.25, 1.25, 6.25, 31.25], [18.5, 22.5, 2.5, 12.5, 42.5, 2.5]], 1e-9), r(dd))
check(T, "例1 反復4 割り当て不変（収束）", np.argmin(dd, 0).tolist() == [0, 0, 1, 0, 0, 1], np.argmin(dd, 0).tolist())
allW = []
for mask in range(1, 2 ** 5 * 2 - 1):
    lab = np.array([(mask >> i) & 1 for i in range(6)])
    if lab[0] == 1 or lab.sum() == 0:
        continue
    allW.append((W_of(K, lab, 2), lab.tolist()))
check(T, "例1 6点の2分割は 31 通り", len(allW) == 31, len(allW))
bestW = min(allW)
check(T, "例1 全31通りの最小 W=16、最適分割 {A,B,D,E},{C,F}（一意）",
      close(bestW[0], 16, 1e-9) and bestW[1] == [0, 0, 1, 0, 0, 1] and sum(abs(w - 16) < 1e-9 for w, _ in allW) == 1, bestW)
# 例2
M = K[[1, 4]].copy()
Ws = []
for _ in range(10):
    lab = np.argmin(sqd(K, M), 1)
    if _ == 0:
        lab0 = lab.copy()
    newM = np.array([K[lab == j].mean(0) for j in range(2)])
    Ws.append(W_of(K, lab, 2))
    if np.allclose(newM, M):
        break
    M = newM
check(T, "例2 初期中心 B,E で反復1は E だけ群2", lab0.tolist() == [0, 0, 0, 0, 1, 0], lab0.tolist())
check(T, "例2 群1の中心 (3.6,4.2)", allclose(K[lab0 == 0].mean(0), [3.6, 4.2], 1e-9), K[lab0 == 0].mean(0).tolist())
check(T, "例2 D の2乗距離 m1 まで 0.16+1.44=1.6、E まで 1+9=10", close(((K[3] - [3.6, 4.2]) ** 2).sum(), 1.6, 1e-9) and ((K[3] - K[4]) ** 2).sum() == 10, "1.6, 10")
check(T, "例2 収束時の割り当ては不変・W=28（局所最適 > 16）", lab.tolist() == lab0.tolist() and close(Ws[-1], 28, 1e-9), Ws)
# W の単調減少（乱数）
okmono = True
for _ in range(100):
    X = rng.normal(size=(30, 2))
    Mc = X[rng.choice(30, 3, replace=False)]
    prev = math.inf
    for it in range(20):
        lab = np.argmin(sqd(X, Mc), 1)
        Wa = ((X - Mc[lab]) ** 2).sum()  # 割り当て後
        okmono &= Wa <= prev + 1e-9
        Mc = np.array([X[lab == j].mean(0) if np.any(lab == j) else Mc[j] for j in range(3)])
        Wu = ((X - Mc[lab]) ** 2).sum()  # 更新後
        okmono &= Wu <= Wa + 1e-9
        prev = Wu
check(T, "乱数 30点×100回で、割り当て・更新のどちらでも W が増えない（許容 1e-9）", okmono, "成立")
m_ = sp.symbols("m", real=True)
xs = sp.symbols("x1:5", real=True)
sol = sp.solve(sp.diff(sum((x - m_) ** 2 for x in xs), m_), m_)
check(T, "Σ(x−m)² を最小にする m は平均（sympy）", sp.simplify(sol[0] - sum(xs) / 4) == 0, sol)
ck = chart(T, "kmeans-steps")
check(T, "charts kmeans-steps k=3,n=90,step=0 がスライダー範囲内",
      all(s["min"] <= ck["params"][s["param"]] <= s["max"] for s in ck["sliders"]) and ck["params"] == {"k": 3, "n": 90, "step": 0}, ck["params"])

# ---------------------------------------------------------------- non-hierarchical-clustering
T = "non-hierarchical-clustering"
lab_opt = np.array([0, 0, 1, 0, 0, 1])
check(T, "例1 k-means の最適分割の中心 (3.5,2),(4.5,6.5)、J=16",
      allclose([K[lab_opt == 0].mean(0), K[lab_opt == 1].mean(0)], [[3.5, 2], [4.5, 6.5]], 1e-12) and close(W_of(K, lab_opt, 2), 16, 1e-9), 16)
DK = squareform(pdist(K))
med = []
for a, b in combinations(range(6), 2):
    J = np.minimum(DK[:, a], DK[:, b]).sum()
    med.append((J, KN[a] + KN[b], np.argmin(np.c_[DK[:, a], DK[:, b]], 1).tolist()))
check(T, "例1 k-medoids の代表点の組 15 通り", len(med) == 15, len(med))
minJ = min(m[0] for m in med)
argm = sorted(m[1] for m in med if abs(m[0] - minJ) < 1e-9)
check(T, "例1 k-medoids の最小 J=√2+√2+√8+√10=8.819、代表 BC または BF",
      close(minJ, 8.819, 5e-4) and close(minJ, 2 * math.sqrt(2) + math.sqrt(8) + math.sqrt(10), 1e-12) and argm == ["BC", "BF"], (round(minJ, 4), argm))
check(T, "例1 k-medoids の分割も {A,B,D,E},{C,F}", all(m[2] in ([0, 0, 1, 0, 0, 1],) for m in med if abs(m[0] - minJ) < 1e-9), "一致")
check(T, "例1 √2,√8,√10 の小数 1.414, 2.828, 3.162", allclose([math.sqrt(2), math.sqrt(8), math.sqrt(10)], [1.414, 2.828, 3.162], 5e-4), "OK")


def stirling2(n, k):
    return sum((-1) ** j * math.comb(k, j) * (k - j) ** n for j in range(k + 1)) // math.factorial(k)


check(T, "S(n,2)=2^{n−1}−1（n=2..15）", all(stirling2(n, 2) == 2 ** (n - 1) - 1 for n in range(2, 16)), "成立")
x1d = np.array([1, 2, 3, 4, 100], float)
sums = [np.abs(x1d - o).sum() for o in x1d]
check(T, "例2 平均 22、距離和を最小にするデータ点 3（和 101）", x1d.mean() == 22 and x1d[np.argmin(sums)] == 3 and min(sums) == 101, sums)
cn = chart(T, "kmeans-steps")
check(T, "charts kmeans-steps k=3,n=90,step=3 がスライダー範囲内",
      all(s["min"] <= cn["params"][s["param"]] <= s["max"] for s in cn["sliders"]), cn["params"])

# ---------------------------------------------------------------- number-of-clusters
T = "number-of-clusters"
N9 = np.array([[1, 1], [2, 1], [1, 2], [5, 5], [6, 5], [5, 6], [9, 1], [9, 2], [8, 1]], float)
D9 = squareform(pdist(N9))


def partitions(n):
    """制限成長列で集合分割を列挙"""
    a = [0] * n

    def rec(i, m):
        if i == n:
            yield list(a)
            return
        for v in range(m + 2):
            a[i] = v
            yield from rec(i + 1, max(m, v))
    a[0] = 0
    yield from rec(1, 0)


def silhouette(Dm, lab):
    lab = np.asarray(lab)
    s = []
    for i in range(len(lab)):
        same = (lab == lab[i])
        if same.sum() == 1:
            s.append(0.0)
            continue
        a = Dm[i, same].sum() / (same.sum() - 1)
        b = min(Dm[i, lab == c].mean() for c in set(lab.tolist()) if c != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


best = {}
count = {}
for p in partitions(9):
    lab = np.array(p)
    k = lab.max() + 1
    if k > 6:
        continue
    count[k] = count.get(k, 0) + 1
    w = W_of(N9, lab, k)
    if k not in best or w < best[k][0] - 1e-9:
        best[k] = (w, [lab])
    elif abs(w - best[k][0]) <= 1e-9:
        best[k][1].append(lab)
check(T, "分割の総数 S(9,k)（k=1..6）が総当たり件数と一致", all(count[k] == stirling2(9, k) for k in range(1, 7)), count)
Wk = [best[k][0] for k in range(1, 7)]
check(T, "例1 W(k)=116.889, 44.667, 4, 3.167, 2.333, 1.5（小数第3位）", allclose(Wk, [116.889, 44.667, 4.0, 3.167, 2.333, 1.5], 5e-4), r(Wk, 3))
check(T, "例1 減少 72.222, 40.667, 0.833×3", allclose(-np.diff(Wk), [72.222, 40.667, 0.833, 0.833, 0.833], 5e-4), r(-np.diff(Wk), 3))
sil = {}
for k in range(2, 7):
    vals = sorted(set(round(silhouette(D9, l), 10) for l in best[k][1]))
    sil[k] = vals
sil_claim = {2: 0.5668, 3: 0.7872, 4: 0.5571, 5: 0.3301, 6: 0.0976}
for k in range(2, 7):
    nopt = len(best[k][1])
    # W 最適分割が複数ある k（4, 5）は本文の注記どおり、その中の最大値と比べる
    ok = close(max(sil[k]), sil_claim[k]) and (len(sil[k]) == 1 or k in (4, 5))
    check(T, f"例1 k={k} の平均シルエット {sil_claim[k]}（W 最適分割 {nopt} 個・値 {len(sil[k])} 種、最大値と比較、小数第4位）", ok, [round(v, 6) for v in sil[k]])
check(T, "本文の注記: W 最適分割が複数で値が分かれるのは k=4,5 のみ", [k for k in sil if len(sil[k]) > 1] == [4, 5], {k: len(v) for k, v in sil.items()})
check(T, "例1 シルエット最大は k=3（どの最適分割を選んでも）", max(sil_claim, key=sil_claim.get) == 3 and min(sil[3]) > max(max(v) for k, v in sil.items() if k != 3), 3)
lab3 = best[3][1][0]
same = lab3 == lab3[0]
a0 = D9[0, same].sum() / (same.sum() - 1)
bs = sorted(D9[0, lab3 == c].mean() for c in set(lab3.tolist()) if c != lab3[0])
check(T, "例2 a=1、他クラスターまでの平均 6.154, 7.687", close(a0, 1, 1e-12) and allclose(bs, [6.154, 7.687], 5e-4), r(bs, 4))
check(T, "例2 距離 √32,√41,√41 = 5.657, 6.403 と 8, √65=8.062, 7", allclose([math.sqrt(32), math.sqrt(41), math.sqrt(65)], [5.657, 6.403, 8.062], 5e-4), "OK")
s0 = (bs[0] - a0) / bs[0]
check(T, "例2 s=(6.154−1)/6.154≈0.8375", close(s0, 0.8375), round(s0, 6))
cl1 = chart(T, "line-data")["params"]
check(T, "charts line-data の W(k)", cl1["x"] == [1, 2, 3, 4, 5, 6] and allclose(cl1["series"][0]["y"], Wk, 5e-4), cl1["series"][0]["y"])
cb = chart(T, "bar-data")["params"]
check(T, "charts bar-data の平均シルエット", cb["labels"] == [f"k={k}" for k in range(2, 7)] and allclose(cb["values"], [max(sil[k]) for k in range(2, 7)], 5e-5), cb["values"])
check(T, "W(n)=0（k=n で各点1個）", W_of(N9, np.arange(9), 9) == 0, 0)
# カリンスキー・ハラバズ指数 = F 統計量の形（B+W=全平方和）
lab = best[3][1][0]
Wt = W_of(N9, lab, 3)
Bt = sum((lab == j).sum() * ((N9[lab == j].mean(0) - N9.mean(0)) ** 2).sum() for j in range(3))
check(T, "B(k)+W(k)=全平方和（k=3）", close(Bt + Wt, Wk[0], 1e-9), round(Bt + Wt, 4))

# ---------------------------------------------------------------- distance-measures
T = "distance-measures"
xa, ya = np.array([1, 5.0]), np.array([4, 1.0])
mk = lambda q: (np.abs(xa - ya) ** q).sum() ** (1 / q)
check(T, "例1 マンハッタン 7、ユークリッド 5、チェビシェフ 4、2乗 25",
      mk(1) == 7 and close(mk(2), 5, 1e-12) and np.abs(xa - ya).max() == 4 and ((xa - ya) ** 2).sum() == 25, "OK")
check(T, "例1 q=3: 91^{1/3}=4.4979、q=4: 337^{1/4}=4.2846", close(mk(3), 4.4979) and close(mk(4), 4.2846), r([mk(3), mk(4)]))
qs = [1, 1.5, 2, 3, 4, 6, 10]
vals = [mk(q) for q in qs]
cm = chart(T, "line-data")["params"]
check(T, "charts line-data のミンコフスキー距離（小数第4位）", cm["x"] == qs and allclose(cm["series"][0]["y"], vals, 5e-5), r(vals))
qq = np.linspace(1, 60, 500)
check(T, "q について単調減少し 4 に収束", np.all(np.diff([mk(q) for q in qq]) < 0) and abs(mk(200) - 4) < 0.01, round(mk(200), 4))
check(T, "scipy minkowski/cityblock/chebyshev と一致",
      close(pdist([xa, ya], "minkowski", p=3)[0], mk(3), 1e-12) and pdist([xa, ya], "cityblock")[0] == 7 and pdist([xa, ya], "chebyshev")[0] == 4, "OK")
check(T, "2乗ユークリッドは三角不等式を満たさない（0,1,2 で 4>1+1）", (2 - 0) ** 2 > (1 - 0) ** 2 + (2 - 1) ** 2, "4 > 2")
# q<1 で三角不等式が破れる例
q = 0.5
f = lambda u, v: (np.abs(np.subtract(u, v)) ** q).sum() ** (1 / q)
check(T, "q=0.5 で三角不等式が破れる（(0,0),(1,0),(1,1)）", f((0, 0), (1, 1)) > f((0, 0), (1, 0)) + f((1, 0), (1, 1)), f((0, 0), (1, 1)))
pA, pB, pC = np.array([170, 60.0]), np.array([180, 60.0]), np.array([170, 63.0])
dcm = (np.linalg.norm(pA - pB), np.linalg.norm(pA - pC))
dm = (np.linalg.norm((pA - pB) * [0.01, 1]), np.linalg.norm((pA - pC) * [0.01, 1]))
dst = (np.linalg.norm((pA - pB) / [6, 8]), np.linalg.norm((pA - pC) / [6, 8]))
check(T, "例2 cm: 10, 3 → C", close(dcm[0], 10, 1e-12) and close(dcm[1], 3, 1e-12) and dcm[1] < dcm[0], dcm)
check(T, "例2 m: 0.1, 3 → B", close(dm[0], 0.1, 1e-12) and close(dm[1], 3, 1e-12) and dm[0] < dm[1], dm)
check(T, "例2 標準化: 1.667, 0.375 → C", close(dst[0], 1.667, 5e-4) and close(dst[1], 0.375, 1e-12) and dst[1] < dst[0], r(dst, 4))
cmb = chart(T, "minkowski-ball")
check(T, "charts minkowski-ball p=2 がスライダー範囲内", cmb["sliders"][0]["min"] <= cmb["params"]["p"] <= cmb["sliders"][0]["max"], cmb["params"])
# 乱数で三角不等式（q≥1）と単調性
okt = True
for _ in range(2000):
    u, v, w_ = rng.normal(size=(3, 4))
    for q in (1, 1.5, 2, 3, 7):
        g = lambda s, t: (np.abs(s - t) ** q).sum() ** (1 / q)
        okt &= g(u, w_) <= g(u, v) + g(v, w_) + 1e-12
    okt &= g(u, v) <= (np.abs(u - v) ** 2).sum() ** 0.5 + 1e-12 if True else True
check(T, "乱数 2000組で q≥1 の三角不等式と d_7 ≤ d_2（許容 1e-12）", okt, "成立")

# ---------------------------------------------------------------- similarity-measures
T = "similarity-measures"
xb_ = np.array([1, 1, 0, 1, 0, 0, 1, 0])
yb_ = np.array([1, 0, 0, 1, 1, 0, 1, 0])
a = int(((xb_ == 1) & (yb_ == 1)).sum())
b = int(((xb_ == 1) & (yb_ == 0)).sum())
c = int(((xb_ == 0) & (yb_ == 1)).sum())
d = int(((xb_ == 0) & (yb_ == 0)).sum())
check(T, "例1 a=3, b=1, c=1, d=3", (a, b, c, d) == (3, 1, 1, 3), (a, b, c, d))
smc, jac, dice = (a + d) / 8, a / (a + b + c), 2 * a / (2 * a + b + c)
check(T, "例1 単純一致 0.75、ジャッカード 0.6、ダイス 0.75", (smc, jac, dice) == (0.75, 0.6, 0.75), (smc, jac, dice))
from scipy.spatial.distance import jaccard, dice as dice_d, hamming
check(T, "scipy の jaccard・dice・hamming（1−類似度）と一致",
      close(1 - jaccard(xb_.astype(bool), yb_.astype(bool)), jac, 1e-12) and close(1 - dice_d(xb_.astype(bool), yb_.astype(bool)), dice, 1e-12)
      and close(1 - hamming(xb_, yb_), smc, 1e-12), "OK")
check(T, "D=2J/(1+J)", close(2 * jac / (1 + jac), dice, 1e-12), 2 * jac / (1 + jac))
check(T, "商品1000で d=995: 単純一致 0.998、ジャッカード 0.6 のまま", close((3 + 995) / 1000, 0.998, 1e-12) and 3 / 5 == 0.6, "0.998, 0.6")
u, v = np.array([3, 0, 1, 2.0]), np.array([1, 1, 0, 2.0])
cos = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
check(T, "例2 コサイン 7/√84≈0.7638", close(cos, 0.7638) and close(cos, 7 / math.sqrt(84), 1e-12) and u @ u == 14 and v @ v == 6, round(cos, 6))
uc, vc = u - u.mean(), v - v.mean()
corr = np.corrcoef(u, v)[0, 1]
check(T, "例2 平均 1.5, 1、中心化 (1.5,−1.5,−0.5,0.5),(0,0,−1,1)", u.mean() == 1.5 and v.mean() == 1 and uc.tolist() == [1.5, -1.5, -0.5, 0.5] and vc.tolist() == [0, 0, -1, 1], "OK")
check(T, "例2 相関 1/√10≈0.3162（np.corrcoef と一致）", close(corr, 0.3162) and close(corr, 1 / math.sqrt(10), 1e-12) and uc @ uc == 5 and vc @ vc == 2, round(corr, 6))
check(T, "例2 2u でもコサイン不変", close((2 * u) @ v / np.linalg.norm(2 * u) / np.linalg.norm(v), cos, 1e-12), "不変")
cs_ = chart(T, "bar-data")["params"]
check(T, "charts bar-data [0.75, 0.6, 0.75]", allclose(cs_["values"], [smc, jac, dice], 1e-12), cs_["values"])
# 乱数: 相関 = 中心化ベクトルのコサイン
okc = True
for _ in range(500):
    s_, t_ = rng.normal(size=(2, 6))
    sc, tc = s_ - s_.mean(), t_ - t_.mean()
    okc &= abs(np.corrcoef(s_, t_)[0, 1] - sc @ tc / np.linalg.norm(sc) / np.linalg.norm(tc)) < 1e-12
check(T, "乱数 500組で 相関係数 = 中心化ベクトルのコサイン（許容 1e-12）", okc, "成立")

# ---------------------------------------------------------------- 集計
print()
print(f"OK {sum(results)} / NG {len(results) - sum(results)}")
