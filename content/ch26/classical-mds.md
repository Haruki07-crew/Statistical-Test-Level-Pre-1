---
id: classical-mds
title: 古典的MDS
reading: こてんてきえむでぃーえす
en: Classical multidimensional scaling
aliases: [計量MDS, 主座標分析, PCoA, トーガソンの方法]
chapter: 26
importance: B
tags: [多次元尺度構成法, 二重中心化, 固有値分解, 距離行列]
related: [multidimensional-scaling, eigenvalue-and-eigenvector, principal-component-score, singular-value-decomposition, distance-measures, nonmetric-mds]
charts:
  - type: custom
    id: bar-data
    params:
      labels: [λ1, λ2, λ3]
      values: [8.0826, 0, -1.0826]
      xlabel: 固有値の番号
      ylabel: 内積行列 B の固有値
    caption: 例2（三角不等式を破る3点）の内積行列 B の固有値。負の固有値が現れ、この距離は平面にも空間にも正確には置けないことがわかる。
---

## 一言でいうと

距離の表を「2乗して二重中心化」して内積の表に直し、その固有値分解から座標を求める多次元尺度構成法です。
反復計算が不要で、答えが一つに決まります。

## 定義・数式

$n\times n$ の距離行列 $D=(d_{ij})$ に対して、

$$
A=\Bigl(-\tfrac12 d_{ij}^2\Bigr),\qquad J=I_n-\tfrac1n\boldsymbol{1}\boldsymbol{1}^\top,\qquad B=JAJ
$$

とおきます。成分で書くと

$$
b_{ij}=a_{ij}-\bar a_{i\cdot}-\bar a_{\cdot j}+\bar a_{\cdot\cdot}
$$

です。$B=E\Lambda E^\top$（$\lambda_1\ge\dots\ge\lambda_n$）と固有値分解し、正の固有値のうち大きい $k$ 個を使って

$$
X_k=E_k\Lambda_k^{1/2}
$$

を $k$ 次元の座標とします。

**ユークリッド性の判定**：$D$ がある次元のユークリッド空間の点の距離になっているための必要十分条件は、$B$ が半正定値（固有値がすべて $0$ 以上）であることです。そのとき必要な次元は正の固有値の個数です。

## 数式の読み解き

点 $\boldsymbol{x}_1,\dots,\boldsymbol{x}_n$ の重心が原点にあるとします。距離の2乗を展開すると

$$
d_{ij}^2=\|\boldsymbol{x}_i\|^2+\|\boldsymbol{x}_j\|^2-2\,\boldsymbol{x}_i^\top\boldsymbol{x}_j
$$

です。$c_i=\|\boldsymbol{x}_i\|^2$、$\bar c=\frac1n\sum c_i$ と書き、$\sum_j\boldsymbol{x}_j=\boldsymbol{0}$ を使って平均を取ると

$$
\bar a_{i\cdot}=-\tfrac12(c_i+\bar c),\quad \bar a_{\cdot j}=-\tfrac12(c_j+\bar c),\quad \bar a_{\cdot\cdot}=-\bar c
$$

これらを $b_{ij}$ の式に代入すると $c_i, c_j, \bar c$ がすべて打ち消し合い、

$$
b_{ij}=\boldsymbol{x}_i^\top\boldsymbol{x}_j,\qquad B=XX^\top
$$

が残ります。内積行列がわかれば、$B=E\Lambda E^\top=(E\Lambda^{1/2})(E\Lambda^{1/2})^\top$ なので $X=E\Lambda^{1/2}$ と取れます。

- $J\boldsymbol{1}=\boldsymbol{0}$ なので、$B$ は必ず固有値 $0$（固有ベクトル $\boldsymbol{1}$）を持ちます。重心を原点に固定した分の自由度です。
- 座標行列 $X$ のユークリッド距離から始めると、$B=X_cX_c^\top$（$X_c$ は中心化行列）となり、$X_c$ の特異値分解の左側と一致します。これが「古典的MDS＝主成分得点」の理由です。

## 具体例

**例1：一直線に並ぶ4駅**

道のりが $d_{AB}=2,\ d_{AC}=5,\ d_{AD}=9,\ d_{BC}=3,\ d_{BD}=7,\ d_{CD}=4$ の4駅です。$A=(-d_{ij}^2/2)$ の第1行は $(0,-2,-12.5,-40.5)$ で、二重中心化すると

$$
B=\begin{pmatrix}16&8&-4&-20\\8&4&-2&-10\\-4&-2&1&5\\-20&-10&5&25\end{pmatrix}
$$

になります。これは $\boldsymbol{x}=(4,2,-1,-5)^\top$ を使って $B=\boldsymbol{x}\boldsymbol{x}^\top$ と書けるので、固有値は $46,0,0,0$ です。
正の固有値は1個なので1次元で完全に再現でき、座標は $\sqrt{46}\,\boldsymbol{e}_1=\pm(4,2,-1,-5)$ です。差を取ると元の道のりがすべて戻ります。トレース $16+4+1+25=46$ は $\sum_{i<j}d_{ij}^2/n=184/4$ とも一致します。

**例2：三角不等式を破る3点**

$d_{12}=1,\ d_{13}=2,\ d_{23}=4$ とします（$1+2<4$）。二重中心化すると

$$
B=\begin{pmatrix}-0.667&0.833&-0.167\\0.833&3.333&-4.167\\-0.167&-4.167&4.333\end{pmatrix}
$$

固有値は $8.083,\ 0,\ -1.083$ です。対角成分 $b_{11}=\|\boldsymbol{x}_1\|^2$ が負になっている時点で、内積行列にはなりえません。
1次元の座標は $\sqrt{8.083}\,\boldsymbol{e}_1\approx(-0.220,\ -1.891,\ 2.111)$ で、当てはまりの割合は $8.083/8.083=1$（正の固有値だけで測る流儀）、$8.083/(8.083+1.083)\approx0.882$（絶対値で測る流儀）です。

## 現場での使われ方

- **生態学**：地点間の種組成の非類似度（ブレイ・カーティス指数など）から「主座標分析」として布置を描きます。ユークリッド距離でない指数を使うと負の固有値が出ることがあり、その大きさを確認します。
- **遺伝学**：個体間の遺伝的距離から主座標を計算し、集団の構造（地域ごとのまとまり）を可視化します。
- **マーケティング**：商品間の非類似度の評価から位置づけ地図を作る最初の手順として使い、非計量MDSの初期値にもします。

## グラフ・イメージ

例2の内積行列 $B$ の固有値を並べた棒グラフです。
3番目の固有値が負になっていることに注目します。負の固有値の絶対値が正の固有値に比べて小さければ、ユークリッド距離からのずれは軽いと判断できます。

## 試験のポイント

- 二重中心化の式 $b_{ij}=a_{ij}-\bar a_{i\cdot}-\bar a_{\cdot j}+\bar a_{\cdot\cdot}$ と、$B=XX^\top$ になる導出を押さえます。
- 座標は $\sqrt{\lambda_m}\,\boldsymbol{e}_m$ です。固有値は「その次元の座標の2乗和」に等しくなります。
- 「$B$ が半正定値 ⇔ ユークリッド距離」と、「正の固有値の個数＝必要な次元」をセットで覚えます。
- ユークリッド距離に対する古典的MDSは主成分分析と同値です。

## よくある間違い

- × 距離を2乗せずに中心化する → ○ $-\frac12 d_{ij}^2$ を中心化します。2乗しないと内積になりません。
- × 固有値 $0$ が出たら計算ミスだと考える → ○ $J\boldsymbol{1}=\boldsymbol{0}$ なので、固有値 $0$ は必ず1つ以上現れます。
- × 古典的MDSなら距離の値が完全に再現される → ○ 再現されるのは $B$ が半正定値で、正の固有値の次元をすべて使うときだけです。
