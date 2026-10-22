---
id: eigenvalue-and-eigenvector
title: 固有値と固有ベクトル
reading: こゆうちとこゆうべくとる
en: Eigenvalue and eigenvector
aliases: [固有値, 固有ベクトル, 固有方程式, スペクトル分解, 固有値分解]
chapter: 22
importance: A
tags: [線形代数, 固有値問題, 対称行列, 主成分分析]
related: [principal-component-analysis, lagrange-multiplier-pca, singular-value-decomposition, variance-covariance-matrix, correlation-matrix, multicollinearity]
charts:
  - type: custom
    id: pca-axes
    params: { rho: 0.5, sx: 1, sy: 1, n: 200 }
    sliders:
      - { param: rho, label: "相関係数 ρ", min: -0.95, max: 0.95, step: 0.05 }
      - { param: sx, label: "x1 の標準偏差", min: 0.5, max: 3, step: 0.1 }
    caption: 標本分散共分散行列の固有ベクトルの向きに軸を描いた図。sx=sy=1 のとき、固有値は 1±ρ、固有ベクトルは 45 度方向になる（標本のゆらぎで少しずれる）。
---

## 一言でいうと

行列を掛けても向きが変わらず、長さだけが何倍かになる特別なベクトルが固有ベクトルで、その倍率が固有値です。
分散共分散行列では、固有ベクトルが「データの広がる主な方向」、固有値が「その方向の分散」を表します。

## 定義・数式

$p\times p$ 行列 $A$ について、

$$
A\boldsymbol{h}=\lambda\boldsymbol{h},\qquad \boldsymbol{h}\ne\boldsymbol{0}
$$

を満たすスカラー $\lambda$ を**固有値**、ベクトル $\boldsymbol{h}$ を**固有ベクトル**といいます。
$(A-\lambda I)\boldsymbol{h}=\boldsymbol{0}$ が $\boldsymbol{0}$ 以外の解を持つ条件から、固有値は**固有方程式**

$$
\det(A-\lambda I)=0
$$

の解です。$A$ が実対称行列（分散共分散行列・相関行列はこれに当たる）のとき、次が成り立ちます。

- 固有値はすべて実数。非負定値行列なら $\lambda_k\ge0$。
- 異なる固有値に属する固有ベクトルは直交する。長さ1にそろえた固有ベクトルを並べた $H$ は直交行列（$H^\top H=I$）にとれる。
- **スペクトル分解**：$A=H\Lambda H^\top=\sum_{k=1}^p\lambda_k\boldsymbol{h}_k\boldsymbol{h}_k^\top$、ただし $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_p)$。
- $\mathrm{tr}\,A=\sum_k\lambda_k$、$\det A=\prod_k\lambda_k$。

## 数式の読み解き

2×2 の対称行列 $A=\begin{pmatrix}a&b\\b&c\end{pmatrix}$ で、固有方程式を展開します。

$$
\det\begin{pmatrix}a-\lambda&b\\b&c-\lambda\end{pmatrix}=(a-\lambda)(c-\lambda)-b^2=\lambda^2-(a+c)\lambda+(ac-b^2)=0
$$

$\lambda$ の係数がトレース、定数項が行列式です。解の公式から

$$
\lambda=\frac{(a+c)\pm\sqrt{(a-c)^2+4b^2}}{2}
$$

で、根号の中が非負なので必ず実数になります。
固有ベクトルは $(a-\lambda)h_1+bh_2=0$ から $\boldsymbol{h}\propto(b,\ \lambda-a)^\top$ で、最後に長さ1に割ります。

特に相関行列 $\begin{pmatrix}1&r\\r&1\end{pmatrix}$ では $\lambda=1\pm r$、固有ベクトルは $(1,1)^\top/\sqrt2$ と $(1,-1)^\top/\sqrt2$ です。
$r$ の値によらず、主成分の向きは必ず45度になります。

直交性の理由も示しておきます。$A\boldsymbol{h}_1=\lambda_1\boldsymbol{h}_1$、$A\boldsymbol{h}_2=\lambda_2\boldsymbol{h}_2$ とすると、$A$ が対称なので

$$
\lambda_1\boldsymbol{h}_2^\top\boldsymbol{h}_1=\boldsymbol{h}_2^\top A\boldsymbol{h}_1=(A\boldsymbol{h}_2)^\top\boldsymbol{h}_1=\lambda_2\boldsymbol{h}_2^\top\boldsymbol{h}_1
$$

$\lambda_1\ne\lambda_2$ なら $\boldsymbol{h}_2^\top\boldsymbol{h}_1=0$ です。

最後に、主成分分析で最大固有値が出てくる理由です。長さ1のベクトル $\boldsymbol{a}$ を固有ベクトルで $\boldsymbol{a}=\sum_k c_k\boldsymbol{h}_k$（$\sum_k c_k^2=1$）と展開すると、

$$
\boldsymbol{a}^\top A\boldsymbol{a}=\sum_{k=1}^{p}\lambda_k c_k^2\le\lambda_1\sum_{k=1}^{p}c_k^2=\lambda_1
$$

です。等号は $\boldsymbol{a}=\pm\boldsymbol{h}_1$ のときに成り立ちます。「方向を変えたときの分散の最大値が最大固有値」であることが、ラグランジュ乗数法を使わずにも確かめられます。

## 具体例

**例1：2×2 の分散共分散行列を手計算で**

$S=\begin{pmatrix}5&2\\2&2\end{pmatrix}$ の固有方程式は

$$
(5-\lambda)(2-\lambda)-2^2=\lambda^2-7\lambda+6=(\lambda-6)(\lambda-1)=0
$$

で、$\lambda_1=6,\ \lambda_2=1$ です。確認：和 $7=\mathrm{tr}\,S$、積 $6=\det S=10-4$。

- $\lambda_1=6$：$(5-6)h_1+2h_2=0$ より $h_1=2h_2$。$\boldsymbol{h}_1=(2,1)^\top/\sqrt5\approx(0.894,\ 0.447)^\top$。
- $\lambda_2=1$：$(5-1)h_1+2h_2=0$ より $h_2=-2h_1$。$\boldsymbol{h}_2=(1,-2)^\top/\sqrt5\approx(0.447,\ -0.894)^\top$。

内積は $(2-2)/5=0$ で直交しています。スペクトル分解で元に戻すと

$$
6\cdot\frac{1}{5}\begin{pmatrix}4&2\\2&1\end{pmatrix}+1\cdot\frac{1}{5}\begin{pmatrix}1&-2\\-2&4\end{pmatrix}=\frac15\begin{pmatrix}25&10\\10&10\end{pmatrix}=\begin{pmatrix}5&2\\2&2\end{pmatrix}
$$

**例2：同じデータを相関行列にすると**

例1の $S$ から相関係数は $r=2/\sqrt{5\cdot2}=2/\sqrt{10}\approx0.6325$ です。相関行列の固有値は $1\pm r$ で $1.6325,\ 0.3675$、固有ベクトルは $(1,1)^\top/\sqrt2$ と $(1,-1)^\top/\sqrt2$ です。
共分散行列のときの向き（$x_1$ 軸から約26.57度）とは違い、45度になります。第1固有値の割合も $6/7\approx0.857$ から $1.6325/2\approx0.816$ に変わります。

**例3：4×4 の相関行列（numpy）**

4教科のテストの相関行列（主成分分析の項目）を `numpy.linalg.eigh` で分解すると、固有値は $3.2501,\ 0.6287,\ 0.1012,\ 0.0200$ で、和は $4.0000=\mathrm{tr}\,R$ です。
積は $\det R\approx0.0041$ と小さく、変数どうしの相関が強い（多重共線性に近い）ことを示しています。
なお `eigh` は固有値を小さい順に返すので、主成分分析では大きい順に並べ替えます。

## 現場での使われ方

- **製造・品質管理**：多数の測定項目の相関行列の固有値を調べ、ほぼ0の固有値があれば「ほぼ一次従属な測定項目の組」があると判断し、測定項目を整理します。
- **金融・保険**：資産の収益率の共分散行列の固有値分解から、市場全体の動き（最大固有値）とそれ以外の要因を分け、分散投資の効果を評価します。
- **IT・Webサービス**：リンク構造を表す行列の最大固有値に対応する固有ベクトルから、ページの重要度を計算する考え方が知られています。

## グラフ・イメージ

図の2本の軸が固有ベクトルの向き、軸の長さが固有値の平方根（標準偏差）に比例します。
$x_1$ の標準偏差を1のままにして $\rho$ を動かすと、軸の向きは45度のまま長さだけが変わります（相関行列の固有値 $1\pm\rho$）。
$x_1$ の標準偏差を変えると、軸の向き自体が変わります。

## 試験のポイント

- 2×2 は「$\lambda^2-(\mathrm{tr})\lambda+\det=0$」で即座に解けるようにします。
- 固有ベクトルは長さ1に正規化してから使います。主成分の分散や負荷量の計算で効いてきます。
- $\mathrm{tr}=\sum\lambda_k$（総分散）と $\det=\prod\lambda_k$（一般化分散）は検算に使えます。
- 相関行列の 2×2 では $\lambda=1\pm r$、向きは45度、という結果は覚えておく価値があります。
- 最小固有値が0に近いことは、多重共線性の兆候です。

## よくある間違い

- × 固有ベクトルは1つに決まる → ○ 定数倍も固有ベクトルです。長さ1に正規化しても符号 $\pm$ の自由が残ります。
- × 固有方程式を $\det(A)-\lambda=0$ とする → ○ 対角成分すべてから $\lambda$ を引いた $\det(A-\lambda I)=0$ です。
- × 固有値を大きい順に返すと思い込む → ○ ソフトの関数によって昇順・降順が違います。主成分は必ず大きい順に並べ直します。
