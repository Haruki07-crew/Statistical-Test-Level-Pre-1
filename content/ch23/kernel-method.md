---
id: kernel-method
title: カーネル法
reading: かーねるほう
en: Kernel method
aliases: [カーネルトリック, カーネル関数, ガウスカーネル, RBFカーネル, 多項式カーネル]
chapter: 23
importance: C
tags: [機械学習, 非線形判別, 内積]
related: [support-vector-machine, linear-discriminant-analysis, principal-component-analysis, ridge-regression, k-nearest-neighbors, cross-validation]
---

## 一言でいうと

データを高い次元の空間に写してから線形の手法を使い、元の空間では曲がった境界や関係を表す方法です。
写した先の座標を直接計算せず、2点の「内積」だけをカーネル関数で計算するのが要点です。

## 定義・数式

入力 $\boldsymbol{x}$ を特徴空間へ写す写像を $\phi(\boldsymbol{x})$ とします。**カーネル関数**は特徴空間での内積

$$
K(\boldsymbol{x},\boldsymbol{y})=\phi(\boldsymbol{x})^\top\phi(\boldsymbol{y})
$$

です。$K$ が対称で、どの点の組 $\boldsymbol{x}_1,\dots,\boldsymbol{x}_n$ についても行列 $(K(\boldsymbol{x}_i,\boldsymbol{x}_j))_{ij}$（グラム行列）が半正定値なら、このような $\phi$ が存在します（マーサーの定理）。代表例は次のとおりです。

$$
\begin{aligned}
&\text{多項式カーネル：}K(\boldsymbol{x},\boldsymbol{y})=(\boldsymbol{x}^\top\boldsymbol{y}+c)^d\\
&\text{ガウスカーネル：}K(\boldsymbol{x},\boldsymbol{y})=\exp\!\left(-\frac{\|\boldsymbol{x}-\boldsymbol{y}\|^2}{2\sigma^2}\right)
\end{aligned}
$$

ガウスカーネルは $\exp(-\gamma\|\boldsymbol{x}-\boldsymbol{y}\|^2)$ と、$\gamma=1/(2\sigma^2)$ で書く流儀もあります。

## 数式の読み解き

- サポートベクターマシンの双対問題や、主成分分析・リッジ回帰の一部は、データを内積 $\boldsymbol{x}_i^\top\boldsymbol{x}_j$ の形でしか使わない書き方ができます。
- その内積を $K(\boldsymbol{x}_i,\boldsymbol{x}_j)$ に置き換えるだけで、特徴空間での線形手法になります。これを**カーネルトリック**といいます。
- ガウスカーネルの特徴空間は無限次元ですが、計算するのは $n\times n$ のグラム行列だけで済みます。
- ガウスカーネルの値は、2点が近いと1に近く、遠いと0に近づきます。$\sigma$ が小さいほど「近い」の範囲が狭くなり、境界が細かく曲がります。

## 具体例

**多項式カーネルが内積になっていることの確認**

$\boldsymbol{x}=(1,2)^\top$、$\boldsymbol{y}=(3,1)^\top$、$K(\boldsymbol{x},\boldsymbol{y})=(\boldsymbol{x}^\top\boldsymbol{y})^2$ とします。$\boldsymbol{x}^\top\boldsymbol{y}=3+2=5$ なので $K=25$ です。
一方、$\phi(\boldsymbol{x})=(x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2)^\top$ と写すと、$\phi(\boldsymbol{x})=(1,\ 2\sqrt2,\ 4)$、$\phi(\boldsymbol{y})=(9,\ 3\sqrt2,\ 1)$ で、

$$
\phi(\boldsymbol{x})^\top\phi(\boldsymbol{y})=9+12+4=25
$$

と一致します。3次元の座標を作らなくても、2次元の内積を2乗するだけで同じ値が出ます。

同じ2点のガウスカーネルは、$\|\boldsymbol{x}-\boldsymbol{y}\|^2=4+1=5$ より、$\sigma=1$ で $e^{-2.5}\approx0.0821$、$\sigma=2$ で $e^{-0.625}\approx0.5353$ です。

直感的な例として、1次元で $-0.5,\ 0.5$ が群A、$-2,\ 2$ が群Bなら、1本のしきい値では分けられません。$\phi(x)=(x,\ x^2)$ と写すと、$x^2=1$ という直線で分けられます。

## 現場での使われ方

- **IT・Webサービス**：文章や画像の分類で、線形では分けにくいデータにガウスカーネルのSVMを使います。
- **製造・品質管理**：正常データだけから「正常の範囲」をカーネルで学習し、外れたものを異常とみなす異常検知に使います。
- **医療**：多数の検査値から疾患の有無を判別するとき、非線形な境界の候補として交差検証で比較します。

## 試験のポイント

- 「内積だけで書ける手法 → 内積をカーネルに置き換える」という流れを説明できるようにします。
- 多項式カーネルを展開して $\phi$ を求め、内積と一致することを確かめる計算は重要です。
- $\sigma$（または $\gamma$）や次数 $d$ は、交差検証で選ぶハイパーパラメータです。

## よくある間違い

- × カーネル法では高次元の座標 $\phi(\boldsymbol{x})$ を実際に計算する → ○ 計算するのはカーネルの値（グラム行列）だけです。
- × どんな2変数関数でもカーネルに使える → ○ グラム行列が常に半正定値になる関数でなければなりません。
- × $\sigma$ を小さくするほど性能が上がる → ○ 小さすぎると訓練データに合わせすぎて過学習になります。
