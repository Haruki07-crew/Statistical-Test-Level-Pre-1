---
id: conditional-probability-density-function
title: 条件付き確率密度関数
reading: じょうけんつきかくりつみつどかんすう
en: Conditional probability density function
aliases: [条件付き密度関数]
chapter: 2
importance: A
tags: [多変量, 連続型, 条件付き分布, 条件付き期待値, 2変量正規分布]
related: [joint-probability-density-function, marginal-probability-density-function, conditional-probability-function, conditional-expectation, conditional-variance, bivariate-normal-distribution]
charts:
  - type: pdf
    dist: norm
    params: { mu: 64.8, sigma: 6.4 }
    sliders:
      - { param: mu, label: "条件付き平均", min: 50, max: 75, step: 0.4 }
      - { param: sigma, label: "条件付き標準偏差", min: 2, max: 10, step: 0.2 }
    x: { min: 30, max: 95 }
    compare:
      - { dist: norm, params: { mu: 60, sigma: 8 }, label: "周辺分布 N(60, 8²)" }
    caption: 実線は身長176cmの人の体重の条件付き密度、破線は全体の体重の周辺密度。条件を付けると中心がずれ、幅が狭くなる。
  - type: scatter
    model: bvnorm
    params: { rho: 0.6, n: 500 }
    sliders:
      - { param: rho, label: "相関係数 ρ", min: -0.95, max: 0.95, step: 0.05 }
    caption: x をある値に固定して縦に細く切った帯の中の点の分布が、条件付き分布。|ρ| が大きいほど帯の中の点の散らばりが小さい。
---

## 一言でいうと

一方の連続型確率変数の値がわかったときの、もう一方の変数の密度です。
同時密度の「切り口」を、面積が1になるように割り直したものです。

## 定義・数式

連続型確率変数 $(X,Y)$ の同時確率密度関数を $f(x,y)$、$X$ の周辺確率密度関数を $f_X(x)$ とします。
$f_X(x)>0$ である $x$ について、$X=x$ が与えられたときの $Y$ の条件付き確率密度関数は

$$
f(y\mid x) = \frac{f(x,y)}{f_X(x)}
$$

です。$x$ を固定すると $y$ の密度になり、$\int f(y\mid x)\,dy=1$ を満たします。

条件付き確率・条件付き期待値は、この密度で計算します。

$$
P(Y\in B\mid X=x) = \int_B f(y\mid x)\,dy,\qquad E[Y\mid X=x] = \int_{-\infty}^{\infty} y\,f(y\mid x)\,dy
$$

同時密度は $f(x,y)=f(y\mid x)f_X(x)=f(x\mid y)f_Y(y)$ と分解できます。
$X,Y$ が独立であることは、$f(y\mid x)=f_Y(y)$ がすべての $x$ で成り立つことと同じです。

**2変量正規分布の条件付き分布**（重要）：$(X,Y)$ が平均 $\mu_X,\mu_Y$、標準偏差 $\sigma_X,\sigma_Y$、相関係数 $\rho$ の2変量正規分布に従うとき、

$$
Y\mid X=x \ \sim\ N\left(\mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X),\ \ \sigma_Y^2(1-\rho^2)\right)
$$

## 数式の読み解き

- $f(x,y)$ を $x$ で固定したもの：同時密度の曲面を $x$ の位置で縦に切った切り口の曲線です。
- $f_X(x)$ で割る：切り口の面積は $f_X(x)$ なので、これで割ると面積1の密度になります。

$P(X=x)=0$ なのに「$X=x$ という条件」を付けられるのは、$x\le X\le x+\Delta x$ という細い帯で条件付けて $\Delta x\to0$ とした極限と考えるからです。

$$
P(Y\le y\mid x\le X\le x+\Delta x) \approx \frac{\int_{-\infty}^{y} f(x,v)\,\Delta x\,dv}{f_X(x)\,\Delta x} = \int_{-\infty}^{y} \frac{f(x,v)}{f_X(x)}\,dv
$$

2変量正規分布の結果は、平均が $x$ の1次式（回帰直線）になり、分散は $x$ によらず $1-\rho^2$ 倍に縮む、と読めます。
$\rho$ が $\pm1$ に近いほど、$x$ を知ることで $Y$ の不確かさが大きく減ります。

分解 $f(x,y)=f(y\mid x)f_X(x)$ を使うと、逆向きの条件付き密度も求められます。

$$
f(x\mid y) = \frac{f(y\mid x)\,f_X(x)}{\int f(y\mid x')\,f_X(x')\,dx'}
$$

これは連続型のベイズの定理で、$x$ をパラメータ、$y$ をデータと読めば事後分布の式そのものです。

## 具体例

**例1：身長から体重を予測する**

成人の身長 $X$（cm）と体重 $Y$（kg）が2変量正規分布に従い、$\mu_X=170,\ \sigma_X=6,\ \mu_Y=60,\ \sigma_Y=8,\ \rho=0.6$ だとします。身長 176 cm の人の体重の条件付き分布は

$$
\begin{aligned}
E[Y\mid X=176] &= 60+0.6\times\tfrac{8}{6}\times(176-170) = 64.8\\
V[Y\mid X=176] &= 8^2\times(1-0.6^2) = 40.96 = 6.4^2
\end{aligned}
$$

体重が 70 kg を超える確率は

$$
P(Y>70\mid X=176) = P\left(Z>\tfrac{70-64.8}{6.4}\right) = P(Z>0.8125)\approx 0.2083
$$

身長の情報がないときの $P(Y>70)=P(Z>1.25)\approx0.1056$ のおよそ2倍です。

**例2：正方形上の密度で条件付ける**

$f(x,y)=x+y\ (0<x<1,\,0<y<1)$、$f_X(x)=x+\tfrac12$ のとき、

$$
f(y\mid x) = \frac{x+y}{x+\tfrac12}\quad(0<y<1)
$$

$x=\tfrac12$ なら $f(y\mid\tfrac12)=y+\tfrac12$ です。

$$
P\left(Y\le\tfrac12\ \middle|\ X=\tfrac12\right) = \int_0^{1/2}\left(y+\tfrac12\right)dy = 0.375,\qquad
E\left[Y\ \middle|\ X=\tfrac12\right] = \int_0^1 y\left(y+\tfrac12\right)dy = \tfrac{7}{12}\approx0.5833
$$

**例3：三角形の台**

$f(x,y)=2\ (0<y<x<1)$、$f_X(x)=2x$ なら $f(y\mid x)=\dfrac{2}{2x}=\dfrac1x\ (0<y<x)$ で、$Y\mid X=x$ は一様分布 $U(0,x)$ です。
したがって $E[Y\mid X=x]=x/2$、全体では $E[Y]=E[X/2]=\tfrac12\cdot\tfrac23=\tfrac13$ となり、周辺密度から直接求めた値と一致します。

## 現場での使われ方

- **医療**：年齢や体格で条件付けた検査値の分布から「年齢別の基準範囲」を作ります。全体の分布で判定するより誤判定が減ります。
- **製造・品質管理**：前工程の寸法で条件付けた後工程の寸法の分布を使い、前工程の測定値から最終製品の規格外れ確率を予測します。
- **IT・Webサービス**：滞在時間で条件付けた購入金額の分布を推定し、滞在が長い利用者向けの推薦の効果を見積もります。
- **金融・保険**：ある資産が大きく下落したという条件のもとでの、別の資産の損益分布を求めてストレステストを行います。

## グラフ・イメージ

1つ目の図では、条件付き密度（実線）が周辺密度（破線）より右にずれ、幅も狭くなっています。条件付き平均のスライダーを動かすと、身長の値を変えたときの体重分布の移動を再現できます。
2つ目の散布図では、ρ を大きくすると、縦に切った帯の中の点の散らばりが小さくなります。これが条件付き分散 $\sigma_Y^2(1-\rho^2)$ の縮小です。

## 試験のポイント

- 2変量正規分布の条件付き平均・条件付き分散の公式は必ず覚え、数値を代入して確率まで出せるようにします。条件付き平均の傾き $\rho\sigma_Y/\sigma_X$ は回帰係数です。
- 「$X$ の分布と、$X$ が与えられたときの $Y$ の条件付き分布」から同時密度を作り、周辺化して $Y$ の分布や期待値を求める階層型の問題は重要です。
- $E[Y]=E[E[Y\mid X]]$、$V[Y]=E[V[Y\mid X]]+V[E[Y\mid X]]$ を使うと、$Y$ の周辺密度を求めずに期待値・分散が出せます。
- 条件付き密度の台（例3の $0<y<x$）を書き忘れないようにします。
- 例：$X\sim U(0,1)$、$Y\mid X=x\sim U(0,x)$ なら、同時密度は $1/x\ (0<y<x<1)$、$Y$ の周辺密度は $-\log y$、$E[Y]=E[X]/2=0.25$ です。

## よくある間違い

- × $f(y\mid x)$ を求めるとき $f_Y(y)$ で割る → ○ 条件側の変数 $X$ の周辺密度 $f_X(x)$ で割ります。
- × 2変量正規分布の条件付き分散を $\sigma_Y^2\rho^2$ とする → ○ $\sigma_Y^2(1-\rho^2)$ です。相関が強いほど小さくなります。
- × $f(y\mid x)$ の積分範囲をいつも周辺の台全体にする → ○ 例3のように、条件付き密度の台は $x$ に依存することがあります。
- × 条件付き分散が $x$ によって変わると思う → ○ 2変量正規分布では $x$ によらず一定です（一般の分布では $x$ によって変わることがあります）。
