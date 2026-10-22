---
id: bivariate-normal-distribution
title: 2変量正規分布
reading: にへんりょうせいきぶんぷ
en: Bivariate normal distribution
aliases: [二変量正規分布, 2次元正規分布, Bivariate normal]
chapter: 6
importance: A
tags: [多次元分布, 条件付き分布, 相関, 回帰]
related: [normal-distribution, multivariate-normal-distribution, covariance-and-correlation, conditional-expectation, conditional-probability-density-function, joint-probability-density-function]
charts:
  - type: scatter
    model: bvnorm
    params: { rho: 0.5, n: 300 }
    fit: true
    sliders:
      - { param: rho, label: "相関係数 ρ", min: -0.95, max: 0.95, step: 0.05 }
      - { param: n, label: "点の数 n", min: 50, max: 1000, step: 50 }
    caption: 標準化した2変量正規分布からの乱数。点の雲は楕円形で、ρ を ±1 に近づけるほど細長くなる。直線は条件付き期待値 E[Y|X=x]=ρx の推定（回帰直線）。
---

## 一言でいうと

2つの量が「それぞれ正規分布に従い、かつ直線的な関係で結びついている」ことを表す同時分布です。
5つのパラメータ（2つの平均、2つの分散、相関係数）だけで完全に決まり、一方の値を知ったときの他方の分布（条件付き分布）がまた正規分布になるので、回帰分析の理論的な土台になります。

## 定義・数式

$\sigma_X,\sigma_Y>0$、$-1<\rho<1$ とします。$(X,Y)$ の同時密度が

$$
f(x,y)=\frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}}\exp\!\left[-\frac{1}{2(1-\rho^2)}\left\{\left(\frac{x-\mu_X}{\sigma_X}\right)^2-2\rho\,\frac{x-\mu_X}{\sigma_X}\cdot\frac{y-\mu_Y}{\sigma_Y}+\left(\frac{y-\mu_Y}{\sigma_Y}\right)^2\right\}\right]
$$

のとき、$(X,Y)$ は2変量正規分布に従うといいます。行列で書くと平均ベクトル $\boldsymbol\mu=(\mu_X,\mu_Y)^\top$、分散共分散行列

$$
\Sigma=\begin{pmatrix}\sigma_X^2 & \rho\sigma_X\sigma_Y\\ \rho\sigma_X\sigma_Y & \sigma_Y^2\end{pmatrix}
$$

の多変量正規分布 $N_2(\boldsymbol\mu,\Sigma)$ です。

| 項目 | 内容 |
|---|---|
| 周辺分布 | $X\sim N(\mu_X,\sigma_X^2)$、$Y\sim N(\mu_Y,\sigma_Y^2)$ |
| 共分散・相関 | $\mathrm{Cov}(X,Y)=\rho\sigma_X\sigma_Y$、相関係数 $\rho$ |
| 条件付き期待値 | $E[Y\mid X=x]=\mu_Y+\rho\dfrac{\sigma_Y}{\sigma_X}(x-\mu_X)$ |
| 条件付き分散 | $V[Y\mid X=x]=\sigma_Y^2(1-\rho^2)$（$x$ によらない） |
| モーメント母関数 | $M(t_1,t_2)=\exp\!\left\{\mu_Xt_1+\mu_Yt_2+\tfrac12(\sigma_X^2t_1^2+2\rho\sigma_X\sigma_Yt_1t_2+\sigma_Y^2t_2^2)\right\}$ |
| 線形結合 | $aX+bY\sim N(a\mu_X+b\mu_Y,\ a^2\sigma_X^2+2ab\rho\sigma_X\sigma_Y+b^2\sigma_Y^2)$ |

## 数式の読み解き

- 指数の中括弧は、標準化した $x,y$ の2次式です。これが一定になる点の集まり（等高線）は楕円になります。
- $\rho>0$ なら楕円は右上がり、$\rho<0$ なら右下がり、$\rho=0$ なら軸に平行です。
- $\sqrt{1-\rho^2}$ は、$|\rho|$ が1に近いほど分布がつぶれて細くなることを表しています。

**条件付き分布の導出**。標準化 $u=(x-\mu_X)/\sigma_X$、$v=(y-\mu_Y)/\sigma_Y$ をすると、指数の中は平方完成で

$$
\frac{u^2-2\rho uv+v^2}{1-\rho^2}=u^2+\frac{(v-\rho u)^2}{1-\rho^2}
$$

と分けられます。$f(x,y)=f_X(x)f(y\mid x)$ の形になり、第1項が周辺分布 $N(0,1)$、第2項が「平均 $\rho u$、分散 $1-\rho^2$ の正規分布」です。元の尺度に戻すと、表の条件付き期待値と条件付き分散になります。

**無相関と独立**。$\rho=0$ を代入すると同時密度が $f_X(x)f_Y(y)$ に分解されます。つまり2変量正規分布では**無相関 ⇔ 独立**です。一般の分布では無相関でも独立とは限りません。

## 具体例

**例1：父親と息子の身長（回帰効果）**

父親の身長 $X$ と息子の身長 $Y$（cm）が、$\mu_X=\mu_Y=170$、$\sigma_X=\sigma_Y=6$、$\rho=0.5$ の2変量正規分布に従うとします。父親が182 cm のとき、

$$
E[Y\mid X=182]=170+0.5\times\frac66\times(182-170)=176,\qquad \sqrt{V[Y\mid X=182]}=6\sqrt{1-0.25}\approx5.196
$$

この条件のもとで息子が180 cm を超える確率は

$$
P(Y>180\mid X=182)=1-\Phi\!\left(\frac{180-176}{5.196}\right)=1-\Phi(0.7698)\approx0.2207
$$

です。何も条件をつけないときの $P(Y>180)=1-\Phi(10/6)\approx0.0478$ より大きくなります。父親が平均より12 cm 高くても、息子の期待値は6 cm しか高くなりません。これが「平均への回帰」です。

**例2：差の分布（試験に近い形）**

同じ設定で、息子が父親より5 cm 以上高い確率を求めます。$Y-X$ は正規分布で

$$
V[Y-X]=36+36-2\times0.5\times6\times6=36
$$

なので $Y-X\sim N(0,6^2)$、$P(Y-X\ge5)=1-\Phi(0.8333)\approx0.2023$ です。相関を無視して分散を72とすると誤りになります。

## 現場での使われ方

- **医療**：最高血圧と最低血圧のように相関する2つの検査値を同時に評価し、楕円形の正常域を設定します。
- **製造・品質管理**：部品の縦と横の寸法を同時に管理するとき、2変数を個別に管理するより正確に異常を検出できます（多変量管理図の考え方）。
- **金融・保険**：2つの資産の収益率を2変量正規分布でモデル化し、ポートフォリオの分散 $a^2\sigma_X^2+2ab\rho\sigma_X\sigma_Y+b^2\sigma_Y^2$ を計算します。
- **行政・調査**：入学時の成績から1年後の成績を予測する際、条件付き期待値（回帰直線）と条件付き分散で予測区間を作ります。

## グラフ・イメージ

$\rho$ を0にすると点の雲は円形、$\pm0.9$ に近づけると細長い楕円になります。直線（回帰直線）の傾きはおよそ $\rho$ で、楕円の長軸（45度の線）より常に寝ていることに注目してください。これが平均への回帰を図で表したものです。

## 試験のポイント

- 条件付き期待値 $\mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)$ と条件付き分散 $\sigma_Y^2(1-\rho^2)$ は暗記し、平方完成による導出も追えるようにします。
- 条件付き分散が $x$ によらない（等分散）ことは、回帰モデルの仮定につながります。
- 線形結合の分散で共分散の項 $2ab\rho\sigma_X\sigma_Y$ を忘れないようにします。
- 「2変量正規なら無相関と独立は同値」は頻繁に問われます。

## よくある間違い

- × 周辺分布がそれぞれ正規分布なら、同時分布は2変量正規分布 → ○ 周辺が正規でも同時分布が2変量正規とは限りません。その場合、無相関でも独立とは言えません。
- × 条件付き分散は $\sigma_Y^2\rho^2$ → ○ $\sigma_Y^2(1-\rho^2)$ です。$X$ の情報で説明できる分 $\rho^2$ を除いた残りです。
- × $E[Y\mid X=x]$ の傾きを $\rho$ そのものとする → ○ 傾きは $\rho\sigma_Y/\sigma_X$ です。標準化したときだけ $\rho$ になります。
- × $X$ から $Y$ を予測する直線と、$Y$ から $X$ を予測する直線は同じ → ○ 2本の回帰直線は $|\rho|=1$ のときを除いて一致しません。
