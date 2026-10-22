---
id: generalized-linear-model
title: 一般化線形モデル
reading: いっぱんかせんけいもでる
en: Generalized linear model
aliases: [GLM, 一般化線型モデル, "generalized linear model"]
chapter: 18
importance: A
tags: [質的回帰, 一般化線形モデル, 指数分散型分布族, 回帰モデル]
related: [link-function, exponential-dispersion-family, logistic-regression, poisson-regression, deviance, multiple-regression-model]
charts:
  - type: custom
    id: line-data
    params: { x: [0, 1, 2, 3, 4], series: [{ label: "観測件数 y", y: [2, 3, 6, 7, 12] }, { label: "ポアソン回帰（対数リンク）の当てはめ値", y: [2.1172, 3.2589, 5.0164, 7.7217, 11.8858] }], xlabel: "広告の出稿量 x", ylabel: "問い合わせ件数" }
    caption: 例1のポアソン回帰。対数リンクなので、x が1増えるごとに平均が一定の倍率 e^β1 ≈ 1.54 で増え、当てはめ値は指数関数的なカーブになる。
  - type: custom
    id: logistic-curve
    params: { beta0: -2.048, beta1: 0.939, xmin: -2, xmax: 6 }
    sliders:
      - { param: beta0, label: "切片 β0", min: -5, max: 5, step: 0.1 }
      - { param: beta1, label: "傾き β1", min: -3, max: 3, step: 0.05 }
    caption: 二項分布＋ロジットリンクの GLM（ロジスティック回帰）。線形予測子 η=β0+β1x は直線だが、逆リンクを通すと平均 μ=p は0と1の間のS字曲線になる。
---

## 一言でいうと

一般化線形モデル（GLM）は、正規分布を前提とした線形回帰を、二項分布やポアソン分布などにも使えるように広げた枠組みです。
「分布」「説明変数の一次式」「その2つをつなぐ関数」の3つを選ぶだけで、ロジスティック回帰もポアソン回帰も同じ考え方で扱えます。

## 定義・数式

GLM は次の3つの要素からなります。

1. **ランダム成分**：応答 $Y_1,\dots,Y_n$ は互いに独立で、それぞれ指数分散型分布族に従う。

$$
f(y_i;\theta_i,\phi)=\exp\left\{\frac{y_i\theta_i-b(\theta_i)}{a(\phi)}+c(y_i,\phi)\right\}
$$

   このとき $E[Y_i]=\mu_i=b'(\theta_i)$、$V[Y_i]=a(\phi)\,b''(\theta_i)=a(\phi)V(\mu_i)$ です。

2. **系統的成分（線形予測子）**：

$$
\eta_i=\beta_0+\beta_1x_{i1}+\cdots+\beta_kx_{ik}=\boldsymbol{x}_i^\top\boldsymbol\beta
$$

3. **リンク関数**：平均と線形予測子を、単調で微分可能な関数 $g$ でつなぐ。

$$
g(\mu_i)=\eta_i\qquad\Longleftrightarrow\qquad \mu_i=g^{-1}(\eta_i)
$$

代表的な組み合わせは次のとおりです。

| モデル | 分布 | 分散関数 $V(\mu)$ | よく使うリンク |
|---|---|---|---|
| 線形回帰 | 正規 | $1$ | 恒等 $g(\mu)=\mu$ |
| ロジスティック回帰 | 二項（ベルヌーイ） | $\mu(1-\mu)$ | ロジット |
| プロビットモデル | 二項（ベルヌーイ） | $\mu(1-\mu)$ | プロビット $\Phi^{-1}(\mu)$ |
| ポアソン回帰 | ポアソン | $\mu$ | 対数 $\log\mu$ |
| ガンマ回帰 | ガンマ | $\mu^2$ | 逆数・対数 |

パラメータ $\boldsymbol\beta$ は最尤法で推定します。尤度方程式（スコア方程式）は

$$
\sum_{i=1}^n\frac{(y_i-\mu_i)}{a(\phi)V(\mu_i)}\,\frac{\partial\mu_i}{\partial\eta_i}\,x_{ij}=0\qquad(j=0,1,\dots,k)
$$

で、反復重み付き最小二乗法（IRLS）で解きます。

## 数式の読み解き

- $\theta_i$：**自然パラメータ**（正準パラメータ）。分布ごとに決まる「平均を変換したもの」です。
- $b(\theta)$：分布の形を決める関数。微分すると平均、2回微分すると分散関数が出てきます。
- $a(\phi)$：ばらつきの大きさを表す**分散パラメータ**の部分。正規分布では $\sigma^2$、二項・ポアソンでは1に固定されます。
- $\eta_i$：説明変数の一次式。ここは線形回帰と同じで、ダミー変数や交互作用もそのまま入れられます。
- $g$：$\mu$ の値の範囲（確率なら $(0,1)$、件数の平均なら $(0,\infty)$）を、$\eta$ の範囲 $(-\infty,\infty)$ に合わせるための関数です。

スコア方程式の導出は連鎖律だけです。対数尤度 $\ell_i=\{y_i\theta_i-b(\theta_i)\}/a(\phi)+c$ を $\beta_j$ で微分すると

$$
\frac{\partial\ell_i}{\partial\beta_j}
=\frac{\partial\ell_i}{\partial\theta_i}\frac{\partial\theta_i}{\partial\mu_i}\frac{\partial\mu_i}{\partial\eta_i}\frac{\partial\eta_i}{\partial\beta_j}
=\frac{y_i-\mu_i}{a(\phi)}\cdot\frac{1}{V(\mu_i)}\cdot\frac{\partial\mu_i}{\partial\eta_i}\cdot x_{ij}
$$

です。$\partial\mu/\partial\theta=b''(\theta)=V(\mu)$ を使いました。
リンクが**正準リンク**（$\theta_i=\eta_i$）のときは $\partial\mu_i/\partial\eta_i=V(\mu_i)$ となって約分され、

$$
\sum_{i=1}^n(y_i-\mu_i)x_{ij}=0\qquad\text{すなわち}\qquad X^\top(\boldsymbol y-\boldsymbol\mu)=\boldsymbol 0
$$

という、正規方程式とそっくりの形になります。

線形回帰は「正規分布＋恒等リンク」の GLM です。このときスコア方程式は $X^\top(\boldsymbol y-X\boldsymbol\beta)=\boldsymbol0$、つまり正規方程式そのものになり、最尤推定量は最小二乗推定量と一致します。

## 具体例

**例1：ポアソン回帰（ポアソン分布＋対数リンク）**

広告の出稿量 $x=0,1,2,3,4$ の週に、問い合わせ件数が $y=2,3,6,7,12$ でした。
$Y_i\sim\text{Po}(\mu_i)$、$\log\mu_i=\beta_0+\beta_1x_i$ を当てはめると、

$$
\hat\beta_0=0.7501,\qquad\hat\beta_1=0.4313
$$

です。出稿量が1増えるごとに平均件数は $e^{0.4313}\approx1.539$ 倍になります。
当てはめ値は $\hat\mu=2.117,\ 3.259,\ 5.016,\ 7.722,\ 11.886$ です。

正準リンクなので、スコア方程式 $X^\top(\boldsymbol y-\hat{\boldsymbol\mu})=\boldsymbol0$ が成り立っているはずです。確かめると

$$
\sum\hat\mu_i=30.000=\sum y_i,\qquad \sum x_i\hat\mu_i=84.000=\sum x_iy_i
$$

となり、合計と「$x$ で重みをつけた合計」が観測値と一致しています。$x=5$ での予測平均は $e^{0.7501+0.4313\times5}\approx18.30$ 件です。

**例2：同じデータで3要素を言い分ける**

殺虫剤の濃度 $x$ と、20匹中の死亡数 $y$ のデータ（ロジスティック回帰の例1）なら、

- ランダム成分：$Y_i\sim\text{Bin}(20,p_i)$。$a(\phi)=1$、$V(\mu)=\mu(1-\mu/20)$（割合 $y/20$ で書けば $p(1-p)/20$）。
- 線形予測子：$\eta_i=\beta_0+\beta_1x_i$。
- リンク：$\log\{p_i/(1-p_i)\}=\eta_i$（正準リンク）。

推定値 $\hat\beta_0=-2.0479$、$\hat\beta_1=0.9391$ から、$x=3$ での平均死亡数は $20\times1/(1+e^{-0.7693})\approx20\times0.6834=13.67$ 匹です。

## 現場での使われ方

- **医療**：発症の有無（二項）、来院回数（ポアソン）、医療費（ガンマ）など、応答の種類に合わせて分布を選び、同じ手順で要因の効果を推定します。
- **金融・保険**：保険料の算定で、事故件数をポアソン回帰（契約期間をオフセットに）、1件あたりの支払額をガンマ回帰でモデル化するのが標準的な方法です。
- **製造・品質管理**：製品あたりの欠点数をポアソン回帰、不良品かどうかをロジスティック回帰で表し、工程条件の影響を調べます。
- **行政・調査**：地域ごとの事故件数や疾病の発生件数を、人口をオフセットにしたポアソン回帰で比較します。

## グラフ・イメージ

1つ目は例1のポアソン回帰です。線形予測子は直線ですが、対数リンクの逆関数 $\exp$ を通すので、平均は指数関数的に増えるカーブになります。
2つ目はロジスティック回帰で、同じ直線 $\eta$ がロジットの逆関数でS字曲線に変わります。どちらも「直線 → 逆リンク → 平均」という同じ構造であることを見てください。

## 試験のポイント

- 3要素（ランダム成分・線形予測子・リンク関数）を、与えられたモデルについて答えさせる問題が重要です。
- 指数分散型分布族の $b(\theta)$ から $\mu=b'(\theta)$、$V[Y]=a(\phi)b''(\theta)$ を求める計算はよく問われます。
- 正準リンクのとき、スコア方程式が $X^\top(\boldsymbol y-\boldsymbol\mu)=\boldsymbol0$ になることを導けるようにします。
- 線形回帰は GLM の特殊な場合（正規分布＋恒等リンク）で、最尤推定量 ＝ 最小二乗推定量です。
- モデルの当てはまりや比較は、残差平方和の代わりに逸脱度で行います（正規分布では逸脱度 ＝ 残差平方和）。

## よくある間違い

- × GLM は「応答を変換してから線形回帰する」方法 → ○ 変換するのは平均 $\mu$ です。$\log y$ を回帰するのと $\log E[Y]$ をモデル化するのは別物で、GLM では $y=0$ でも問題なく扱えます。
- × 一般化線形モデル（GLM）と一般線形モデル（GLM と略されることもある）を同じものとする → ○ 一般線形モデルは正規誤差の線形モデル（分散分析・重回帰）を指し、一般化線形モデルはそれを指数分散型分布族に広げたものです。
- × 「線形」だから $\mu$ が $x$ の一次式 → ○ 一次式なのは $\eta=g(\mu)$ のほうで、$\mu$ 自体は $x$ の非線形関数になります。
