---
id: score-function
title: スコア関数
reading: すこあかんすう
en: Score function
aliases: [スコア, 有効スコア]
chapter: 8
importance: A
tags: [尤度, 対数尤度, 尤度方程式, フィッシャー情報量]
related: [likelihood-function, maximum-likelihood-estimator, fisher-information, cramer-rao-inequality, score-test]
charts:
  - type: custom
    id: line-data
    params:
      x: [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6]
      xlabel: "ポアソン分布の母数 λ"
      ylabel: "スコア U(λ)"
      series:
        - { label: "U(λ) = 15/λ − 5（データ 2,4,3,1,5）", y: [10, 7, 5, 3.571, 2.5, 1.667, 1, 0.455, 0, -0.385, -0.714, -1, -1.25, -1.471, -1.667, -1.842, -2, -2.143, -2.273, -2.391, -2.5] }
    caption: ポアソン分布のスコア関数。λ が最尤推定値 3 より小さいと正（λ を増やせば尤度が上がる）、大きいと負になり、λ=3 でちょうど 0 を横切る。
---

## 一言でいうと

対数尤度を母数で微分したもので、「いまの母数の値から、どちらへ動かせば尤度が上がるか」を示す傾きです。
スコアが 0 になる点が最尤推定値で、スコアのばらつき（分散）がフィッシャー情報量です。

## 定義・数式

1個の観測 $X$ の密度（確率関数）を $f(x;\theta)$ とするとき、

$$
s(\theta;X) = \frac{\partial}{\partial\theta}\log f(X;\theta)
$$

を**スコア関数**といいます。独立な標本 $X_1,\dots,X_n$ 全体のスコアは各観測のスコアの和です。

$$
U(\theta) = \frac{\partial \ell(\theta)}{\partial\theta} = \sum_{i=1}^n \frac{\partial}{\partial\theta}\log f(X_i;\theta)
$$

尤度方程式は $U(\hat\theta)=0$ です。

**性質**（正則条件：台が $\theta$ に依存せず、微分と積分の順序を交換できる）

$$
E_\theta[s(\theta;X)] = 0,\qquad V_\theta[s(\theta;X)] = E_\theta\big[s(\theta;X)^2\big] = I(\theta)
$$

$I(\theta)$ はフィッシャー情報量です。標本全体では $E[U(\theta)]=0$、$V[U(\theta)]=nI(\theta)$ です。

## 数式の読み解き

- $U(\theta)>0$：$\theta$ を増やすと対数尤度が上がる（最尤推定値はもっと右）。
- $U(\theta)<0$：$\theta$ を減らすと対数尤度が上がる（最尤推定値はもっと左）。
- スコアは確率変数 $X$ の関数なので、それ自体が確率変数です。「真の $\theta$ で評価したスコアは平均 0 のまわりでばらつく」というのが $E[s]=0$ の意味です。

**$E[s]=0$ の導出**（連続型。離散型は積分を和に置き換える）

$\frac{\partial}{\partial\theta}\log f = \frac{1}{f}\frac{\partial f}{\partial\theta}$ を使うと

$$
E_\theta[s] = \int \frac{1}{f(x;\theta)}\frac{\partial f(x;\theta)}{\partial\theta}\,f(x;\theta)\,dx = \int \frac{\partial f(x;\theta)}{\partial\theta}\,dx = \frac{\partial}{\partial\theta}\int f(x;\theta)\,dx = \frac{\partial}{\partial\theta}1 = 0
$$

3つ目の等号で微分と積分を交換しています。ここに正則条件が必要です。

**$V[s]=-E[\partial s/\partial\theta]$ の導出**

恒等式 $\int s(\theta;x)f(x;\theta)\,dx = 0$ を、もう一度 $\theta$ で微分します。積の微分と $\partial f/\partial\theta = s\,f$ を使うと

$$
\int \frac{\partial s}{\partial\theta}\,f\,dx + \int s\cdot s\,f\,dx = 0
\quad\Longrightarrow\quad
E[s^2] = -E\Big[\frac{\partial s}{\partial\theta}\Big] = -E\Big[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\Big]
$$

左辺がスコアの分散、右辺が対数尤度の曲がり具合（2階微分の符号を変えた期待値）です。どちらもフィッシャー情報量 $I(\theta)$ の定義として使われます。
直感的には、「スコアが大きく揺れる」ことと「対数尤度の山が鋭い」ことは同じで、どちらもデータが母数について多くの情報をもつことを表します。

**各分布のスコア（1個あたり）**

| 分布 | $\log f(x;\theta)$（定数を除く） | スコア $s(\theta;x)$ |
|---|---|---|
| ベルヌーイ $Bern(p)$ | $x\log p+(1-x)\log(1-p)$ | $\dfrac{x}{p}-\dfrac{1-x}{1-p} = \dfrac{x-p}{p(1-p)}$ |
| ポアソン $Po(\lambda)$ | $x\log\lambda-\lambda$ | $\dfrac{x}{\lambda}-1 = \dfrac{x-\lambda}{\lambda}$ |
| 正規 $N(\mu,\sigma^2)$, $\sigma^2$ 既知 | $-\dfrac{(x-\mu)^2}{2\sigma^2}$ | $\dfrac{x-\mu}{\sigma^2}$ |
| 指数（密度 $\lambda e^{-\lambda x}$） | $\log\lambda-\lambda x$ | $\dfrac1\lambda - x$ |

どれも期待値が 0 になることを直接確かめられます。たとえばベルヌーイでは $E[X-p]=0$、指数分布では $E[X]=1/\lambda$ より $E[1/\lambda-X]=0$ です。

## 具体例

**例1：来店客数（ポアソン）**

データ 2, 4, 3, 1, 5（合計15、$n=5$）の全体のスコアは

$$
U(\lambda) = \sum_{i=1}^5\Big(\frac{x_i}{\lambda}-1\Big) = \frac{15}{\lambda} - 5
$$

- $U(2) = 7.5-5 = 2.5$（正：$\lambda$ をもっと大きくすべき）
- $U(3) = 5-5 = 0$（最尤推定値）
- $U(4) = 3.75-5 = -1.25$（負：$\lambda$ を小さくすべき）

**例2：スコアの分散を確かめる（ベルヌーイ）**

$p=0.3$ のとき、$X=1$ ならスコアは $(1-0.3)/0.21 = 3.3333$、$X=0$ なら $-0.3/0.21=-1.4286$ です。

$$
E[s] = 0.3\times3.3333 + 0.7\times(-1.4286) = 1.0 - 1.0 = 0
$$

$$
E[s^2] = 0.3\times3.3333^2 + 0.7\times1.4286^2 = 3.3333 + 1.4286 = 4.7619
$$

これは $I(p)=1/\{p(1-p)\}=1/0.21\approx4.7619$ に一致します。

## 現場での使われ方

- **医療・疫学**：スコア検定（ログランク検定もその一種）は、帰無仮説の値でスコアを評価するだけで計算できるため、対立仮説のもとでの推定を行わずに検定できます。
- **IT・Webサービス**：最尤推定を数値的に解くとき、スコア（勾配）を使ってパラメータを更新します（勾配法、ニュートン法、フィッシャーのスコア法）。
- **製造・品質管理**：工程の母数が基準値からずれ始めたかを監視するとき、基準値で評価したスコアを累積する方法（CUSUM の考え方）が使われます。

## グラフ・イメージ

ポアソン分布のスコアを $\lambda$ の関数として描いています。曲線が 0 を横切る点 $\lambda=3$ が最尤推定値です。
対数尤度のグラフ（最尤推定量の項目）の「傾き」をそのまま描いたものと考えると、関係がよくわかります。

## 試験のポイント

- $E[s]=0$ と $V[s]=I(\theta)$ は、フィッシャー情報量・クラメール・ラオの不等式の証明の土台です。導出の流れ（$\frac{\partial}{\partial\theta}\int f=0$）を押さえます。
- 各分布のスコアを計算し、尤度方程式 $U(\theta)=0$ を解く流れは最尤推定と同じです。
- スコア検定の統計量は $U(\theta_0)^2/\{nI(\theta_0)\}$ で、帰無仮説のもとで漸近的に $\chi^2_1$ に従います（第10章）。
- スコアが推定量の1次式 $a(\theta)(T-\theta)$ になると、$T$ は有効推定量になります（クラメール・ラオの等号成立条件）。

## よくある間違い

- × スコアは $\log f$ を $x$ で微分したもの → ○ 微分するのは**母数** $\theta$ です。
- × どんな分布でも $E[s]=0$ → ○ 微分と積分を交換できること（正則条件）が必要です。一様分布 $U(0,\theta)$ のように台が $\theta$ に依存すると成り立ちません。
- × スコアの分散はスコアの二乗の期待値と別物 → ○ $E[s]=0$ なので $V[s]=E[s^2]$ です。
