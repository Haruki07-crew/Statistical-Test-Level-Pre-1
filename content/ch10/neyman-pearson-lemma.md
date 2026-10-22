---
id: neyman-pearson-lemma
title: ネイマン・ピアソンの補題
reading: ねいまん・ぴあそんのほだい
en: Neyman-Pearson lemma
aliases: [ネイマン・ピアソンの基本定理, 最強力検定, "NP補題"]
chapter: 10
importance: A
tags: [仮説検定, 検定法の導出, 尤度比, 最強力検定, 単純仮説]
related: [uniformly-most-powerful-test, monotone-likelihood-ratio, likelihood-ratio-test, likelihood-function, power-function, randomized-test]
charts:
  - type: custom
    id: power-curve
    params: { mu0: 0, mu1: 1, sigma: 1, n: 9, alpha: 0.05, alt: greater }
    sliders:
      - { param: mu1, label: "対立仮説の μ1", min: 0.1, max: 2, step: 0.05 }
      - { param: alpha, label: "有意水準 α", min: 0.01, max: 0.2, step: 0.01 }
      - { param: n, label: "標本サイズ n", min: 2, max: 50, step: 1 }
    caption: 単純仮説 μ=0 対 μ=μ1 の最強力検定は「標本平均が棄却限界を超えたら棄却」。棄却限界は μ1 の値によらず、α と n だけで決まる。
---

## 一言でいうと

単純な帰無仮説と単純な対立仮説を比べるとき、有意水準 $\alpha$ の検定の中で最も検出力が高い検定（最強力検定）は、**尤度比が大きいときに棄却する検定**である、という定理です。
検定の棄却域を「なんとなく」ではなく、理論的に導くための出発点です。

## 定義・数式

標本 $\boldsymbol x$ の同時密度（確率関数）を $f(\boldsymbol x;\theta)$ とし、単純仮説

$$
H_0:\theta=\theta_0\qquad\text{対}\qquad H_1:\theta=\theta_1
$$

を考えます。定数 $k>0$ に対して棄却域を

$$
R=\left\{\boldsymbol x:\ \frac{f(\boldsymbol x;\theta_1)}{f(\boldsymbol x;\theta_0)}>k\right\},\qquad P_{\theta_0}(\boldsymbol X\in R)=\alpha
$$

と定めます。このとき、有意水準 $\alpha$ の**任意の**検定（棄却域 $R'$、$P_{\theta_0}(\boldsymbol X\in R')\le\alpha$）について

$$
P_{\theta_1}(\boldsymbol X\in R)\ \ge\ P_{\theta_1}(\boldsymbol X\in R')
$$

が成り立ちます。つまり $R$ は水準 $\alpha$ の**最強力検定**（most powerful test）です。

離散分布で $P_{\theta_0}(\boldsymbol X\in R)=\alpha$ をちょうど満たす $k$ がないときは、尤度比が $k$ に等しい点で確率 $\gamma$ で棄却する**無作為化検定**にすれば、同じ結論が成り立ちます。

## 数式の読み解き

- 分子 $f(\boldsymbol x;\theta_1)$ は「対立仮説のもとでのデータのもっともらしさ」、分母は「帰無仮説のもとでのもっともらしさ」です。
- 尤度比が大きい点は、「$H_0$ のもとでは出にくいのに、$H_1$ のもとでは出やすい」点です。限られた予算 $\alpha$（$H_0$ のもとでの確率）で、$H_1$ のもとでの確率（検出力）をできるだけ多く買うには、**費用対効果（尤度比）の高い点から順に棄却域に入れる**のが最適、という直感です。

**証明の筋道**（連続型）：$R$ の上では $f_1>kf_0$、$R$ の外では $f_1\le kf_0$ なので、どの点でも

$$
\big(\mathbf 1_R(\boldsymbol x)-\mathbf 1_{R'}(\boldsymbol x)\big)\big(f_1(\boldsymbol x)-kf_0(\boldsymbol x)\big)\ge0
$$

です（$\mathbf 1_R$ は $R$ の上で1、外で0）。これを積分すると

$$
\big\{P_1(R)-P_1(R')\big\}\ \ge\ k\big\{P_0(R)-P_0(R')\big\}=k\big\{\alpha-P_0(R')\big\}\ \ge\ 0
$$

となり、$P_1(R)\ge P_1(R')$ が示されます。

**使い方の手順**

1. 尤度比 $\Lambda(\boldsymbol x)=f(\boldsymbol x;\theta_1)/f(\boldsymbol x;\theta_0)$ を計算する。
2. 対数をとって整理し、$\Lambda>k$ を「十分統計量 $T$ が $c$ より大きい（または小さい）」の形に書き換える。
3. $H_0$ のもとでの $T$ の分布から、$P_{\theta_0}(T>c)=\alpha$ となる $c$ を決める（$k$ を直接求める必要はない）。

## 具体例

**例1：正規分布の平均**（$\sigma$ 既知）

$X_1,\dots,X_n\sim N(\mu,\sigma^2)$、$H_0:\mu=\mu_0$ 対 $H_1:\mu=\mu_1$（$\mu_1>\mu_0$）。

$$
\log\Lambda=-\frac{1}{2\sigma^2}\sum(x_i-\mu_1)^2+\frac{1}{2\sigma^2}\sum(x_i-\mu_0)^2
=\frac{n(\mu_1-\mu_0)}{\sigma^2}\bar x-\frac{n(\mu_1^2-\mu_0^2)}{2\sigma^2}
$$

$\mu_1-\mu_0>0$ なので $\log\Lambda$ は $\bar x$ の増加関数で、$\Lambda>k\iff\bar x>c$ です。$H_0$ のもとで $\bar X\sim N(\mu_0,\sigma^2/n)$ より $c=\mu_0+z_\alpha\sigma/\sqrt n$。

数値：$\mu_0=0$、$\mu_1=1$、$\sigma=1$、$n=9$、$\alpha=0.05$ なら $c=1.645/3\approx0.5483$、検出力は

$$
P_{\mu=1}(\bar X>0.5483)=1-\Phi\big((0.5483-1)\times3\big)=\Phi(1.355)\approx0.9123
$$

**例2：ベルヌーイ分布**

$X_i\sim Bern(p)$、$H_0:p=p_0$ 対 $H_1:p=p_1$（$p_1>p_0$）、$T=\sum X_i$ とすると

$$
\log\Lambda=T\log\frac{p_1}{p_0}+(n-T)\log\frac{1-p_1}{1-p_0}
=T\log\frac{p_1(1-p_0)}{p_0(1-p_1)}+n\log\frac{1-p_1}{1-p_0}
$$

$p_1>p_0$ ならオッズ比 $\frac{p_1(1-p_0)}{p_0(1-p_1)}>1$ なので、$T$ の増加関数です。棄却域は $T\ge c$。

数値：$n=10$、$p_0=0.5$、$p_1=0.8$、$\alpha=0.05$。$P_{0.5}(T\ge9)=0.0107$、$P_{0.5}(T\ge8)=0.0547$ なので、非無作為化なら $T\ge9$（検出力 $P_{0.8}(T\ge9)\approx0.3758$）。$T=8$ のとき確率 $\gamma=(0.05-0.0107)/P_{0.5}(T=8)\approx0.8933$ で棄却する無作為化検定にすると、サイズがちょうど 0.05、検出力は約 0.6456 になります。

**例3：ポアソン分布**

$X_i\sim Po(\lambda)$、$H_0:\lambda=2$ 対 $H_1:\lambda=3$、$n=5$、$T=\sum X_i$。

$$
\log\Lambda=T\log\frac{3}{2}-n(3-2)=0.4055\,T-5
$$

で $T$ の増加関数、棄却域は $T\ge c$ です。$H_0$ のもとで $T\sim Po(10)$ なので、$P(T\ge15)=0.0835$、$P(T\ge16)=0.0487$ より $c=16$（サイズ 0.0487）。$H_1$ のもとで $T\sim Po(15)$ なので検出力は $P(T\ge16)\approx0.4319$ です。

**例4：指数分布**

$X_i$ が密度 $\lambda e^{-\lambda x}$ の指数分布、$H_0:\lambda=1$ 対 $H_1:\lambda=0.5$（平均寿命が延びた）、$n=10$。

$$
\log\Lambda=n\log\frac{\lambda_1}{\lambda_0}-(\lambda_1-\lambda_0)\sum x_i=10\log0.5+0.5\sum x_i
$$

で $\sum x_i$ の増加関数、棄却域は $\sum X_i>c$。$2\lambda\sum X_i\sim\chi^2(2n)$ を使うと、$H_0$ のもとで $2\sum X_i\sim\chi^2(20)$ なので $c=\chi^2_{0.05}(20)/2=31.410/2\approx15.705$。$H_1$ のもとでは $\sum X_i\sim\chi^2(20)$ なので、検出力は $P(\chi^2(20)>15.705)\approx0.7347$ です。

## 現場での使われ方

- **製造・品質管理**：逐次抜取検査（ワルドの逐次確率比検定）は、1個検査するたびに尤度比を更新し、上限・下限を超えた時点で合格・不合格を決めます。ネイマン・ピアソンの考え方を逐次化したものです。
- **IT・Webサービス**：迷惑メールや不正アクセスの判定で、「不正のときの尤度」と「正常のときの尤度」の比をスコアにし、誤検知率を一定に保ちながら検出率を最大にするしきい値を選びます。
- **通信・信号処理**：雑音の中に信号があるかを判定するとき、誤警報率を固定して検出確率を最大にする検出器として尤度比検定が使われます。

## グラフ・イメージ

$\mu_0=0$ 対 $\mu_1$ の最強力検定の、標本平均の分布と検出力です。棄却限界は $\alpha$ と $n$ だけで決まり、$\mu_1$ を動かしても変わりません。
このことが、どの $\mu_1>0$ に対しても同じ検定が最強力になる、つまり一様最強力検定につながります。

## 試験のポイント

- 「尤度比をとる → 対数をとる → 十分統計量の単調関数に直す → $H_0$ のもとでの分布で棄却限界を決める」の流れを、正規・二項・ポアソン・指数で一通り練習しておきます。
- 不等号の向きは $\theta_1>\theta_0$ か $\theta_1<\theta_0$ かで逆になります。係数の符号を必ず確認します。
- 定数 $k$ そのものは求めなくてよく、統計量の棄却限界 $c$ を求めれば十分です。
- 補題は**単純対単純**の仮説についての結果です。複合仮説への拡張が一様最強力検定、一般化が尤度比検定です。

## よくある間違い

- × 尤度比を $f(\boldsymbol x;\theta_0)/f(\boldsymbol x;\theta_1)$ にして「大きいとき棄却」 → ○ 分子は対立仮説です（逆にとるなら「小さいとき棄却」）。
- × 棄却限界 $c$ を対立仮説のもとでの分布で決める → ○ $c$ は $H_0$ のもとでの分布で決め、$H_1$ のもとでの分布は検出力の計算に使います。
- × 指数分布で平均寿命が延びたかを調べるとき「$\sum X_i$ が小さければ棄却」 → ○ $\lambda_1<\lambda_0$（平均が大きい）なら $\sum X_i$ が大きいときに棄却します。レートと平均のどちらで書かれているかに注意します。
