---
id: instrumental-variable-method
title: 操作変数法
reading: そうさへんすうほう
en: Instrumental variable method
aliases: [IV法, 操作変数, 操作変数推定量, ワルド推定量, instrumental variable]
chapter: 19
importance: A
tags: [回帰分析, 内生性, 因果推論, 計量経済学, 一致推定量]
related: [endogeneity, two-stage-least-squares, omitted-variable-bias, least-squares-method, consistent-estimator, method-of-moments]
charts:
  - type: custom
    id: iv-scatter
    params: { rho: 0.6, strength: 1, n: 200, beta: 1 }
    sliders:
      - { param: strength, label: "操作変数の強さ", min: 0.1, max: 2, step: 0.1 }
      - { param: rho, label: "x と誤差の相関", min: -0.9, max: 0.9, step: 0.1 }
    caption: 真の直線・最小二乗法の直線・操作変数法の直線。操作変数を弱くすると、操作変数法の直線が不安定になり、真の直線から大きく外れることがある（弱い操作変数）。
  - type: custom
    id: line-data
    params:
      x: [1.5, 2.0, 3.0, 3.5, 4.5, 5.0]
      series:
        - { label: "観測値 y（x の小さい順）", y: [2.0, 3.4, 3.1, 4.6, 4.3, 5.5] }
        - { label: "最小二乗法の直線", y: [2.3933, 2.8, 3.6133, 4.02, 4.8333, 5.24] }
        - { label: "操作変数法の直線", y: [2.7667, 3.0667, 3.6667, 3.9667, 4.5667, 4.8667] }
      xlabel: "肥料の投入量 x"
      ylabel: "収量 y"
    caption: 例1の6戸の観測値と2本の直線。最小二乗法の直線（傾き0.8133）は操作変数法の直線（傾き0.6）より急で、肥料の効果を大きく見積もっている。
---

## 一言でいうと

説明変数 $x$ が誤差と相関している（内生性がある）とき、「$x$ には影響するが、誤差とは無関係な変数 $z$」を使って、$x$ の変動のうち誤差と無関係な部分だけから効果を推定する方法です。
この $z$ を**操作変数**といいます。

## 定義・数式

モデル $y_i=\boldsymbol{x}_i^\top\boldsymbol{\beta}+\varepsilon_i$ で、$\boldsymbol{x}_i$ の一部が内生的とします。次の2条件を満たす変数 $\boldsymbol{z}_i$（$\boldsymbol{x}_i$ のうち外生的な変数は自分自身を操作変数にする）を**操作変数**といいます。

1. **外生性（除外制約）**：$E[\boldsymbol{z}_i\varepsilon_i]=\boldsymbol{0}$。$z$ は $x$ を通してしか $y$ に影響しない。
2. **関連性**：$\boldsymbol{z}_i$ と $\boldsymbol{x}_i$ が相関する（$E[\boldsymbol{z}_i\boldsymbol{x}_i^\top]$ が列フルランク）。

操作変数の数と説明変数の数が等しい（**丁度識別**）とき、**操作変数推定量**は

$$
\hat{\boldsymbol{\beta}}_{IV}=(Z^\top X)^{-1}Z^\top\boldsymbol{y}
$$

です（$Z$ は $\boldsymbol{z}_i^\top$ を並べた行列）。単回帰 $y=\beta_0+\beta_1x+\varepsilon$ で操作変数が $z$ 1つなら

$$
\hat\beta_{1,IV}=\frac{\sum_i(z_i-\bar z)(y_i-\bar y)}{\sum_i(z_i-\bar z)(x_i-\bar x)}=\frac{s_{zy}}{s_{zx}},\qquad \hat\beta_{0,IV}=\bar y-\hat\beta_{1,IV}\bar x
$$

$z$ が0/1の2値なら、群ごとの平均の差の比 $\hat\beta_{1,IV}=(\bar y_1-\bar y_0)/(\bar x_1-\bar x_0)$ になり、これを**ワルド推定量**といいます。

操作変数のほうが多い（**過剰識別**）ときは二段階最小二乗法を使います。

## 数式の読み解き

**モーメント法としての導出**：外生性 $E[\boldsymbol{z}_i(y_i-\boldsymbol{x}_i^\top\boldsymbol{\beta})]=\boldsymbol{0}$ の標本版

$$
\frac{1}{n}Z^\top(\boldsymbol{y}-X\boldsymbol{\beta})=\boldsymbol{0}\;\Longrightarrow\;Z^\top X\boldsymbol{\beta}=Z^\top\boldsymbol{y}
$$

を解けば $\hat{\boldsymbol{\beta}}_{IV}$ です。最小二乗法の正規方程式 $X^\top X\boldsymbol{\beta}=X^\top\boldsymbol{y}$ の左側の $X^\top$ を $Z^\top$ に置き換えた形です。

単回帰では、$y=\beta_0+\beta_1x+\varepsilon$ と $z$ の共分散をとると

$$
\mathrm{Cov}(z,y)=\beta_1\mathrm{Cov}(z,x)+\mathrm{Cov}(z,\varepsilon)=\beta_1\mathrm{Cov}(z,x)
$$

なので $\beta_1=\mathrm{Cov}(z,y)/\mathrm{Cov}(z,x)$ です。直感的には「$z$ が1増えると $y$ が何増えるか」を「$z$ が1増えると $x$ が何増えるか」で割っています。

**一致性**：$\hat{\boldsymbol{\beta}}_{IV}=\boldsymbol{\beta}+(\frac1nZ^\top X)^{-1}\frac1nZ^\top\boldsymbol{\varepsilon}$ で、$\frac1nZ^\top\boldsymbol{\varepsilon}\xrightarrow{p}\boldsymbol{0}$ なので $\hat{\boldsymbol{\beta}}_{IV}\xrightarrow{p}\boldsymbol{\beta}$ です。ただし小標本では偏りがあり、不偏ではありません。

**分散**：誤差が等分散 $\sigma^2$ なら、近似的に

$$
V[\hat{\boldsymbol{\beta}}_{IV}]\approx\sigma^2(Z^\top X)^{-1}Z^\top Z(X^\top Z)^{-1}
$$

単回帰では $V[\hat\beta_{1,IV}]\approx\sigma^2/(n\,s_x^2\,r_{zx}^2)$ で、最小二乗法の分散 $\sigma^2/(n\,s_x^2)$ の $1/r_{zx}^2$ 倍です（$r_{zx}$ は $z$ と $x$ の相関係数）。相関の弱い操作変数ほど推定が不安定になります。

**弱い操作変数**：$\mathrm{Cov}(z,x)$ が0に近いと分母が小さくなり、推定値が大きく振れ、小標本の偏りも最小二乗法の方向に大きくなります。外生性がわずかに破れていても、$\mathrm{plim}\,\hat\beta_{1,IV}=\beta_1+\mathrm{Cov}(z,\varepsilon)/\mathrm{Cov}(z,x)$ の第2項が拡大されます。

## 具体例

**例1：肥料の効果（助成額を操作変数に）**

肥料の投入量 $x$ は、土地の良し悪し（誤差に含まれる）を見て農家が決めるので内生的と考えられます。抽選で決まった助成額 $z$ を操作変数にします。

| 農家 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 助成額 $z$ | 1 | 2 | 3 | 4 | 5 | 6 |
| 投入量 $x$ | 2.0 | 1.5 | 3.5 | 3.0 | 5.0 | 4.5 |
| 収量 $y$ | 3.4 | 2.0 | 4.6 | 3.1 | 5.5 | 4.3 |

$\bar z=3.5,\ \bar x=3.25,\ \bar y=3.8167$、偏差積和は $S_{zx}=11.25$、$S_{zy}=6.75$、$S_{xx}=9.375$、$S_{xy}=7.625$ です。

$$
\hat\beta_{1,OLS}=\frac{7.625}{9.375}=0.8133,\qquad
\hat\beta_{1,IV}=\frac{6.75}{11.25}=0.6000
$$

$\hat\beta_{0,IV}=3.8167-0.6\times3.25=1.8667$（最小二乗法の切片は $1.1733$）です。行列で書くと

$$
Z^\top X=\begin{pmatrix}6 & 19.5\\ 21 & 79.5\end{pmatrix},\quad Z^\top\boldsymbol{y}=\begin{pmatrix}22.9\\ 86.9\end{pmatrix},\quad (Z^\top X)^{-1}Z^\top\boldsymbol{y}=\begin{pmatrix}1.8667\\ 0.6000\end{pmatrix}
$$

で一致します。$z$ と $x$ の相関は $0.8783$ と十分あります。残差から $s^2=0.4733$ とすると、傾きの標準誤差は $0.2558$ です。
最小二乗法は、良い土地ほど肥料を多く入れる効果まで肥料の効果に含めてしまい、過大に推定しています。

**例2：ワルド推定量（クーポンの無作為配布）**

会員に無作為にクーポンを配りました（$z=1$）。サービス利用率は配布群 0.7、非配布群 0.2、1人あたり購買額（千円）は配布群 5.4、非配布群 4.4 でした。

$$
\hat\beta_{IV}=\frac{5.4-4.4}{0.7-0.2}=\frac{1.0}{0.5}=2.0
$$

クーポン自体が購買額に直接影響しない（除外制約）なら、サービス利用による購買額の増加は 2.0 千円と推定されます（効果が人によって異なる場合は、クーポンで利用を始めた人での平均効果と解釈します）。

## 現場での使われ方

- **行政・調査**：教育の賃金への効果を、生まれ月による就学年数の違いや、学校までの距離を操作変数にして推定します。
- **医療**：無作為化試験で割り付けどおりに治療を受けない人がいるとき、割り付けを操作変数にして、実際に治療を受けたことの効果を推定します。
- **マーケティング**：価格の効果を、需要とは無関係な原材料費や為替の変動を操作変数にして推定します。
- **IT・Webサービス**：A/Bテストで新機能の「表示」を無作為化し、実際に「使った」ことの効果を表示の有無を操作変数にして推定します。

## グラフ・イメージ

上の図で「操作変数の強さ」を小さくすると、操作変数法の直線が大きく揺れ、最小二乗法の直線に近づくことがあります。弱い操作変数の問題です。
下の図は例1で、最小二乗法の直線のほうが傾きが急で、$x$ の効果を大きく見積もっています。

## 試験のポイント

- 操作変数の2条件（**外生性**：$\mathrm{Cov}(z,\varepsilon)=0$、**関連性**：$\mathrm{Cov}(z,x)\neq0$）を正確に言えることが重要です。
- 単回帰の $\hat\beta_{IV}=s_{zy}/s_{zx}$ と、行列形 $(Z^\top X)^{-1}Z^\top\boldsymbol{y}$ を、モーメント条件から導けるようにします。
- 2値の操作変数ではワルド推定量 $(\bar y_1-\bar y_0)/(\bar x_1-\bar x_0)$ になります。数値を入れて計算させる問題が典型です。
- 操作変数推定量は**一致推定量だが不偏ではない**、分散は最小二乗法より大きい（$1/r_{zx}^2$ 倍）、という性質を押さえます。
- 関連性はデータ（第1段階の回帰の F 統計量など）で確かめられますが、外生性は丁度識別ではデータから検定できません。

## よくある間違い

- × 操作変数は $y$ と強く相関していればよい → ○ 必要なのは $x$ との相関と、誤差との無相関です。$x$ を経由しない経路で $y$ と関係していてはいけません。
- × 操作変数推定量は不偏 → ○ 一致性はありますが、有限標本では偏りがあります。
- × 外生性は $z$ と $x$ の相関が強ければ満たされる → ○ 外生性と関連性は別の条件です。外生性は理論や実験の設計で正当化します。
- × 弱い操作変数でも標本が大きければ問題ない → ○ 関連が弱いと漸近近似が悪く、偏りと推定のばらつきが大きくなります。
