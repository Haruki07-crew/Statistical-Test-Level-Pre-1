---
id: nonlinear-regression
title: 非線形回帰
reading: ひせんけいかいき
en: Nonlinear regression
aliases: [非線形最小二乗法, ガウス・ニュートン法, 成長曲線]
chapter: 19
importance: B
tags: [回帰分析, 非線形最小二乗法, 成長曲線, 反復計算]
related: [least-squares-method, polynomial-regression, logistic-regression, log-transformation, generalized-linear-model]
charts:
  - type: custom
    id: line-data
    params:
      x: [0, 1, 2, 3, 4, 5]
      series:
        - { label: "観測値 y", y: [10.3, 6.0, 3.9, 2.1, 1.5, 0.8] }
        - { label: "非線形最小二乗の曲線", y: [10.245, 6.176, 3.723, 2.244, 1.353, 0.815] }
        - { label: "対数変換した回帰の曲線", y: [10.197, 6.175, 3.74, 2.265, 1.371, 0.831] }
      xlabel: "経過時間 x（時間）"
      ylabel: "血中濃度 y"
    caption: 例の指数減衰モデル。非線形最小二乗法の曲線は元の尺度での残差平方和を最小にし、対数変換して直線を当てはめた曲線とはわずかに異なる。
---

## 一言でいうと

目的変数の平均が、パラメータについて線形でない式（指数関数、ロジスティック曲線など）で表されるモデルを、残差平方和の最小化で当てはめる方法です。
解を式で書けないので、初期値から反復計算で求めます。

## 定義・数式

$$
y_i=f(\boldsymbol{x}_i;\boldsymbol{\theta})+\varepsilon_i,\qquad \varepsilon_i\ \text{は独立で}\ E[\varepsilon_i]=0,\ V[\varepsilon_i]=\sigma^2\quad(i=1,\dots,n)
$$

で、$f$ がパラメータ $\boldsymbol{\theta}$ について非線形なものを**非線形回帰モデル**といいます。推定値は

$$
\hat{\boldsymbol{\theta}}=\arg\min_{\boldsymbol{\theta}}S(\boldsymbol{\theta}),\qquad S(\boldsymbol{\theta})=\sum_{i=1}^{n}\{y_i-f(\boldsymbol{x}_i;\boldsymbol{\theta})\}^2
$$

です（非線形最小二乗法）。誤差が正規分布なら最尤推定量と一致します。

代表的なモデル（$t$ は時間）：

| 名前 | 式 | 特徴 |
|---|---|---|
| 指数減衰 | $a e^{-bt}$ | 薬物濃度、放射性崩壊 |
| ロジスティック曲線 | $\dfrac{K}{1+e^{-r(t-t_0)}}$ | 上限 $K$ に向かうS字、変曲点は $K/2$ |
| ゴンペルツ曲線 | $K\exp\{-b\,e^{-ct}\}$ | 非対称なS字、変曲点は $K/e$ |
| ミカエリス・メンテン | $\dfrac{V x}{k+x}$ | 酵素反応速度 |

「$x^2$ を含む多項式」はパラメータについて線形なので、非線形回帰ではありません（通常の重回帰で解けます）。

## 数式の読み解き

**ガウス・ニュートン法**：現在の値 $\boldsymbol{\theta}^{(k)}$ のまわりで $f$ を1次近似します。

$$
f(\boldsymbol{x}_i;\boldsymbol{\theta})\approx f(\boldsymbol{x}_i;\boldsymbol{\theta}^{(k)})+\boldsymbol{j}_i^\top(\boldsymbol{\theta}-\boldsymbol{\theta}^{(k)}),\qquad
\boldsymbol{j}_i=\left.\frac{\partial f(\boldsymbol{x}_i;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\right|_{\boldsymbol{\theta}^{(k)}}
$$

すると残差 $r_i=y_i-f(\boldsymbol{x}_i;\boldsymbol{\theta}^{(k)})$ を、ヤコビ行列 $J$（第 $i$ 行が $\boldsymbol{j}_i^\top$）で線形回帰する問題になり、

$$
\boldsymbol{\theta}^{(k+1)}=\boldsymbol{\theta}^{(k)}+(J^\top J)^{-1}J^\top\boldsymbol{r}
$$

をくり返します。$J$ が通常の回帰の $X$ の役割をします。収束が不安定なときは、$J^\top J$ の対角に正の数を足すレーベンバーグ・マーカート法を使います。

**推測**：大標本では $\hat{\boldsymbol{\theta}}$ は近似的に $N(\boldsymbol{\theta},\ \sigma^2(J^\top J)^{-1})$ に従い、$\sigma^2$ は $S(\hat{\boldsymbol{\theta}})/(n-q)$（$q$ はパラメータ数）で推定します。線形回帰と違い、これは近似です。

**線形化との違い**：$y=ae^{-bx}$ は対数をとると $\log y=\log a-bx$ と直線になります。ただし、これは誤差が掛け算型（$\log$ の尺度で等分散）のときに適した方法で、加法的な誤差のモデルとは推定値が異なります。

## 具体例

**例：薬の血中濃度の減衰**

投与後 $x=0,1,\dots,5$ 時間の濃度 $y=(10.3,\ 6.0,\ 3.9,\ 2.1,\ 1.5,\ 0.8)$ に $y=ae^{-bx}+\varepsilon$ を当てはめます。

初期値 $(a,b)=(10,\ 0.5)$ では残差平方和 $S=0.1824$ です。ヤコビ行列の列は $\partial f/\partial a=e^{-bx}$、$\partial f/\partial b=-axe^{-bx}$ で、

$$
J^\top J=\begin{pmatrix}1.5781 & -8.9486\\ -8.9486 & 181.8803\end{pmatrix},\qquad
J^\top\boldsymbol{r}=\begin{pmatrix}0.3306\\ -1.0608\end{pmatrix}
$$

これを解くと更新量は $(0.2447,\ 0.0062)$、新しい値は $(10.2447,\ 0.5062)$ です。もう1回の反復でほぼ動かなくなり、

$$
\hat a=10.2452,\qquad \hat b=0.5062,\qquad S(\hat{\boldsymbol{\theta}})=0.1080
$$

に収束します。$s^2=0.1080/(6-2)=0.0270$、近似的な標準誤差は $a$ が $0.1543$、$b$ が $0.0142$ です。
半減期は $\log2/\hat b=1.369$ 時間です。

対数をとって直線を当てはめると $(10.1968,\ 0.5016)$ になり、元の尺度での残差平方和は $0.1116$ と少し大きくなります。

## 現場での使われ方

- **医療**：薬物動態の解析で、血中濃度の時間変化に指数関数の和（コンパートメントモデル）を当てはめ、半減期やクリアランスを推定します。
- **マーケティング**：新製品の累計普及数にロジスティック曲線やゴンペルツ曲線を当てはめ、普及の上限と時期を予測します。
- **製造・品質管理**：信頼性試験で、劣化量の時間変化を非線形モデルで表し、寿命を推定します。
- **行政・調査**：感染者数の初期の増え方や、人口の成長曲線の推定に使います。

## グラフ・イメージ

観測値と2本の曲線を重ねています。非線形最小二乗法の曲線は、値の大きい $x=0$ 付近の点にもよく合っています。
対数変換による当てはめは、値の小さい右側の点を相対的に重く扱うため、曲線がわずかに異なります。

## 試験のポイント

- 「パラメータについて線形かどうか」で線形・非線形を区別します。$\beta_0+\beta_1x+\beta_2x^2$ は線形モデルです。
- ガウス・ニュートン法の更新式 $\boldsymbol{\theta}+(J^\top J)^{-1}J^\top\boldsymbol{r}$ と、それが「1次近似した線形回帰をくり返す」ことを説明できるようにします。
- ロジスティック曲線の上限 $K$・変曲点 $K/2$、ゴンペルツ曲線の変曲点 $K/e$ など、曲線の性質が問われます。
- 対数変換で線形化できるモデルでも、誤差の入り方（加法か乗法か）で適した推定法が変わります。

## よくある間違い

- × 説明変数に $x^2$ や $\log x$ を含めば非線形回帰 → ○ パラメータについて線形なら線形回帰です。
- × 反復計算は初期値によらず同じ解に収束する → ○ 局所解に収束したり発散したりすることがあります。線形化した回帰などで妥当な初期値を作ります。
- × 標準誤差 $\sqrt{s^2(J^\top J)^{-1}}$ は正確 → ○ 1次近似にもとづく大標本での近似です。
