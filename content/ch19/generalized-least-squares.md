---
id: generalized-least-squares
title: 一般化最小二乗法
reading: いっぱんかさいしょうにじょうほう
en: Generalized least squares
aliases: [GLS, 一般化最小二乗推定量, エイトケン推定量, Aitken estimator]
chapter: 19
importance: A
tags: [回帰分析, 系列相関, 不均一分散, 共分散行列, BLUE]
related: [weighted-least-squares, least-squares-method, gauss-markov-theorem, serial-correlation, heteroscedasticity, durbin-watson-statistic]
charts:
  - type: custom
    id: dw-sim
    params: { phi: 0.6, n: 30 }
    sliders:
      - { param: phi, label: "誤差の自己相関 φ", min: -0.9, max: 0.9, step: 0.1 }
    caption: AR(1) 誤差をもつ回帰の残差を時間順に並べた図。φ が大きいと同じ符号が続き、誤差の共分散行列が単位行列の定数倍ではなくなる。
  - type: custom
    id: bar-data
    params:
      labels: ["y1", "y2", "y3", "y4", "y5"]
      values: [0.3125, 0.125, 0.125, 0.125, 0.3125]
      xlabel: "観測"
      ylabel: "GLS での重み"
    caption: 例1（AR(1)、ρ=0.6、n=5）で平均を推定するときの GLS の重み。OLS ならどれも 0.2 だが、GLS では両端の観測が重くなる。
---

## 一言でいうと

誤差の分散がそろっていない、あるいは誤差どうしに相関があるときに、その共分散の構造を使って最良の推定を行う最小二乗法です。
データを「誤差が独立で等分散になるように」変換してから最小二乗法をかける、と考えます。

## 定義・数式

重回帰モデル $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$（$X$ は列フルランク）で、

$$
E[\boldsymbol{\varepsilon}]=\boldsymbol{0},\qquad V[\boldsymbol{\varepsilon}]=\sigma^2\Omega\quad(\Omega\ \text{は既知の}\ n\times n\ \text{正定値対称行列})
$$

とします。**一般化最小二乗推定量**は

$$
S_\Omega(\boldsymbol{\beta})=(\boldsymbol{y}-X\boldsymbol{\beta})^\top\Omega^{-1}(\boldsymbol{y}-X\boldsymbol{\beta})
$$

を最小にする $\boldsymbol{\beta}$ で、

$$
\hat{\boldsymbol{\beta}}_{G}=(X^\top\Omega^{-1}X)^{-1}X^\top\Omega^{-1}\boldsymbol{y},\qquad
E[\hat{\boldsymbol{\beta}}_{G}]=\boldsymbol{\beta},\qquad
V[\hat{\boldsymbol{\beta}}_{G}]=\sigma^2(X^\top\Omega^{-1}X)^{-1}
$$

です（エイトケンの定理：$\hat{\boldsymbol{\beta}}_G$ は線形不偏推定量の中で分散最小、すなわち BLUE）。
$\Omega=I$ なら通常の最小二乗法（OLS）、$\Omega$ が対角行列なら加重最小二乗法（WLS）です。

## 数式の読み解き

- $\Omega$：誤差の相関・分散の「形」。対角成分が分散の比、非対角成分が相関を表します。
- $\Omega^{-1}$：重み行列。分散の大きい観測や、他と強く相関して情報が重複している観測を軽く扱います。

**導出：白色化して OLS に帰着させる**

$\Omega$ は正定値なので、$P^\top P=\Omega^{-1}$ となる正則行列 $P$ がとれます（コレスキー分解など）。このとき $P\Omega P^\top=I$ です（$\Omega=P^{-1}(P^\top)^{-1}$ より）。モデルに左から $P$ をかけると

$$
P\boldsymbol{y}=PX\boldsymbol{\beta}+P\boldsymbol{\varepsilon},\qquad V[P\boldsymbol{\varepsilon}]=\sigma^2P\Omega P^\top=\sigma^2I
$$

変換後の誤差は無相関・等分散なので、ガウス・マルコフの定理が使えます。$\boldsymbol{y}^*=P\boldsymbol{y},\ X^*=PX$ に OLS をかけると

$$
\hat{\boldsymbol{\beta}}_G=(X^{*\top}X^*)^{-1}X^{*\top}\boldsymbol{y}^*=(X^\top P^\top PX)^{-1}X^\top P^\top P\boldsymbol{y}=(X^\top\Omega^{-1}X)^{-1}X^\top\Omega^{-1}\boldsymbol{y}
$$

分散は $\sigma^2(X^{*\top}X^*)^{-1}=\sigma^2(X^\top\Omega^{-1}X)^{-1}$ です。

**OLS を使い続けるとどうなるか**：OLS も不偏ですが、分散は

$$
V[\hat{\boldsymbol{\beta}}_{OLS}]=\sigma^2(X^\top X)^{-1}X^\top\Omega X(X^\top X)^{-1}
$$

で、GLS 以上になります。さらに、ソフトの出す $\sigma^2(X^\top X)^{-1}$ はこれと一致しないので、標準誤差が誤ります。

**AR(1) 誤差の場合**：$\varepsilon_t=\rho\varepsilon_{t-1}+u_t$ なら $\Omega_{ij}=\rho^{|i-j|}$ です。$P$ は「1行目に $\sqrt{1-\rho^2}$、2行目以降は $y_t-\rho y_{t-1}$」という差分変換（プレイス・ウィンステン変換）になります。

**$\Omega$ が未知のとき**は、残差から $\Omega$（たとえば $\rho$）を推定して代入します。これを**実行可能な GLS**（FGLS）といいます。

## 具体例

**例1：自己相関のある測定値から平均を推定する**

1時間ごとの測定値 $y=(10.2,\ 11.0,\ 10.6,\ 9.4,\ 9.8)$ を、$y_t=\mu+\varepsilon_t$、誤差は AR(1) で $\rho=0.6$、$V[\varepsilon_t]=\sigma^2$ とします。$X=\boldsymbol{1}$ なので

$$
\hat\mu_G=\frac{\boldsymbol{1}^\top\Omega^{-1}\boldsymbol{y}}{\boldsymbol{1}^\top\Omega^{-1}\boldsymbol{1}}
$$

AR(1) の $\Omega^{-1}$ は三重対角で、$(1-\rho^2)\Omega^{-1}$ の対角は $(1,\ 1.36,\ 1.36,\ 1.36,\ 1)$、隣どうしは $-0.6$ です。$\boldsymbol{1}^\top\Omega^{-1}=(0.625,\ 0.25,\ 0.25,\ 0.25,\ 0.625)$、合計は $2$ なので、重みは

$$
(0.3125,\ 0.125,\ 0.125,\ 0.125,\ 0.3125)
$$

$$
\hat\mu_G=0.3125\times(10.2+9.8)+0.125\times(11.0+10.6+9.4)=6.25+3.875=10.125
$$

単純平均（OLS）は $10.2$ です。分散は $V[\hat\mu_G]=\sigma^2/2=0.5\,\sigma^2$、単純平均は $\boldsymbol{1}^\top\Omega\boldsymbol{1}/25\cdot\sigma^2=0.5233\,\sigma^2$ です。
注意すべきは、独立を仮定した公式 $\sigma^2/5=0.2\,\sigma^2$ が本当の分散の半分以下になることです。
プレイス・ウィンステン変換で $y^*=(8.16,\ 4.88,\ 4.00,\ 3.04,\ 4.16)$、$x^*=(0.8,\ 0.4,\ 0.4,\ 0.4,\ 0.4)$ とし、原点を通る OLS をかけても $\sum x^*y^*/\sum x^{*2}=10.125$ が得られます。

**例2：GLS と OLS が一致する場合**

誤差がどの2つの間でも相関 $r=0.3$（等相関）のとき、$\boldsymbol{1}^\top\Omega^{-1}$ の成分はすべて等しく（$n=5$ で $0.4545$）、平均の GLS 推定量は単純平均と一致します。
$\Omega$ が単位行列でなくても、$X$ の列が $\Omega$ の固有ベクトルで張られる空間に入っていれば GLS と OLS は一致します。

## 現場での使われ方

- **金融・保険**：月次データの回帰で誤差に自己相関があるとき、AR(1) を仮定した FGLS（コクラン・オーカット法など）で係数を推定します。
- **製造・品質管理**：同じ装置で連続して測った値は誤差が相関しやすく、その相関を考慮して校正直線を引きます。
- **医療**：同じ患者をくり返し測定した経時データで、患者内の相関構造を指定した GLS（混合モデルの一部）を使います。
- **行政・調査**：地域データで隣接地域の誤差が相関する場合、空間的な共分散を $\Omega$ に入れて推定します。

## グラフ・イメージ

上の図で $\phi$ を大きくすると残差が同じ符号で長く続き、小さく（負に）すると符号が交互になります。どちらも $\Omega\neq I$ の状態です。
下の図は例1の重みで、AR(1) で正の相関があると、情報の重複が少ない両端の観測が重くなります。

## 試験のポイント

- $\hat{\boldsymbol{\beta}}_G=(X^\top\Omega^{-1}X)^{-1}X^\top\Omega^{-1}\boldsymbol{y}$ と $V=\sigma^2(X^\top\Omega^{-1}X)^{-1}$ を、$P^\top P=\Omega^{-1}$ となる変換から導けることが重要です。
- WLS は $\Omega$ が対角の特別な場合、OLS は $\Omega=I$ の場合、という包含関係を押さえます。
- 誤差に系列相関があっても OLS は不偏ですが、BLUE ではなく、通常の標準誤差は誤っています（正の自己相関ならふつう過小）。
- AR(1) 誤差の差分変換 $y_t-\rho y_{t-1}$ と、$\rho$ を推定して用いる FGLS の流れを説明できるようにします。

## よくある間違い

- × GLS の重み行列は $\Omega$ → ○ 重みは逆行列 $\Omega^{-1}$ です。
- × 誤差に相関があると OLS は偏る → ○ 説明変数が外生的なら OLS は不偏のままです。失われるのは効率と標準誤差の正しさです（説明変数に遅れの $y$ が入る場合は別で、偏りも生じます）。
- × 変換行列 $P$ は $\Omega^{1/2}$ → ○ $P^\top P=\Omega^{-1}$、つまり $\Omega^{-1/2}$ の側です。
- × FGLS も小標本で BLUE → ○ 推定した $\hat\Omega$ を使うと推定量は非線形になり、BLUE の保証は大標本での近似的なものになります。
