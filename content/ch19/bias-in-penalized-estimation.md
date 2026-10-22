---
id: bias-in-penalized-estimation
title: 罰則付き推定量のバイアス
reading: ばっそくつきすいていりょうのばいあす
en: Bias of penalized estimators
aliases: [リッジ推定量のバイアス, 縮小推定量のバイアス, 正則化のバイアス]
chapter: 19
importance: B
tags: [正則化, 平均二乗誤差, バイアス・バリアンス, 縮小推定]
related: [ridge-regression, lasso, regularization, bias-variance-tradeoff, mean-squared-error, unbiased-estimator]
charts:
  - type: custom
    id: line-data
    params:
      x: [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
      series:
        - { label: "バイアスの2乗", y: [0, 0.04, 0.1111, 0.1837, 0.25, 0.36, 0.4444, 0.5625] }
        - { label: "分散", y: [0.5, 0.32, 0.2222, 0.1633, 0.125, 0.08, 0.0556, 0.0312] }
        - { label: "MSE", y: [0.5, 0.36, 0.3333, 0.3469, 0.375, 0.44, 0.5, 0.5938] }
      xlabel: "λ"
      ylabel: "値"
    caption: 例1（直交計画、β=1、σ²=0.5）のリッジ推定量。λ=0 の MSE（＝OLS の分散 0.5）から下がり、λ=σ²/β²=0.5 で最小 0.3333 になる。λ=2 で OLS と同じ水準に戻る。
---

## 一言でいうと

リッジ回帰や Lasso のような罰則付き推定量は、係数を0の方向へ引き寄せるので、期待値が真の値からずれます（偏りがある）。
その代わり分散が減り、適度な罰則なら平均二乗誤差（MSE）は最小二乗推定量より小さくなります。

## 定義・数式

推定量 $\hat{\boldsymbol{\beta}}$ のバイアスは $E[\hat{\boldsymbol{\beta}}]-\boldsymbol{\beta}$、MSE は

$$
\mathrm{MSE}(\hat{\boldsymbol{\beta}})=E\|\hat{\boldsymbol{\beta}}-\boldsymbol{\beta}\|^2=\|E[\hat{\boldsymbol{\beta}}]-\boldsymbol{\beta}\|^2+\mathrm{tr}\,V[\hat{\boldsymbol{\beta}}]
$$

です。線形モデル $\boldsymbol{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}$、$V[\boldsymbol{\varepsilon}]=\sigma^2I$ のリッジ推定量 $\hat{\boldsymbol{\beta}}_R=(X^\top X+\lambda I)^{-1}X^\top\boldsymbol{y}$ では

$$
\mathrm{Bias}=-\lambda(X^\top X+\lambda I)^{-1}\boldsymbol{\beta},\qquad
V[\hat{\boldsymbol{\beta}}_R]=\sigma^2(X^\top X+\lambda I)^{-1}X^\top X(X^\top X+\lambda I)^{-1}
$$

**直交計画**（$X^\top X=I$）では係数ごとに

$$
E[\hat\beta_{R,j}]=\frac{\beta_j}{1+\lambda},\quad
\mathrm{Bias}=-\frac{\lambda\beta_j}{1+\lambda},\quad
V=\frac{\sigma^2}{(1+\lambda)^2},\quad
\mathrm{MSE}_j(\lambda)=\frac{\sigma^2+\lambda^2\beta_j^2}{(1+\lambda)^2}
$$

です。**ホーエルとケナードの定理**：任意の $\boldsymbol{\beta}$ と $\sigma^2>0$ に対し、$\mathrm{MSE}(\hat{\boldsymbol{\beta}}_R(\lambda))<\mathrm{MSE}(\hat{\boldsymbol{\beta}}_{OLS})$ となる $\lambda>0$ が存在します。

## 数式の読み解き

**直交計画の MSE の導出**：OLS 推定値 $b_j$ は $N(\beta_j,\sigma^2)$ に従い、$\hat\beta_{R,j}=b_j/(1+\lambda)$ です。期待値は $\beta_j/(1+\lambda)$、分散は $\sigma^2/(1+\lambda)^2$ なので、

$$
\mathrm{MSE}_j=\left(\frac{\beta_j}{1+\lambda}-\beta_j\right)^2+\frac{\sigma^2}{(1+\lambda)^2}=\frac{\lambda^2\beta_j^2+\sigma^2}{(1+\lambda)^2}
$$

**最適な $\lambda$**：$\lambda$ で微分すると

$$
\frac{d\,\mathrm{MSE}_j}{d\lambda}=\frac{2\lambda\beta_j^2(1+\lambda)-2(\sigma^2+\lambda^2\beta_j^2)}{(1+\lambda)^3}=\frac{2(\lambda\beta_j^2-\sigma^2)}{(1+\lambda)^3}
$$

で、$\lambda^*=\sigma^2/\beta_j^2$ のとき最小、最小値は $\sigma^2\beta_j^2/(\sigma^2+\beta_j^2)$ です。
$\lambda=0$ での傾きは $-2\sigma^2<0$ なので、少しでも罰則をかければ MSE は必ず下がります。これが定理の直感です。

**Lasso のバイアス**：直交計画のソフト閾値 $\mathrm{sign}(b)(|b|-\lambda)_+$ では、

- 真の係数が大きい（$|\beta_j|\gg\lambda$）とき、ほぼ確実に生き残り、バイアスは約 $-\lambda\,\mathrm{sign}(\beta_j)$ で一定です。
- 真の係数が小さいときは0に落ちやすく、バイアスは $-\beta_j$ に近づきます。

リッジのバイアス $-\lambda\beta_j/(1+\lambda)$ は係数に比例し、Lasso は大きな係数でも一定量の偏りが残る点が違います。

## 具体例

**例1：リッジの最適な罰則（直交計画）**

$\beta=1$、$\sigma^2=0.5$ とします。OLS の MSE は分散 $0.5$ です。

| $\lambda$ | バイアスの2乗 | 分散 | MSE |
|---|---|---|---|
| 0 | 0 | 0.5 | 0.5 |
| 0.25 | 0.04 | 0.32 | 0.36 |
| 0.5 | 0.1111 | 0.2222 | 0.3333 |
| 1 | 0.25 | 0.125 | 0.375 |
| 2 | 0.4444 | 0.0556 | 0.5 |

$\lambda^*=\sigma^2/\beta^2=0.5$ で最小値 $0.5\times1/(0.5+1)=0.3333$、OLS より3分の1小さくなります。$\lambda=2$ では OLS と同じ、それ以上では悪化します。

**例2：Lasso の偏り（直交計画、$\sigma=1$、$\lambda=1$）**

$b\sim N(\beta,1)$ にソフト閾値をかけた推定量の期待値を数値積分すると、

| 真の $\beta$ | 0.5 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| バイアス | -0.3315 | -0.6095 | -0.9171 | -0.9915 | -1.0000 |
| MSE | 0.314 | 0.7249 | 1.5931 | 1.9433 | 1.9999 |

真の係数が大きいほどバイアスは $-\lambda=-1$ に近づき、MSE は OLS の $1$ の約2倍になります。一方 $\beta=0$ では MSE が $0.1507$ で、OLS より大幅に小さくなります。

## 現場での使われ方

- **医療**：Lasso で選んだ変数の効果の大きさを報告するとき、縮小の偏りを避けるため、選ばれた変数だけで当てはめ直す（リラックスド Lasso など）ことがあります。
- **金融・保険**：少数データの区分の料率を全体の方向へ縮めると、偏りは入りますが年ごとの料率の変動（分散）が小さくなります。
- **マーケティング**：広告効果の推定でリッジを使うと効果が小さめに出るので、予算配分の判断ではその偏りを念頭に置きます。

## グラフ・イメージ

例1で $\lambda$ を増やすと、バイアスの2乗は0から増え、分散は $0.5$ から減ります。
MSE は $\lambda=0.5$ で底を打ち、$\lambda=2$ で OLS と同じ高さに戻ります。「罰則は弱すぎても強すぎてもいけない」ことが読み取れます。

## 試験のポイント

- リッジ推定量の期待値 $(X^\top X+\lambda I)^{-1}X^\top X\boldsymbol{\beta}$ とバイアス $-\lambda(X^\top X+\lambda I)^{-1}\boldsymbol{\beta}$ を導けるようにします。
- 直交計画での $\mathrm{MSE}(\lambda)=(\sigma^2+\lambda^2\beta^2)/(1+\lambda)^2$ と、最適な $\lambda^*=\sigma^2/\beta^2$ を計算させる問題が重要です。
- 「不偏推定量の中で最良（ガウス・マルコフ）」と「偏りを許せばもっと MSE を小さくできる」は矛盾しない、という整理が問われます。
- Lasso のバイアスは大きな係数で約 $\lambda$ に留まり、リッジのバイアスは係数に比例する、という違いを押さえます。

## よくある間違い

- × OLS は BLUE なので、どんな推定量より MSE が小さい → ○ BLUE は「線形かつ不偏」の中での最良です。偏りのある推定量なら MSE で OLS に勝てます。
- × 最適な $\lambda^*=\sigma^2/\beta^2$ をそのまま使える → ○ $\beta$ と $\sigma^2$ は未知なので実際には使えません。交差検証などで選びます。
- × Lasso で残った係数は偏りがない → ○ 生き残った係数も $\lambda$ だけ縮められています。
