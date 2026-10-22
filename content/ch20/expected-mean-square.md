---
id: expected-mean-square
title: 期待平均平方
reading: きたいへいきんへいほう
en: Expected mean square
aliases: [EMS, 平均平方の期待値]
chapter: 20
importance: B
tags: [分散分析, F検定, 分散成分]
related: [fixed-and-random-effects, one-way-anova, two-way-anova-with-replication, noncentral-f-distribution, split-plot-design, unbiased-estimator]
charts:
  - type: pdf
    dist: ncf
    params: { df1: 2, df2: 9, nc: 6 }
    sliders:
      - { param: nc, label: "非心度 λ = nΣα²/σ²", min: 0, max: 30, step: 0.5 }
    compare:
      - { dist: f, params: { df1: 2, df2: 9 }, label: "H₀ のもとでの F(2, 9)" }
    x: { min: 0, max: 15 }
    caption: 効果があるとき F = V_A/V_E は非心 F 分布に従い、右へずれる。非心度を大きくすると、E[V_A] が σ² から離れる様子が分布の移動として見える。
---

## 一言でいうと

分散分析表の各行の平均平方 $V$ が、平均的にどんな値になるかを、誤差分散 $\sigma^2$ と効果の大きさで表したものです。
「どの平均平方とどの平均平方の比をとれば F 検定になるか」を決める根拠になります。

## 定義・数式

平均平方 $V=S/\phi$（平方和 ÷ 自由度）の期待値 $E[V]$ を期待平均平方といいます。

一元配置（$k$ 水準、水準 $i$ に $n_i$ 個、$N=\sum n_i$、母数模型 $y_{ij}=\mu+\alpha_i+\varepsilon_{ij}$, $\sum n_i\alpha_i=0$）では

$$
E[V_E]=\sigma^2,\qquad E[V_A]=\sigma^2+\frac{\sum_i n_i\alpha_i^2}{k-1}
$$

です。

繰り返しありの二元配置（各セル $r$ 個）では、因子の種類によって次のようになります。

| 要因 | 母数模型 | 変量模型 | 混合模型（A 母数, B 変量） |
|---|---|---|---|
| A | $\sigma^2+br\,\theta_A$ | $\sigma^2+r\sigma_{AB}^2+br\sigma_A^2$ | $\sigma^2+r\sigma_{AB}^2+br\,\theta_A$ |
| B | $\sigma^2+ar\,\theta_B$ | $\sigma^2+r\sigma_{AB}^2+ar\sigma_B^2$ | $\sigma^2+ar\sigma_B^2$ |
| A×B | $\sigma^2+r\,\theta_{AB}$ | $\sigma^2+r\sigma_{AB}^2$ | $\sigma^2+r\sigma_{AB}^2$ |
| 誤差 | $\sigma^2$ | $\sigma^2$ | $\sigma^2$ |

ここで $\theta_A=\sum\alpha_i^2/(a-1)$、$\theta_B=\sum\beta_j^2/(b-1)$、$\theta_{AB}=\sum\sum(\alpha\beta)_{ij}^2/\{(a-1)(b-1)\}$ です。混合模型の B の行は、交互作用に制約をおく流儀（制約付きモデル）の形で、制約をおかない流儀では $r\sigma_{AB}^2$ も加わります。

## 数式の読み解き

**$E[V_E]=\sigma^2$ の導出**：群 $i$ の中の偏差平方和は、$n_i$ 個の独立な $N(\mu_i,\sigma^2)$ の偏差平方和なので期待値は $(n_i-1)\sigma^2$ です。足して $E[S_E]=(N-k)\sigma^2$、自由度で割って $\sigma^2$ になります。

**$E[V_A]$ の導出**：$\bar y_{i\cdot}=\mu+\alpha_i+\bar\varepsilon_{i\cdot}$、$\bar y_{\cdot\cdot}=\mu+\bar\varepsilon_{\cdot\cdot}$（$\sum n_i\alpha_i=0$ より）です。

$$
S_A=\sum_i n_i(\alpha_i+\bar\varepsilon_{i\cdot}-\bar\varepsilon_{\cdot\cdot})^2
=\sum_i n_i\alpha_i^2+2\sum_i n_i\alpha_i(\bar\varepsilon_{i\cdot}-\bar\varepsilon_{\cdot\cdot})+\sum_i n_i(\bar\varepsilon_{i\cdot}-\bar\varepsilon_{\cdot\cdot})^2
$$

真ん中の項の期待値は0です。最後の項は「誤差だけで作った群間平方和」で、期待値は $(k-1)\sigma^2$ になります（$\sum n_i\bar\varepsilon_{i\cdot}^2$ の期待値が $k\sigma^2$、$N\bar\varepsilon_{\cdot\cdot}^2$ の期待値が $\sigma^2$ なので差が $(k-1)\sigma^2$）。よって

$$
E[S_A]=\sum_i n_i\alpha_i^2+(k-1)\sigma^2,\qquad E[V_A]=\sigma^2+\frac{\sum_i n_i\alpha_i^2}{k-1}
$$

**F 検定の分母の選び方**：帰無仮説のもとで期待値が一致する2つの平均平方の比をとります。

- 母数模型：どの効果も $V_E$ で割ります。
- 変量模型：A の検定は $V_A/V_{A\times B}$（$\sigma_A^2=0$ なら両方とも $\sigma^2+r\sigma_{AB}^2$）。
- 分割法：1次単位の因子は1次誤差、2次単位の因子は2次誤差で割ります。

**分散成分の推定**：期待平均平方の式を連立して解き、期待値を実際の平均平方で置き換えます。たとえば変量模型の二元配置では $\hat\sigma_{AB}^2=(V_{A\times B}-V_E)/r$ です。

## 具体例

**例1：期待平均平方を数値で確かめる**

$k=3$、各 $n=4$、$\alpha=(-2,1,1)$、$\sigma^2=4$ の母数模型では

$$
E[V_A]=4+\frac{4\times(4+1+1)}{2}=4+12=16,\qquad E[V_E]=4
$$

です。この設定で20万回のシミュレーションをすると、$V_A$ の平均は 15.98、$V_E$ の平均は 3.99 になり、理論値と一致します。F 統計量は非心度 $\lambda=\sum n_i\alpha_i^2/\sigma^2=24/4=6$ の非心 F 分布 $F'(2,9;6)$ に従い、5% 水準の検出力は 0.441 です。

**例2：変量模型の二元配置で分散成分を推定する**

$a=3$, $b=2$, $r=2$ の二元配置で $V_A=65.333$, $V_B=56.333$, $V_{A\times B}=17.333$, $V_E=2.000$ が得られたとし、両因子を変量とみます。

$$
\hat\sigma_{AB}^2=\frac{17.333-2.000}{2}=7.667,\quad
\hat\sigma_A^2=\frac{65.333-17.333}{2\times2}=12.000,\quad
\hat\sigma_B^2=\frac{56.333-17.333}{3\times2}=6.500
$$

## 現場での使われ方

- **製造・品質管理**：ロット・作業者・繰り返し測定などの変量因子の分散成分を期待平均平方から推定し、どの段階のばらつきを減らすべきかを決めます。
- **農業・生物**：圃場試験の設計で、どの平均平方を分母にするか（どの誤差で検定するか）を事前に確認します。
- **研究計画（医療など）**：期待平均平方から非心度を計算し、検出力が十分になるように繰り返し数を決めます。

## グラフ・イメージ

効果があるときの F 統計量の分布（非心 F 分布）と、効果がないときの F 分布を重ねています。
非心度のスライダーを0にすると2つが重なり、大きくすると分布が右へ移ります。$E[V_A]$ が $\sigma^2$ より大きくなる分だけ F が右にずれる、というのが F 検定の仕組みです。

## 試験のポイント

- 一元配置の $E[V_A]$, $E[V_E]$ は導出も含めて重要です。
- 変量模型・混合模型の二元配置で、どの平均平方を F の分母にするかを期待平均平方の表から判断させる問題が典型です。
- 分散成分の推定値（$\hat\sigma_A^2=(V_A-V_E)/n$ など）を表から計算できるようにします。
- 非心度 $\lambda$ は $E[V_A]$ の「$\sigma^2$ を超える部分」に比例します。検出力の計算に使います。

## よくある間違い

- × $E[V_A]=\sigma^2+\sum\alpha_i^2$ と $n_i$ を忘れる → ○ 各群の $n_i$ 個の観測が同じ効果をもつので、$n_i$ を掛けた和を自由度で割ります。
- × 期待平均平方を「平均平方の観測値」と混同する → ○ 期待平均平方は理論的な期待値で、観測された $V$ はその推定値です。
- × 変量模型でも主効果を $V_E$ で割る → ○ 交互作用の分散成分が含まれるので、$V_{A\times B}$ で割ります。
