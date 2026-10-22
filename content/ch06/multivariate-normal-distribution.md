---
id: multivariate-normal-distribution
title: 多変量正規分布
reading: たへんりょうせいきぶんぷ
en: Multivariate normal distribution
aliases: [多次元正規分布, "N_p(μ,Σ)", Multivariate Gaussian]
chapter: 6
importance: A
tags: [多次元分布, 分散共分散行列, マハラノビス距離, 線形変換]
related: [bivariate-normal-distribution, variance-covariance-matrix, mean-vector, mahalanobis-distance, chi-squared-distribution, precision-matrix]
charts:
  - type: scatter
    model: bvnorm
    params: { rho: 0.5, n: 400 }
    sliders:
      - { param: rho, label: "相関係数 ρ", min: -0.95, max: 0.95, step: 0.05 }
    caption: 多変量正規分布を2つの変数に絞って見た散布図。点の雲の等高線は楕円で、その形と向きは分散共分散行列 Σ で決まる。
  - type: pdf
    dist: chi2
    params: { df: 2 }
    sliders:
      - { param: df, label: "次元 p（自由度）", min: 1, max: 10, step: 1 }
    x: { min: 0, max: 25 }
    caption: マハラノビス距離の2乗 (X−μ)ᵀΣ⁻¹(X−μ) の分布は自由度 p のカイ二乗分布。次元 p が大きいほど、中心から遠い点が普通になる。
---

## 一言でいうと

正規分布を $p$ 個の変数に拡張した同時分布で、平均ベクトル $\boldsymbol\mu$ と分散共分散行列 $\Sigma$ だけで決まります。
線形変換・周辺化・条件付けのどれをしても正規分布のままという扱いやすさから、回帰分析、主成分分析、判別分析など多変量解析のほぼすべての手法の前提になっています。

## 定義・数式

$\boldsymbol\mu\in\mathbb{R}^p$、$\Sigma$ を $p\times p$ の正定値対称行列とします。$\boldsymbol X=(X_1,\dots,X_p)^\top$ の同時密度が

$$
f(\boldsymbol x)=\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}\exp\!\left\{-\frac12(\boldsymbol x-\boldsymbol\mu)^\top\Sigma^{-1}(\boldsymbol x-\boldsymbol\mu)\right\}
$$

のとき $\boldsymbol X\sim N_p(\boldsymbol\mu,\Sigma)$ と書きます。同値な定義として「任意の定数ベクトル $\boldsymbol a$ について $\boldsymbol a^\top\boldsymbol X$ が1次元正規分布に従う」とも言えます。

| 項目 | 内容 |
|---|---|
| 台 | $\mathbb{R}^p$ |
| パラメータ | 平均ベクトル $\boldsymbol\mu$、分散共分散行列 $\Sigma$（正定値） |
| 期待値 | $E[\boldsymbol X]=\boldsymbol\mu$ |
| 分散共分散行列 | $V[\boldsymbol X]=\Sigma$ |
| モーメント母関数 | $M(\boldsymbol t)=\exp\!\left(\boldsymbol t^\top\boldsymbol\mu+\tfrac12\boldsymbol t^\top\Sigma\boldsymbol t\right)$ |
| 特性関数 | $\varphi(\boldsymbol t)=\exp\!\left(i\boldsymbol t^\top\boldsymbol\mu-\tfrac12\boldsymbol t^\top\Sigma\boldsymbol t\right)$ |
| 線形変換 | $A\boldsymbol X+\boldsymbol b\sim N_q(A\boldsymbol\mu+\boldsymbol b,\ A\Sigma A^\top)$（$A$ は $q\times p$） |
| 2次形式 | $(\boldsymbol X-\boldsymbol\mu)^\top\Sigma^{-1}(\boldsymbol X-\boldsymbol\mu)\sim\chi^2(p)$ |

## 数式の読み解き

- $(\boldsymbol x-\boldsymbol\mu)^\top\Sigma^{-1}(\boldsymbol x-\boldsymbol\mu)$：**マハラノビス距離**の2乗。ばらつきの大きい方向・相関の向きを考慮した「中心からの遠さ」です。$p=1$ では $(x-\mu)^2/\sigma^2$ になり、1次元の正規分布の指数部分と一致します。
- $|\Sigma|^{1/2}$：分布の広がり（楕円体の体積）に応じた正規化定数です。
- 等高線はマハラノビス距離が一定の点の集まりで、楕円（楕円体）になります。

**主な性質**。$\boldsymbol X=(\boldsymbol X_1^\top,\boldsymbol X_2^\top)^\top$ と分割し、$\Sigma$ も対応して $\Sigma_{11},\Sigma_{12},\Sigma_{21},\Sigma_{22}$ に分けます。

- **周辺分布**：$\boldsymbol X_1\sim N(\boldsymbol\mu_1,\Sigma_{11})$。必要な成分を取り出すだけです。
- **条件付き分布**：

$$
\boldsymbol X_1\mid\boldsymbol X_2=\boldsymbol x_2\sim N\!\left(\boldsymbol\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(\boldsymbol x_2-\boldsymbol\mu_2),\ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\right)
$$

条件付き分散は $\boldsymbol x_2$ によりません。$p=2$ にすると2変量正規分布の公式に戻ります。
- **独立性**：$\Sigma_{12}=O$（ブロックどうしが無相関）なら $\boldsymbol X_1$ と $\boldsymbol X_2$ は独立です。
- **2次形式がカイ二乗になる理由**：$\Sigma=\Sigma^{1/2}\Sigma^{1/2}$ と分解し $\boldsymbol Z=\Sigma^{-1/2}(\boldsymbol X-\boldsymbol\mu)$ とおくと、線形変換の性質から $\boldsymbol Z\sim N_p(\boldsymbol0,I)$。よって $\boldsymbol Z^\top\boldsymbol Z=\sum Z_i^2\sim\chi^2(p)$ です。

## 具体例

**例1：3資産のポートフォリオ**

3つの資産の年間収益率（%）が $N_3(\boldsymbol\mu,\Sigma)$ に従い、

$$
\boldsymbol\mu=\begin{pmatrix}5\\3\\4\end{pmatrix},\qquad
\Sigma=\begin{pmatrix}400&60&150\\60&100&30\\150&30&225\end{pmatrix}
$$

とします（標準偏差 20, 10, 15、相関 0.3, 0.5, 0.2）。投資比率 $\boldsymbol w=(0.4,0.3,0.3)^\top$ の収益率 $R=\boldsymbol w^\top\boldsymbol X$ は正規分布で、

$$
E[R]=\boldsymbol w^\top\boldsymbol\mu=4.1,\qquad V[R]=\boldsymbol w^\top\Sigma\boldsymbol w=64+9+20.25+2(7.2+18+2.7)=149.05
$$

標準偏差は $\sqrt{149.05}\approx12.209$ で、損失が出る確率は $P(R<0)=\Phi(-4.1/12.209)\approx0.3685$ です。

**例2：マハラノビス距離による外れ値判定（試験に近い形）**

$\boldsymbol X\sim N_2(\boldsymbol0,\Sigma)$、$\Sigma=\begin{pmatrix}1&0.5\\0.5&1\end{pmatrix}$ とします。$\Sigma^{-1}=\dfrac{1}{0.75}\begin{pmatrix}1&-0.5\\-0.5&1\end{pmatrix}$ です。

- 点 $(1.5,\ 1.5)$：$D^2=3$。
- 点 $(1.5,\ -1.5)$：$D^2=9$。

どちらも原点からのユークリッド距離の2乗は 4.5 で同じですが、相関の向きに沿った前者は「よくある点」、相関に逆らう後者は「珍しい点」です。$\chi^2(2)$ の上側確率は $e^{-d/2}$ なので、$P(D^2>3)=e^{-1.5}\approx0.2231$、$P(D^2>9)=e^{-4.5}\approx0.0111$ です。上側5%点 5.991 を基準にすると、後者だけが外れ値と判定されます。

## 現場での使われ方

- **金融・保険**：多数の資産の収益率を多変量正規分布で表し、ポートフォリオのリスク（分散）を $\boldsymbol w^\top\Sigma\boldsymbol w$ で計算します。
- **製造・品質管理**：複数の品質特性を同時に監視し、マハラノビス距離（ホテリングの $T^2$）が管理限界を超えたら異常とします。
- **医療**：複数の検査値の組み合わせが健康な人の分布から外れているかを、マハラノビス距離で評価します。
- **IT・Webサービス**：異常検知や、正規分布を仮定した判別分析（第23章）・ガウス過程回帰の基礎として使います。

## グラフ・イメージ

1つ目は、多変量正規分布から2変数を取り出した散布図です。$\rho$ を動かすと楕円の傾きと細さが変わります。
2つ目は、マハラノビス距離の2乗が従う $\chi^2(p)$ の密度です。次元 $p$ を増やすと分布が右へ移り、高次元では「中心付近にほとんど点がない」ことがわかります。

## 試験のポイント

- 線形変換 $A\boldsymbol X+\boldsymbol b\sim N(A\boldsymbol\mu+\boldsymbol b,A\Sigma A^\top)$ を使って、和・差・平均の分布を求める問題が典型です。
- 条件付き分布の平均 $\boldsymbol\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(\boldsymbol x_2-\boldsymbol\mu_2)$ と分散 $\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ を書けるようにします。
- 多変量正規では「無相関 ⇔ 独立」。
- マハラノビス距離の2乗が $\chi^2(p)$ に従うことは、第23章（判別分析）や第11章（ホテリングの $T^2$）で使います。

## よくある間違い

- × 各成分が正規分布なら、ベクトル全体も多変量正規分布 → ○ すべての線形結合が正規である必要があります。
- × $V[A\boldsymbol X]=A^2\Sigma$ や $A^\top\Sigma A$ → ○ $A\Sigma A^\top$ です。行列の次元を確かめると間違いに気づけます。
- × 外れ値の判定を、変数ごとのユークリッド距離で行う → ○ 相関がある場合はマハラノビス距離を使います。
