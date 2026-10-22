---
id: hotelling-t2
title: ホテリングのT二乗検定
reading: ほてりんぐのてぃーにじょうけんてい
en: Hotelling's T-squared test
aliases: [ホテリングのT², "T²統計量", 多変量t検定, 平均ベクトルの検定]
chapter: 11
importance: B
tags: [仮説検定, 多変量正規分布, 平均ベクトル, F分布, マハラノビス距離]
related: [multivariate-normal-distribution, mahalanobis-distance, one-sample-t-test, f-distribution, mean-vector, variance-covariance-matrix]
charts:
  - type: pdf-cdf
    dist: f
    params: { df1: 2, df2: 13 }
    x: { min: 0, max: 12 }
    caption: T² を F に変換した統計量の帰無分布 F(2,13)。x を例1の観測値 8.956 に合わせると、右すその面積（p 値）は約 0.0036 とごく小さい。
---

## 一言でいうと

複数の変数をまとめて、母平均**ベクトル**が基準のベクトルと等しいかを1回で検定する方法です。
t 検定を多変量に拡張したもので、変数間の相関を考慮した「マハラノビス距離」で平均のずれを測ります。

## 定義・数式

**仮定**：$p$ 次元のデータ $\boldsymbol X_1,\dots,\boldsymbol X_n$ が互いに独立に多変量正規分布 $N_p(\boldsymbol\mu,\Sigma)$ に従う。$\Sigma$ は未知で正定値、$n>p$。

**仮説**：$H_0:\boldsymbol\mu=\boldsymbol\mu_0$、$H_1:\boldsymbol\mu\ne\boldsymbol\mu_0$。

**検定統計量**（1標本）：

$$
T^2=n\,(\bar{\boldsymbol X}-\boldsymbol\mu_0)^\top S^{-1}(\bar{\boldsymbol X}-\boldsymbol\mu_0)
$$

$\bar{\boldsymbol X}$ は標本平均ベクトル、$S=\frac{1}{n-1}\sum_{i}(\boldsymbol X_i-\bar{\boldsymbol X})(\boldsymbol X_i-\bar{\boldsymbol X})^\top$ は不偏な標本分散共分散行列です。

**帰無仮説のもとでの分布**：

$$
F=\frac{n-p}{p(n-1)}\,T^2\sim F(p,\ n-p)
$$

**棄却域**：$F\ge F_\alpha(p,n-p)$、すなわち $T^2\ge\dfrac{p(n-1)}{n-p}F_\alpha(p,n-p)$。

**2標本の場合**（共通の $\Sigma$、大きさ $n_1,n_2$、プールした分散共分散行列 $S_p$）：

$$
T^2=\frac{n_1n_2}{n_1+n_2}(\bar{\boldsymbol X}_1-\bar{\boldsymbol X}_2)^\top S_p^{-1}(\bar{\boldsymbol X}_1-\bar{\boldsymbol X}_2),\qquad
\frac{n_1+n_2-p-1}{p(n_1+n_2-2)}T^2\sim F(p,\ n_1+n_2-p-1)
$$

## 数式の読み解き

- $p=1$ のとき $S^{-1}=1/s^2$ なので $T^2=n(\bar x-\mu_0)^2/s^2=t^2$ となり、1標本 t 統計量の二乗に一致します。変換係数も $(n-1)/(n-1)=1$ で、$t^2\sim F(1,n-1)$ という t 分布と F 分布の関係に戻ります。
- $(\bar{\boldsymbol x}-\boldsymbol\mu_0)^\top S^{-1}(\bar{\boldsymbol x}-\boldsymbol\mu_0)$ は、平均のずれの**マハラノビス距離の二乗**です。ばらつきの大きい方向のずれは小さく、ばらつきの小さい方向のずれは大きく評価します。
- 相関も考慮されます。2つの変数に正の相関があるのに、平均のずれが**逆向き**に出ていると、それは「起こりにくいずれ」として大きく評価されます。
- 変数ごとに t 検定を $p$ 回繰り返すと、検定の多重性で第1種の過誤が膨らみます。$T^2$ 検定は1回で全体の水準 $\alpha$ を保ちます。

## 具体例

**例1：2つの品質特性の同時検定**

ある製品の2つの特性（寸法と重量を、それぞれ規格値からのずれとして測定）について、$n=15$ 個のデータから

$$
\bar{\boldsymbol x}-\boldsymbol\mu_0=\begin{pmatrix}1.2\\-0.8\end{pmatrix},\qquad
S=\begin{pmatrix}4.00&1.50\\1.50&2.25\end{pmatrix}
$$

が得られました。規格値どおり（$\boldsymbol\mu=\boldsymbol\mu_0$）かを有意水準5%で検定します。

- $\det S=4.00\times2.25-1.50^2=6.75$、$S^{-1}=\dfrac{1}{6.75}\begin{pmatrix}2.25&-1.50\\-1.50&4.00\end{pmatrix}$
- 二次形式：$\dfrac{2.25\times1.2^2-2\times1.50\times1.2\times(-0.8)+4.00\times0.8^2}{6.75}=\dfrac{3.24+2.88+2.56}{6.75}=1.2859$
- $T^2=15\times1.2859=19.289$
- $F=\dfrac{15-2}{2\times14}\times19.289=8.956$、自由度 $(2,13)$

$F_{0.05}(2,13)=3.806$ なので $H_0$ を棄却します。p値は $P(F_{2,13}\ge8.956)\approx0.0036$ です。

**変数ごとの t 検定との比較**：寸法だけなら $t=1.2/\sqrt{4/15}=2.324$（p値 $0.0357$）、重量だけなら $t=-0.8/\sqrt{2.25/15}=-2.066$（p値 $0.0579$）です。
重量単独では有意でないのに、$T^2$ ではずっと強く棄却されます。2特性は正の相関（相関係数 $0.5$）をもつのに、ずれが逆向きに出ているためです。

## 現場での使われ方

- **製造・品質管理**：複数の品質特性をまとめて監視する**多変量管理図（$T^2$ 管理図）**の基礎です。各特性が個別には管理限界内でも、組み合わせとして異常な状態を検出できます。
- **医療**：複数の検査項目（血圧・コレステロール・血糖など）を同時に比べ、治療群と対照群で「全体として」差があるかを2標本の $T^2$ で調べます。
- **マーケティング**：複数の評価項目（価格・品質・デザインの満足度）の平均プロファイルが、2つの顧客層で異なるかを検定します。

## グラフ・イメージ

図は、$T^2$ を F 統計量に変換したときの帰無分布 $F(2,13)$ です。x を例1の $8.956$ に合わせると、右すその面積は約 $0.0036$ です。
F 分布の右すそで判定するので、$T^2$ 検定は向きを区別しない（両側に相当する）検定です。

## 試験のポイント

- 統計量 $T^2=n(\bar{\boldsymbol x}-\boldsymbol\mu_0)^\top S^{-1}(\bar{\boldsymbol x}-\boldsymbol\mu_0)$ と、F 分布への変換係数 $(n-p)/\{p(n-1)\}$、自由度 $(p,n-p)$ が問われます。
- 2×2 行列の逆行列 $\begin{pmatrix}a&b\\c&d\end{pmatrix}^{-1}=\dfrac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$ を使った手計算に慣れておきます。
- $p=1$ で t 検定に一致すること、マハラノビス距離との関係を説明させる問題があります。
- 2標本版は判別分析（第23章）や多変量分散分析につながります。

## よくある間違い

- × $T^2$ をそのまま $\chi^2(p)$ や $F(p,n-p)$ と比べる → ○ $\Sigma$ が未知なら、係数 $(n-p)/\{p(n-1)\}$ をかけてから $F(p,n-p)$ と比べます（$\Sigma$ 既知なら $n(\bar{\boldsymbol x}-\boldsymbol\mu_0)^\top\Sigma^{-1}(\bar{\boldsymbol x}-\boldsymbol\mu_0)\sim\chi^2(p)$ です）。
- × 変数ごとの t 検定をすべて非有意なら、$T^2$ も非有意になる → ○ 相関を考慮するので、個別には有意でなくても全体として有意になることがあります（例1の重量）。
- × $S^{-1}$ の代わりに対角成分だけの逆数を使う → ○ 共分散を無視すると、相関の効果が反映されません。
- × 自由度を $(p,n-1)$ にする → ○ 分母の自由度は $n-p$ です。
