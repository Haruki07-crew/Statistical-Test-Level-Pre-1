---
id: lagrange-multiplier-pca
title: ラグランジュ乗数法による導出
reading: らぐらんじゅじょうすうほうによるどうしゅつ
en: Derivation of PCA by Lagrange multipliers
aliases: [ラグランジュの未定乗数法, 主成分の導出, 分散最大化]
chapter: 22
importance: B
tags: [主成分分析, 最適化, 固有値問題, 導出]
related: [eigenvalue-and-eigenvector, principal-component, principal-component-analysis, variance-covariance-matrix, linear-discriminant-analysis]
charts:
  - type: custom
    id: line-data
    params:
      x: [0, 7.5, 15, 22.5, 30, 37.5, 45, 52.5, 60, 67.5, 75, 82.5, 90, 97.5, 105, 112.5, 120, 127.5, 135, 142.5, 150, 157.5, 165, 172.5, 180]
      xlabel: 方向の角度 θ（度、x1 軸から）
      ylabel: 射影した分散 a'Sa
      series:
        - { label: "S = [[5,2],[2,2]] の射影分散", y: [5, 5.4665, 5.799, 5.9749, 5.9821, 5.8201, 5.5, 5.0436, 4.4821, 3.8536, 3.201, 2.5687, 2, 1.5335, 1.201, 1.0251, 1.0179, 1.1799, 1.5, 1.9564, 2.5179, 3.1464, 3.799, 4.4313, 5] }
    caption: 長さ1の方向 a=(cosθ, sinθ) に射影した分散。最大値 6（θ≈26.57度）と最小値 1（θ≈116.57度）が、それぞれ第1・第2固有値に一致する。
---

## 一言でいうと

「長さ1の係数ベクトルという制約のもとで、合成変数の分散を最大にする」という条件付き最大化問題を、ラグランジュ乗数法で解くと固有値問題になる、という主成分分析の導出です。

## 定義・数式

**第1主成分**：分散共分散行列 $\Sigma$ のもとで、次の問題を解きます。

$$
\max_{\boldsymbol{a}}\ \boldsymbol{a}^\top\Sigma\boldsymbol{a}\quad\text{subject to}\quad\boldsymbol{a}^\top\boldsymbol{a}=1
$$

ラグランジュ関数を

$$
L(\boldsymbol{a},\lambda)=\boldsymbol{a}^\top\Sigma\boldsymbol{a}-\lambda(\boldsymbol{a}^\top\boldsymbol{a}-1)
$$

とおき、$\boldsymbol{a}$ で偏微分して $\boldsymbol{0}$ とおきます。ベクトルの微分の公式 $\partial(\boldsymbol{a}^\top A\boldsymbol{a})/\partial\boldsymbol{a}=2A\boldsymbol{a}$（$A$ は対称）を使うと

$$
\frac{\partial L}{\partial\boldsymbol{a}}=2\Sigma\boldsymbol{a}-2\lambda\boldsymbol{a}=\boldsymbol{0}\quad\Longrightarrow\quad\Sigma\boldsymbol{a}=\lambda\boldsymbol{a}
$$

となり、$\boldsymbol{a}$ は $\Sigma$ の固有ベクトル、$\lambda$ は固有値です。このときの目的関数の値は

$$
\boldsymbol{a}^\top\Sigma\boldsymbol{a}=\boldsymbol{a}^\top(\lambda\boldsymbol{a})=\lambda\,\boldsymbol{a}^\top\boldsymbol{a}=\lambda
$$

なので、最大値は最大固有値 $\lambda_1$ で、そのときの $\boldsymbol{a}$ が $\boldsymbol{h}_1$ です。

**第2主成分**：さらに第1主成分と無相関という条件 $\mathrm{Cov}[\boldsymbol{a}^\top\boldsymbol{X},\boldsymbol{h}_1^\top\boldsymbol{X}]=\boldsymbol{a}^\top\Sigma\boldsymbol{h}_1=\lambda_1\boldsymbol{a}^\top\boldsymbol{h}_1=0$、つまり $\boldsymbol{a}^\top\boldsymbol{h}_1=0$ を加えます。

$$
L=\boldsymbol{a}^\top\Sigma\boldsymbol{a}-\lambda(\boldsymbol{a}^\top\boldsymbol{a}-1)-\mu\,\boldsymbol{a}^\top\boldsymbol{h}_1
$$

$$
\frac{\partial L}{\partial\boldsymbol{a}}=2\Sigma\boldsymbol{a}-2\lambda\boldsymbol{a}-\mu\boldsymbol{h}_1=\boldsymbol{0}
$$

左から $\boldsymbol{h}_1^\top$ を掛けると、$\boldsymbol{h}_1^\top\Sigma\boldsymbol{a}=\lambda_1\boldsymbol{h}_1^\top\boldsymbol{a}=0$、$\boldsymbol{h}_1^\top\boldsymbol{a}=0$、$\boldsymbol{h}_1^\top\boldsymbol{h}_1=1$ より $\mu=0$ です。
よって再び $\Sigma\boldsymbol{a}=\lambda\boldsymbol{a}$ となり、$\boldsymbol{h}_1$ と直交する固有ベクトルのうち最大の固有値 $\lambda_2$ のものが第2主成分です。以下同様に続きます。

## 数式の読み解き

- 制約 $\boldsymbol{a}^\top\boldsymbol{a}=1$ がないと、$\boldsymbol{a}$ を $c$ 倍すれば分散が $c^2$ 倍になり、最大値が存在しません。
- 乗数 $\lambda$ は、最適解ではそのまま「最大化された分散」になります。乗数に意味がある典型例です。
- 停留点（微分が0の点）はすべての固有ベクトルで、最大を与えるのは最大固有値のものだけです。他の固有ベクトルは、制約付きの鞍点や最小点にあたります。

## 具体例

**例1：2×2 で連立方程式として解く**

$\Sigma=\begin{pmatrix}5&2\\2&2\end{pmatrix}$ のとき、$\Sigma\boldsymbol{a}=\lambda\boldsymbol{a}$ は

$$
\begin{cases}5a_1+2a_2=\lambda a_1\\2a_1+2a_2=\lambda a_2\end{cases}
$$

です。$\boldsymbol{a}\ne\boldsymbol{0}$ の解を持つ条件 $(5-\lambda)(2-\lambda)-4=0$ から $\lambda=6,\ 1$。$\lambda=6$ では $a_1=2a_2$、制約 $a_1^2+a_2^2=1$ から $\boldsymbol{a}=(2,1)^\top/\sqrt5$ で、最大分散は6です。

**例2：角度で確かめる**

長さ1の方向を $\boldsymbol{a}=(\cos\theta,\sin\theta)^\top$ とおくと、制約は自動的に満たされ、

$$
\boldsymbol{a}^\top\Sigma\boldsymbol{a}=5\cos^2\theta+4\sin\theta\cos\theta+2\sin^2\theta=3.5+1.5\cos2\theta+2\sin2\theta
$$

です。$1.5\cos2\theta+2\sin2\theta$ の振幅は $\sqrt{1.5^2+2^2}=2.5$ なので、最大値は $3.5+2.5=6$、最小値は $3.5-2.5=1$ で、2つの固有値に一致します。
最大になるのは $\tan2\theta=2/1.5$、すなわち $\theta\approx26.57$ 度で、$\tan\theta=1/2$ は固有ベクトル $(2,1)$ の向きです。

## 現場での使われ方

- **IT・Webサービス**：同じ「制約付きで2次形式を最大化 → 固有値問題」という形は、データの次元削減や特徴抽出の多くの手法に共通する計算の骨組みとして使われています。
- **マーケティング**：判別分析（群の差を最も大きくする方向）や正準相関分析も、同じラグランジュ乗数法で一般化固有値問題に帰着させて計算します。

## グラフ・イメージ

横軸は方向の角度、縦軸はその方向に射影したときの分散です。
山の頂上（約26.57度、分散6）が第1主成分、谷の底（約116.57度、分散1）が第2主成分で、2つの方向はちょうど90度離れています。

## 試験のポイント

- ラグランジュ関数の設定 → 微分 → $\Sigma\boldsymbol{a}=\lambda\boldsymbol{a}$ → 目的関数の値が $\lambda$、という流れを記述できるようにします。
- 第2主成分の導出で $\mu=0$ となる理由（$\boldsymbol{h}_1$ を左から掛ける）が重要です。
- 2次形式の微分 $\partial(\boldsymbol{a}^\top A\boldsymbol{a})/\partial\boldsymbol{a}=2A\boldsymbol{a}$ は、$A$ が対称のときの公式です。

## よくある間違い

- × 微分して得た $\Sigma\boldsymbol{a}=\lambda\boldsymbol{a}$ の解はすべて最大点 → ○ 停留点にすぎません。分散が $\lambda$ になるので、最大は最大固有値のときです。
- × 制約を $\boldsymbol{a}^\top\Sigma\boldsymbol{a}=1$ と書く → ○ 制約は係数の長さ $\boldsymbol{a}^\top\boldsymbol{a}=1$ です。分散を1に固定すると最大化する対象がなくなります。
