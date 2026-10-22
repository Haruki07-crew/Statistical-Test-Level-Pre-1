---
id: canonical-correlation-analysis
title: 正準相関分析
reading: せいじゅんそうかんぶんせき
en: Canonical correlation analysis
aliases: [CCA, 正準相関]
chapter: 26
importance: A
tags: [多変量解析, 2組の変数群, 固有値問題, 相関]
related: [canonical-variate, correlation-matrix, coefficient-of-determination, principal-component-analysis, eigenvalue-and-eigenvector, lagrange-multiplier-pca]
charts:
  - type: scatter
    model: bvnorm
    params: { rho: 0.57, n: 300 }
    sliders:
      - { param: rho, label: "正準相関係数 ρ", min: 0, max: 0.95, step: 0.01 }
    caption: 第1正準変量 U（横軸）と V（縦軸）の散布図のイメージ。U と V はどちらも平均0・分散1に標準化されていて、その相関が正準相関係数になる。例1では ρ≈0.57。
---

## 一言でいうと

2組の変数群（たとえば「生活習慣の変数たち」と「健康指標の変数たち」）の間の関係を、それぞれの組の合成変数どうしの相関が最大になるように要約する方法です。
2つの群を結ぶ「いちばん強いつながり」を1本ずつ取り出していきます。

## 定義・数式

$p$ 次元の $\boldsymbol{x}$ と $q$ 次元の $\boldsymbol{y}$ について、分散共分散行列を

$$
\mathrm{Var}\begin{pmatrix}\boldsymbol{x}\\\boldsymbol{y}\end{pmatrix}=\begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\\Sigma_{21}&\Sigma_{22}\end{pmatrix},\qquad \Sigma_{21}=\Sigma_{12}^\top
$$

とします（$\Sigma_{11},\Sigma_{22}$ は正則）。線形結合 $U=\boldsymbol{a}^\top\boldsymbol{x}$, $V=\boldsymbol{b}^\top\boldsymbol{y}$ の相関

$$
\mathrm{Corr}(U,V)=\frac{\boldsymbol{a}^\top\Sigma_{12}\boldsymbol{b}}{\sqrt{\boldsymbol{a}^\top\Sigma_{11}\boldsymbol{a}}\sqrt{\boldsymbol{b}^\top\Sigma_{22}\boldsymbol{b}}}
$$

を最大にする $(\boldsymbol{a},\boldsymbol{b})$ を求めます。最大値 $\rho_1$ が**第1正準相関係数**、$U_1,V_1$ が**第1正準変量**です。

解は次の固有値問題で与えられます。

$$
\Sigma_{11}^{-1}\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\,\boldsymbol{a}=\rho^2\boldsymbol{a},\qquad
\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}\,\boldsymbol{b}=\rho^2\boldsymbol{b}
$$

固有値を大きい順に $\rho_1^2\ge\rho_2^2\ge\cdots$ と並べると、正準相関係数は $\rho_1\ge\rho_2\ge\dots\ge\rho_{\min(p,q)}\ge0$ の $\min(p,q)$ 個です。第2組以降は、それより前の正準変量と無相関という条件の下で相関を最大化したものです。

## 数式の読み解き

相関は $\boldsymbol{a},\boldsymbol{b}$ を定数倍しても変わらないので、$\mathrm{Var}(U)=\boldsymbol{a}^\top\Sigma_{11}\boldsymbol{a}=1$, $\mathrm{Var}(V)=\boldsymbol{b}^\top\Sigma_{22}\boldsymbol{b}=1$ と固定し、共分散 $\boldsymbol{a}^\top\Sigma_{12}\boldsymbol{b}$ を最大化します。ラグランジュ関数は

$$
L=\boldsymbol{a}^\top\Sigma_{12}\boldsymbol{b}-\frac{\lambda}{2}(\boldsymbol{a}^\top\Sigma_{11}\boldsymbol{a}-1)-\frac{\mu}{2}(\boldsymbol{b}^\top\Sigma_{22}\boldsymbol{b}-1)
$$

$\boldsymbol{a},\boldsymbol{b}$ で微分して $\boldsymbol{0}$ とおくと

$$
\Sigma_{12}\boldsymbol{b}=\lambda\Sigma_{11}\boldsymbol{a},\qquad \Sigma_{21}\boldsymbol{a}=\mu\Sigma_{22}\boldsymbol{b}
$$

左から $\boldsymbol{a}^\top$, $\boldsymbol{b}^\top$ を掛けると、制約から $\lambda=\mu=\boldsymbol{a}^\top\Sigma_{12}\boldsymbol{b}=\rho$ がわかります。2本目の式から $\boldsymbol{b}=\rho^{-1}\Sigma_{22}^{-1}\Sigma_{21}\boldsymbol{a}$ として1本目に代入すると、

$$
\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\boldsymbol{a}=\rho^2\Sigma_{11}\boldsymbol{a}
$$

両辺に $\Sigma_{11}^{-1}$ を掛ければ上の固有値問題になります。最大の相関がほしいので、最大固有値を選びます。

- 変数を標準化すれば、$\Sigma$ を相関行列 $R$ に置き換えて同じ計算ができます。正準相関係数は変数の単位によりません。
- $q=1$（$\boldsymbol{y}$ が1変数）のとき、$\rho_1^2=\boldsymbol{r}^\top R_{11}^{-1}\boldsymbol{r}$ は $y$ を $\boldsymbol{x}$ で回帰したときの決定係数です。つまり正準相関は**重相関係数の一般化**です。さらに $p=q=1$ なら $\rho_1=|r_{xy}|$ です。
- 主成分分析が1組の変数の「分散」を最大化するのに対し、正準相関分析は2組の間の「相関」を最大化します。

## 具体例

**例1：生活習慣と健康指標（$n=100$）**

$\boldsymbol{x}=$（運動時間, 睡眠時間）、$\boldsymbol{y}=$（体力スコア, 集中力スコア）の相関行列が

$$
R_{11}=\begin{pmatrix}1&0.4\\0.4&1\end{pmatrix},\quad
R_{22}=\begin{pmatrix}1&0.3\\0.3&1\end{pmatrix},\quad
R_{12}=\begin{pmatrix}0.5&0.3\\0.4&0.2\end{pmatrix}
$$

だったとします。逆行列は $R_{11}^{-1}=\frac{1}{0.84}\begin{pmatrix}1&-0.4\\-0.4&1\end{pmatrix}$, $R_{22}^{-1}=\frac{1}{0.91}\begin{pmatrix}1&-0.3\\-0.3&1\end{pmatrix}$ です。順に掛けると

$$
R_{12}R_{22}^{-1}R_{21}=\begin{pmatrix}0.2747&0.2132\\0.2132&0.1670\end{pmatrix},\qquad
M=R_{11}^{-1}R_{12}R_{22}^{-1}R_{21}=\begin{pmatrix}0.2255&0.1743\\0.1230&0.0973\end{pmatrix}
$$

$M$ の固有値は特性方程式 $\lambda^2-0.3229\lambda+0.000523=0$ の解で、

$$
\rho_1^2=0.3212,\quad \rho_2^2=0.0016\qquad\Longrightarrow\qquad \rho_1=0.567,\quad \rho_2=0.040
$$

です。分散1に基準化した係数は $\boldsymbol{a}_1=(0.758,\ 0.416)$, $\boldsymbol{b}_1=(0.877,\ 0.285)$ で、第1正準変量は「運動中心の生活習慣」と「体力中心の健康状態」の結びつきを表します。

**例2：$\boldsymbol{y}$ が1変数のとき**

例1の体力スコア $y_1$ だけを $\boldsymbol{y}$ とします。$\boldsymbol{r}=(0.5,0.4)^\top$ なので

$$
\rho_1^2=\boldsymbol{r}^\top R_{11}^{-1}\boldsymbol{r}=\frac{0.5^2+0.4^2-2\times0.4\times0.5\times0.4}{1-0.4^2}=\frac{0.25}{0.84}=0.2976
$$

$\rho_1=0.546$ で、これは $y_1$ を運動時間と睡眠時間で重回帰したときの重相関係数と一致します。

## 現場での使われ方

- **医療・健康**：生活習慣の質問項目群と、血液検査値の群との関係を要約し、どの習慣の組み合わせがどの検査値の組み合わせと結びつくかを調べます。
- **マーケティング**：顧客の属性・行動の変数群と、商品評価の変数群の対応を調べ、評価を高める顧客像を見つけます。
- **教育**：入試の科目群の得点と、入学後の成績群の関係を測り、入試が入学後の力をどれだけ予測しているかを評価します。
- **IT・Webサービス**：画像の特徴量とテキストの特徴量のように、2種類のデータの共通の軸を見つける手法の土台になっています。

## グラフ・イメージ

第1正準変量の組 $(U_1,V_1)$ を散布図にしたイメージです。どちらも分散1に基準化されているので、点の雲の細長さがそのまま正準相関係数の大きさを表します。
スライダーで $\rho$ を大きくすると、2群の間に強い共通の軸があるときの姿になります。

## 試験のポイント

- ラグランジュの未定乗数法から固有値問題 $\Sigma_{11}^{-1}\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\boldsymbol{a}=\rho^2\boldsymbol{a}$ を導く流れが重要です。
- 固有値は $\rho$ ではなく $\rho^2$ です。平方根を取って正準相関係数にします。
- 正準相関の個数は $\min(p,q)$ です。$2\times2$ の場合は特性方程式（トレースと行列式）で手計算できます。
- $q=1$ なら重相関係数、$p=q=1$ なら相関係数の絶対値になる、という特殊な場合の確認がよく使われます。
- 第1正準相関係数は、$\boldsymbol{x}$ 側のどの変数と $\boldsymbol{y}$ 側のどの変数の相関の絶対値よりも小さくなりません。

## よくある間違い

- × 固有値をそのまま正準相関係数として答える → ○ 固有値は $\rho^2$ です。例1なら $0.3212$ ではなく $\rho_1=0.567$ です。
- × $\Sigma_{11}^{-1}\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ を対称行列として扱い、固有ベクトルが直交すると考える → ○ この行列は一般に非対称です。直交するのは $\boldsymbol{a}_k$ そのものではなく、正準変量 $U_k$ どうし（無相関）です。
- × 正準相関が大きいので、$\boldsymbol{x}$ の各変数が $\boldsymbol{y}$ をよく説明すると考える → ○ 正準相関は合成変数どうしの相関です。個々の変数がどれだけ説明されるかは、正準負荷量や冗長性係数で別に確認します。
