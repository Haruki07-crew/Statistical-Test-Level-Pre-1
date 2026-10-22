---
id: yule-walker-equations
title: ユール・ウォーカー方程式
reading: ゆーる・うぉーかーほうていしき
en: Yule-Walker equations
aliases: [ユール・ウォーカー推定, "Yule-Walker estimator", ユールウォーカー方程式]
chapter: 27
importance: A
tags: [時系列解析, ARモデル, 推定, 自己相関]
related: [ar-model, partial-autocorrelation, autocovariance-autocorrelation, method-of-moments, stationarity-condition]
charts:
  - type: custom
    id: acf-plot
    params: { model: ar2, phi: 0.5, phi2: 0.3, n: 300, lags: 15, pacf: 1 }
    sliders:
      - { param: phi, label: "φ1", min: -1, max: 1.5, step: 0.05 }
      - { param: phi2, label: "φ2", min: -0.9, max: 0.9, step: 0.05 }
    caption: AR(2) の自己相関の理論値（点）は、ユール・ウォーカー方程式 ρ(h)=φ1ρ(h-1)+φ2ρ(h-2) の漸化式で決まる。標本自己相関（棒）を使って逆に φ1, φ2 を推定するのがユール・ウォーカー推定。
---

## 一言でいうと

AR モデルの係数と自己相関を結びつける連立1次方程式です。
係数から自己相関を求めることにも、標本自己相関から係数を推定することにも使えます。

## 定義・数式

定常な AR($p$) $X_t=\phi_1X_{t-1}+\cdots+\phi_pX_{t-p}+\varepsilon_t$（平均0、$\varepsilon_t\sim WN(0,\sigma^2)$）について、

$$
\rho(h)=\phi_1\rho(h-1)+\phi_2\rho(h-2)+\cdots+\phi_p\rho(h-p),\qquad h=1,2,\dots
$$

が成り立ちます。$h=1,\dots,p$ を並べた行列の形

$$
\begin{pmatrix}
1 & \rho(1) & \cdots & \rho(p-1)\\
\rho(1) & 1 & \cdots & \rho(p-2)\\
\vdots & & \ddots & \vdots\\
\rho(p-1) & \rho(p-2) & \cdots & 1
\end{pmatrix}
\begin{pmatrix}\phi_1\\ \vdots\\ \phi_p\end{pmatrix}
=
\begin{pmatrix}\rho(1)\\ \vdots\\ \rho(p)\end{pmatrix}
$$

を**ユール・ウォーカー方程式**といいます。雑音の分散は

$$
\sigma^2=\gamma(0)\{1-\phi_1\rho(1)-\cdots-\phi_p\rho(p)\}
$$

です。

**ユール・ウォーカー推定**：$\rho(h)$ を標本自己相関 $r_h$、$\gamma(0)$ を標本分散 $\hat\gamma(0)$ に置き換えて解いた $\hat\phi_1,\dots,\hat\phi_p,\ \hat\sigma^2$ です。

AR(1)・AR(2) では明示的に解けます。

$$
\text{AR(1)：}\ \hat\phi=r_1,\qquad
\text{AR(2)：}\ \hat\phi_1=\frac{r_1(1-r_2)}{1-r_1^2},\quad \hat\phi_2=\frac{r_2-r_1^2}{1-r_1^2}
$$

## 数式の読み解き

**導出**：モデルの両辺に $X_{t-h}$（$h\ge1$）を掛けて期待値をとります。

$$
E[X_tX_{t-h}]=\phi_1E[X_{t-1}X_{t-h}]+\cdots+\phi_pE[X_{t-p}X_{t-h}]+E[\varepsilon_tX_{t-h}]
$$

$\varepsilon_t$ は過去の値 $X_{t-h}$ と無相関なので最後の項は0で、$\gamma(h)=\phi_1\gamma(h-1)+\cdots+\phi_p\gamma(h-p)$。$\gamma(0)$ で割れば上の式です。
$h=0$（$X_t$ を掛ける）では $E[\varepsilon_tX_t]=\sigma^2$ が残り、$\gamma(0)=\sum\phi_j\gamma(j)+\sigma^2$ から $\sigma^2$ の式が出ます。

- 左辺の行列は自己相関行列で、対角線に平行な線上で値が同じ（テプリッツ行列）です。
- 推定は「理論モーメント＝標本モーメント」とおくので、**モーメント法**の一種です。
- 標本自己共分散を $n$ で割って作ると行列が正定値になり、推定されたモデルは必ず定常になります。
- 方程式の最後の係数 $\phi_{pp}$ を次数 $p=1,2,\dots$ と変えて並べたものが偏自己相関です。ダービン・レビンソン法はこの方程式を次数の低い順に解く方法です。

**推定量の性質**：$n$ が大きいとき $\sqrt n(\hat{\boldsymbol\phi}-\boldsymbol\phi)$ は近似的に $N(\boldsymbol 0,\ \sigma^2\Gamma_p^{-1})$（$\Gamma_p$ は自己共分散行列）に従います。AR(1) では

$$
V[\hat\phi]\approx\frac{1-\phi^2}{n}
$$

で、正規分布の雑音なら最尤推定量と同じ漸近効率を持ちます。

**ほかの推定法との比較**

- **最小二乗法**：$X_t$ を $X_{t-1},\dots,X_{t-p}$ に回帰します。ユール・ウォーカー推定との違いは端の数項の扱いだけで、$n$ が大きければほぼ同じ値になります。
- **最尤法**：正規分布を仮定して尤度を最大化します。MA 項を含むモデルではこちらが標準です。
- 係数が単位根に近い（$\phi$ が1に近い）ときや $n$ が小さいとき、ユール・ウォーカー推定は0の方向に偏りやすいことが知られています。定常性が必ず保たれるのは利点ですが、精度が必要なら最尤法を使います。

## 具体例

**例1：AR(2) の係数を推定する**

ある系列（$n=200$）で $\hat\gamma(0)=2$、$r_1=0.6$、$r_2=0.45$ でした。AR(2) を当てはめると、

$$
\hat\phi_1=\frac{0.6(1-0.45)}{1-0.36}=\frac{0.33}{0.64}=0.515625,\qquad
\hat\phi_2=\frac{0.45-0.36}{0.64}=\frac{0.09}{0.64}=0.140625
$$

$$
\hat\sigma^2=2\{1-0.515625\times0.6-0.140625\times0.45\}=2\times0.627344\approx1.2547
$$

AR(1) で当てはめると $\hat\phi=0.6$、$\hat\sigma^2=2(1-0.36)=1.28$ です。AR(2) にすると雑音の分散が少し小さくなります（$\hat\phi_2=\hat\phi_{22}=0.1406$ が偏自己相関）。

**例2：信頼区間**

$n=100$ で $r_1=0.6$ から AR(1) の $\hat\phi=0.6$ を得たとき、標準誤差は $\sqrt{(1-0.36)/100}=0.08$ なので、近似95%信頼区間は

$$
0.6\pm1.96\times0.08=[0.4432,\ 0.7568]
$$

です。

**例3：係数から自己相関を求める（逆向きの使い方）**

$\phi_1=0.5,\ \phi_2=0.3$ の AR(2) では、$h=1$ の式 $\rho(1)=0.5+0.3\rho(1)$ から $\rho(1)=0.5/0.7\approx0.7143$、$h=2$ の式から $\rho(2)=0.5\times0.7143+0.3\approx0.6571$ が得られます。

## 現場での使われ方

- **信号処理**：音声の線形予測係数を、短い区間ごとにユール・ウォーカー方程式（ダービン・レビンソン法）で高速に求めます。
- **経済・需要予測**：AR モデルの初期推定値をユール・ウォーカー法で求め、それを出発点に最尤法で精密化します。
- **製造・品質管理**：工程データの AR(1) 係数を推定し、自己相関を考慮した管理限界を設定します。

## グラフ・イメージ

点で示した理論自己相関は、ユール・ウォーカー方程式の漸化式で計算した値です。
φ1, φ2 を動かすと理論値と標本値（棒）がそろって変わり、標本自己相関から係数を逆算できる理由がわかります。

## 試験のポイント

- 導出の要点「$X_{t-h}$ を掛けて期待値、$E[\varepsilon_tX_{t-h}]=0$」は必ず書けるようにします。
- AR(2) の明示解と、$\sigma^2=\gamma(0)(1-\sum\phi_j\rho(j))$ を使った雑音分散の推定は計算問題で重要です。
- AR(1) の $\hat\phi=r_1$、漸近分散 $(1-\phi^2)/n$ を覚えておきます。
- 偏自己相関（最後の係数）との関係も合わせて問われます。

## よくある間違い

- × $\sigma^2$ を $\hat\gamma(0)$ のまま使う → ○ 自己回帰で説明された分を引いて $\gamma(0)(1-\sum\phi_j\rho(j))$ にします。
- × $h=0$ の式でも $E[\varepsilon_tX_t]=0$ とする → ○ $X_t$ 自身が $\varepsilon_t$ を含むので $\sigma^2$ になります。
- × ユール・ウォーカー方程式を MA モデルの推定にもそのまま使う → ○ MA の自己相関は係数について非線形で、この線形方程式は成り立ちません。
