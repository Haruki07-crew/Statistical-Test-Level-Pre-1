---
id: logistic-likelihood-estimation
title: ロジスティック回帰の最尤推定
reading: ろじすてぃっくかいきのさいゆうすいてい
en: Maximum likelihood estimation for logistic regression
aliases: [ロジスティック回帰の尤度方程式, ニュートン・ラフソン法, ニュートン法, スコア方程式, 完全分離]
chapter: 18
importance: B
tags: [質的回帰, 最尤法, 数値計算, フィッシャー情報量]
related: [logistic-regression, maximum-likelihood-estimator, score-function, fisher-information, iteratively-reweighted-least-squares, wald-test]
charts:
  - type: custom
    id: line-data
    params: { x: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], series: [{ label: "プロファイル対数尤度（β0 は各 β1 で最大化）", y: [-57.833, -56.154, -54.882, -53.984, -53.428, -53.179, -53.207, -53.48, -53.971, -54.654, -55.508, -56.511] }], xlabel: "傾き β1", ylabel: "対数尤度（定数項を除く）" }
    caption: 例1のデータで、β1 を固定して β0 について最大化した対数尤度。山の頂上が最尤推定値 β1=0.939（対数尤度 −53.159）。山が鋭いほど標準誤差は小さい。
---

## 一言でいうと

ロジスティック回帰の係数は、二項分布の尤度を最大にする値（最尤推定量）として求めます。
尤度方程式は解析的には解けないので、ニュートン・ラフソン法で反復して解きます。

## 定義・数式

$Y_i\sim\text{Bin}(m_i,p_i)$（個別データなら $m_i=1$）が独立で、$\operatorname{logit}p_i=\eta_i=\boldsymbol x_i^\top\boldsymbol\beta$ とします。

**対数尤度**（$\boldsymbol\beta$ を含まない二項係数を除く）

$$
\ell(\boldsymbol\beta)=\sum_{i=1}^n\{y_i\log p_i+(m_i-y_i)\log(1-p_i)\}
=\sum_{i=1}^n\Bigl\{y_i\eta_i-m_i\log\bigl(1+e^{\eta_i}\bigr)\Bigr\}
$$

**スコア関数（尤度方程式）**

$$
\boldsymbol U(\boldsymbol\beta)=\frac{\partial\ell}{\partial\boldsymbol\beta}=\sum_{i=1}^n(y_i-m_ip_i)\boldsymbol x_i=X^\top(\boldsymbol y-\boldsymbol\mu)=\boldsymbol0
$$

**ヘッセ行列とフィッシャー情報行列**

$$
\frac{\partial^2\ell}{\partial\boldsymbol\beta\,\partial\boldsymbol\beta^\top}=-\sum_{i=1}^nm_ip_i(1-p_i)\boldsymbol x_i\boldsymbol x_i^\top=-X^\top WX,\qquad
W=\operatorname{diag}\bigl(m_ip_i(1-p_i)\bigr)
$$

**ニュートン・ラフソン法の更新式**

$$
\boldsymbol\beta^{(t+1)}=\boldsymbol\beta^{(t)}+\bigl(X^\top W^{(t)}X\bigr)^{-1}X^\top\bigl(\boldsymbol y-\boldsymbol\mu^{(t)}\bigr)
$$

収束値 $\hat{\boldsymbol\beta}$ の漸近的な分散共分散行列は $(X^\top\hat WX)^{-1}$ で、その対角成分の平方根が標準誤差です。

## 数式の読み解き

**対数尤度の書き換え**

$\log p_i=\eta_i-\log(1+e^{\eta_i})$、$\log(1-p_i)=-\log(1+e^{\eta_i})$ を代入すると、

$$
y_i\log p_i+(m_i-y_i)\log(1-p_i)=y_i\eta_i-m_i\log(1+e^{\eta_i})
$$

となります。

**スコアの導出**

$\dfrac{d}{d\eta}\log(1+e^\eta)=\dfrac{e^\eta}{1+e^\eta}=p$ なので、$\partial\ell/\partial\eta_i=y_i-m_ip_i$。$\partial\eta_i/\partial\boldsymbol\beta=\boldsymbol x_i$ を掛けて足せばスコアです。
「観測値 − 期待値」に説明変数を掛けて足したものが0、という形は線形回帰の正規方程式 $X^\top(\boldsymbol y-X\boldsymbol\beta)=\boldsymbol0$ とそっくりです（ロジットが正準リンクであるため）。

**ヘッセ行列の導出**

$dp/d\eta=p(1-p)$ なので、$\partial(y_i-m_ip_i)/\partial\boldsymbol\beta^\top=-m_ip_i(1-p_i)\boldsymbol x_i^\top$ です。
$y$ を含まないので、観測情報量とフィッシャー情報量は同じ $X^\top WX$ になり、ニュートン・ラフソン法とフィッシャーのスコア法は同じ反復になります。

**凹性と解の存在**

$X$ が列フルランクで $0<p_i<1$ なら $X^\top WX$ は正定値なので、対数尤度は狭義の凹関数です。最大値があれば1つに決まります。
ただし、ある一次式で $y=1$ と $y=0$ が完全に分けられる（**完全分離**）と、係数を大きくするほど尤度が1に近づき、最尤推定値は存在しません（推定値と標準誤差が発散します）。

## 具体例

**例1：殺虫剤のデータでニュートン法を1歩ずつ進める**

濃度 $x=0,1,2,3,4$、各 $m_i=20$ 匹、死亡数 $y=3,4,10,12,18$ です。初期値を $\boldsymbol\beta^{(0)}=(0,0)$ とします。

① 全グループで $p_i=0.5$、$m_ip_i=10$、$w_i=20\times0.25=5$。

② スコア：$\sum(y_i-10)=47-50=-3$、$\sum x_i(y_i-10)=132-100=32$ より $\boldsymbol U=(-3,\ 32)^\top$。

③ 情報行列：$\sum x_i=10$、$\sum x_i^2=30$ より

$$
X^\top WX=5\begin{pmatrix}5&10\\10&30\end{pmatrix}=\begin{pmatrix}25&50\\50&150\end{pmatrix},\qquad
(X^\top WX)^{-1}=\frac{1}{1250}\begin{pmatrix}150&-50\\-50&25\end{pmatrix}
$$

④ 更新：

$$
\boldsymbol\beta^{(1)}=\frac{1}{1250}\begin{pmatrix}150\times(-3)-50\times32\\-50\times(-3)+25\times32\end{pmatrix}=\begin{pmatrix}-2050/1250\\950/1250\end{pmatrix}=\begin{pmatrix}-1.64\\0.76\end{pmatrix}
$$

これを繰り返すと $(-2.0088,\ 0.9224)\to(-2.0475,\ 0.9389)\to(-2.0479,\ 0.9391)$ と、4回ほどで小数第4位まで収束します。

⑤ 収束値 $\hat\beta_0=-2.0479$、$\hat\beta_1=0.9391$ で、$\hat w_i=m_i\hat p_i(1-\hat p_i)=(2.024,\ 3.731,\ 4.964,\ 4.328,\ 2.597)$、

$$
X^\top\hat WX=\begin{pmatrix}17.644&37.030\\37.030&104.088\end{pmatrix},\qquad
(X^\top\hat WX)^{-1}=\begin{pmatrix}0.22371&-0.07959\\-0.07959&0.03792\end{pmatrix}
$$

より $\mathrm{SE}(\hat\beta_0)=\sqrt{0.22371}=0.4730$、$\mathrm{SE}(\hat\beta_1)=\sqrt{0.03792}=0.1947$。ワルド統計量は $z=0.9391/0.1947\approx4.82$ です。

**例2：説明変数が0/1のときは手で解ける**

群 $x=0$ で $m_0$ 人中 $y_0$ 人、群 $x=1$ で $m_1$ 人中 $y_1$ 人が「1」とします。スコア方程式は

$$
(y_0-m_0p_0)+(y_1-m_1p_1)=0,\qquad y_1-m_1p_1=0
$$

なので $\hat p_1=y_1/m_1$、$\hat p_0=y_0/m_0$ です。よって $\hat\beta_0=\operatorname{logit}(y_0/m_0)$、$\hat\beta_1=\operatorname{logit}(y_1/m_1)-\operatorname{logit}(y_0/m_0)$（標本対数オッズ比）になります。
50人中18人と50人中30人なら、$\hat\beta_0=\log(18/32)=-0.5754$、$\hat\beta_1=\log(30/20)-\log(18/32)=0.9808$ です。

## 現場での使われ方

- **医療**：臨床研究のロジスティック回帰はすべてこの最尤推定で計算され、報告されるオッズ比の信頼区間は $(X^\top\hat WX)^{-1}$ から作られます。
- **金融・保険**：信用スコアのモデルを作るとき、変数が多いと完全分離や収束しない問題が起きます。原因の変数を見つけたり、罰則付き推定に切り替えたりします。
- **IT・Webサービス**：大量データのクリック予測では、ニュートン法の代わりに確率的勾配法でスコア（勾配）$\sum(y_i-p_i)\boldsymbol x_i$ を少しずつ使って更新します。

## グラフ・イメージ

例1のデータで、$\beta_1$ の値ごとに最大化した対数尤度を描いた図です。山の頂上が最尤推定値 $\hat\beta_1=0.939$ です。
頂上付近の曲がり具合（2階微分）が情報量で、山が鋭いほど標準誤差が小さくなります。頂上から 1.92（$=\chi^2_{1,0.95}/2$）だけ下がった幅が、尤度に基づく95%信頼区間の目安です。

## 試験のポイント

- 対数尤度を $\sum\{y_i\eta_i-m_i\log(1+e^{\eta_i})\}$ に書き換え、スコア $X^\top(\boldsymbol y-\boldsymbol\mu)$ とヘッセ行列 $-X^\top WX$ を導けるようにします。
- 標準誤差は $(X^\top\hat WX)^{-1}$ の対角成分の平方根、ワルド検定は $\hat\beta_j/\mathrm{SE}$ です。
- 説明変数がダミー1つなら、MLE は群ごとの観測割合を再現し、係数は対数オッズ比になります。
- 完全分離のとき最尤推定値が存在しないことは重要です。

## よくある間違い

- × スコア方程式を最小二乗法と同じように1回の行列計算で解く → ○ $\boldsymbol\mu$ が $\boldsymbol\beta$ に非線形に依存するので反復計算が必要です。
- × 重み $w_i$ を $p_i$ とする → ○ ベルヌーイ分布の分散に相当する $m_ip_i(1-p_i)$ です。
- × 推定がうまく収束しないのはソフトの問題 → ○ 完全分離や準完全分離、強い多重共線性など、データに原因があることが多いです。
- × 標準誤差を初期値での情報行列から計算する → ○ 収束した $\hat{\boldsymbol\beta}$ で $W$ を計算し直します。
