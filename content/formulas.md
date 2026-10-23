準1級でよく使う公式を分野ごとにまとめた早見表です。くわしい導出や例は「→」の後に書いた用語のページを見てください。
記号の約束：$z_\alpha$、$t_\alpha(k)$、$\chi^2_\alpha(k)$、$F_\alpha(m,n)$ はそれぞれの分布の**上側 $\alpha$ 点**、$\Phi$ は標準正規分布の累積分布関数、$q=1-p$ です。

## 離散型分布

| 分布 | 台 | 確率関数 $P(X=x)$ | 期待値 | 分散 | モーメント母関数 $M(t)$ | 確率母関数 $G(s)$ |
|---|---|---|---|---|---|---|
| ベルヌーイ $\mathrm{Bern}(p)$ | $\{0,1\}$ | $p^x q^{1-x}$ | $p$ | $pq$ | $q+pe^t$ | $q+ps$ |
| 二項 $\mathrm{Bin}(n,p)$ | $\{0,1,\dots,n\}$ | $\binom{n}{x}p^x q^{n-x}$ | $np$ | $npq$ | $(q+pe^t)^n$ | $(q+ps)^n$ |
| ポアソン $\mathrm{Po}(\lambda)$ | $\{0,1,2,\dots\}$ | $\dfrac{\lambda^x e^{-\lambda}}{x!}$ | $\lambda$ | $\lambda$ | $\exp\{\lambda(e^t-1)\}$ | $\exp\{\lambda(s-1)\}$ |
| 幾何 $\mathrm{Geo}(p)$（失敗回数） | $\{0,1,2,\dots\}$ | $p\,q^x$ | $\dfrac{q}{p}$ | $\dfrac{q}{p^2}$ | $\dfrac{p}{1-qe^t}$（$t<-\log q$） | $\dfrac{p}{1-qs}$（$\lvert s\rvert<1/q$） |
| 負の二項 $\mathrm{NB}(r,p)$（失敗回数） | $\{0,1,2,\dots\}$ | $\binom{r+x-1}{x}p^r q^x$ | $\dfrac{rq}{p}$ | $\dfrac{rq}{p^2}$ | $\left(\dfrac{p}{1-qe^t}\right)^r$（$t<-\log q$） | $\left(\dfrac{p}{1-qs}\right)^r$ |
| 超幾何 $\mathrm{HG}(N,K,n)$ | $\max(0,n-N+K)\le x\le\min(n,K)$ | $\dfrac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}$ | $n\dfrac{K}{N}$ | $n\dfrac{K}{N}\left(1-\dfrac{K}{N}\right)\dfrac{N-n}{N-1}$ | 存在する（有限和）が簡単な閉じた形はない | 同左 |
| 離散一様（$1,\dots,m$） | $\{1,\dots,m\}$ | $\dfrac{1}{m}$ | $\dfrac{m+1}{2}$ | $\dfrac{m^2-1}{12}$ | $\dfrac{e^t(e^{mt}-1)}{m(e^t-1)}$（$t\ne0$） | $\dfrac{s(1-s^m)}{m(1-s)}$（$s\ne1$） |

- 幾何分布・負の二項分布は**失敗の回数**（台が 0 から）で定義しています。**試行の回数**で数える流儀では $Y=X+1$（負の二項は $Y=X+r$）となり、期待値は $1/p$（$r/p$）に、MGF・PGF には $e^t$（$s$）の $1$ 乗（$r$ 乗）が掛かります。分散は変わりません。
- 母関数から：$E[X^k]=M^{(k)}(0)$、$E[X]=G'(1)$、$E[X(X-1)]=G''(1)$、$P(X=k)=G^{(k)}(0)/k!$。→ モーメント母関数、確率母関数

## 連続型分布

| 分布 | 台 | 確率密度関数 $f(x)$ | 期待値 | 分散 | モーメント母関数 $M(t)$ |
|---|---|---|---|---|---|
| 一様 $U(a,b)$ | $[a,b]$ | $\dfrac{1}{b-a}$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ | $\dfrac{e^{tb}-e^{ta}}{t(b-a)}$（$t\ne0$）、$M(0)=1$ |
| 正規 $N(\mu,\sigma^2)$ | $(-\infty,\infty)$ | $\dfrac{1}{\sqrt{2\pi}\sigma}\exp\!\left\{-\dfrac{(x-\mu)^2}{2\sigma^2}\right\}$ | $\mu$ | $\sigma^2$ | $\exp\!\left(\mu t+\tfrac12\sigma^2t^2\right)$ |
| 指数 $\mathrm{Exp}(\lambda)$（レート） | $[0,\infty)$ | $\lambda e^{-\lambda x}$ | $\dfrac{1}{\lambda}$ | $\dfrac{1}{\lambda^2}$ | $\dfrac{\lambda}{\lambda-t}$（$t<\lambda$） |
| ガンマ $\mathrm{Ga}(\alpha,\beta)$（形状・尺度） | $(0,\infty)$ | $\dfrac{x^{\alpha-1}e^{-x/\beta}}{\Gamma(\alpha)\beta^\alpha}$ | $\alpha\beta$ | $\alpha\beta^2$ | $(1-\beta t)^{-\alpha}$（$t<1/\beta$） |
| ベータ $\mathrm{Be}(a,b)$ | $(0,1)$ | $\dfrac{x^{a-1}(1-x)^{b-1}}{B(a,b)}$ | $\dfrac{a}{a+b}$ | $\dfrac{ab}{(a+b)^2(a+b+1)}$ | すべての $t$ で存在するが初等関数では書けない（$1+\sum_{k\ge1}\prod_{r=0}^{k-1}\frac{a+r}{a+b+r}\frac{t^k}{k!}$） |
| カイ二乗 $\chi^2(k)$ | $(0,\infty)$ | $\dfrac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)}$ | $k$ | $2k$ | $(1-2t)^{-k/2}$（$t<1/2$） |
| $t$ 分布 $t(\nu)$ | $(-\infty,\infty)$ | $\dfrac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})}\left(1+\dfrac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$ | $0$（$\nu>1$） | $\dfrac{\nu}{\nu-2}$（$\nu>2$） | 存在しない |
| $F$ 分布 $F(m,n)$ | $(0,\infty)$ | $\dfrac{1}{B(\frac m2,\frac n2)}\left(\dfrac{m}{n}\right)^{\frac m2}x^{\frac m2-1}\left(1+\dfrac{m}{n}x\right)^{-\frac{m+n}{2}}$ | $\dfrac{n}{n-2}$（$n>2$） | $\dfrac{2n^2(m+n-2)}{m(n-2)^2(n-4)}$（$n>4$） | 存在しない |
| 対数正規 $LN(\mu,\sigma^2)$ | $(0,\infty)$ | $\dfrac{1}{\sqrt{2\pi}\sigma x}\exp\!\left\{-\dfrac{(\log x-\mu)^2}{2\sigma^2}\right\}$ | $e^{\mu+\sigma^2/2}$ | $e^{2\mu+\sigma^2}(e^{\sigma^2}-1)$ | 存在しない（$t>0$ で発散） |
| コーシー $C(\mu,\gamma)$ | $(-\infty,\infty)$ | $\dfrac{\gamma}{\pi\{\gamma^2+(x-\mu)^2\}}$ | 存在しない | 存在しない | 存在しない |
| ワイブル $W(k,\eta)$（形状・尺度） | $[0,\infty)$ | $\dfrac{k}{\eta}\left(\dfrac{x}{\eta}\right)^{k-1}e^{-(x/\eta)^k}$ | $\eta\,\Gamma(1+\frac1k)$ | $\eta^2\left\{\Gamma(1+\frac2k)-\Gamma(1+\frac1k)^2\right\}$ | 閉じた形はない（$k\ge1$ なら原点の近くで存在、$k<1$ では $t>0$ で発散） |

- ガンマ分布は**形状 $\alpha$・尺度 $\beta$**（平均 $\alpha\beta$）です。レート $\lambda=1/\beta$ で書く流儀では密度が $\lambda^\alpha x^{\alpha-1}e^{-\lambda x}/\Gamma(\alpha)$、平均 $\alpha/\lambda$、MGF $(1-t/\lambda)^{-\alpha}$ になります。$\mathrm{Exp}(\lambda)=\mathrm{Ga}(1,1/\lambda)$、$\chi^2(k)=\mathrm{Ga}(k/2,2)$。
- 指数分布は**レート $\lambda$**（平均 $1/\lambda$）です。平均 $\theta$ で書く本では $\lambda=1/\theta$ と読み替えます。コーシー分布は $t(1)$ と同じ（$\mu=0,\gamma=1$）。→ 指数分布、ガンマ分布、ワイブル分布はハザード関数

## 期待値と分散

**基本の性質**（→ 期待値と分散の性質）

$$
E[aX+b]=aE[X]+b,\qquad V[aX+b]=a^2V[X],\qquad V[X]=E[X^2]-(E[X])^2
$$

$$
V[X\pm Y]=V[X]+V[Y]\pm2\,\mathrm{Cov}(X,Y),\qquad \mathrm{Cov}(aX+b,\,cY+d)=ac\,\mathrm{Cov}(X,Y)
$$

- 相関係数 $\rho=\mathrm{Cov}(X,Y)/\sqrt{V[X]V[Y]}$ は $-1\le\rho\le1$。独立なら $\mathrm{Cov}=0$（逆は成り立たない）。→ 共分散と相関係数

**ベクトル版**：$E[A\boldsymbol X+\boldsymbol b]=A\boldsymbol\mu+\boldsymbol b$、$V[A\boldsymbol X+\boldsymbol b]=A\Sigma A^\top$。線形結合 $\boldsymbol a^\top\boldsymbol X$ の分散は $\boldsymbol a^\top\Sigma\boldsymbol a$。→ 分散共分散行列

**繰り返し期待値と全分散の公式**（→ 条件付き期待値、条件付き分散）

$$
E[Y]=E\big[E[Y\mid X]\big],\qquad V[Y]=E\big[V[Y\mid X]\big]+V\big[E[Y\mid X]\big]
$$

- 混合分布・階層モデル（例：ポアソン分布の平均がガンマ分布に従う）の期待値・分散を出すときに使う。

**標本平均と不偏分散**：独立同分布で $E[X_i]=\mu$、$V[X_i]=\sigma^2$ のとき

$$
E[\bar X]=\mu,\qquad V[\bar X]=\frac{\sigma^2}{n},\qquad s^2=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar X)^2,\quad E[s^2]=\sigma^2
$$

**歪度・尖度**：$\beta_1=E[(X-\mu)^3]/\sigma^3$、$\beta_2=E[(X-\mu)^4]/\sigma^4$（正規分布で $\beta_2=3$。$\beta_2-3$ を超過尖度という）。→ 歪度と尖度

**キュムラント**：$K(t)=\log M(t)$ とおくと $K'(0)=\mu$、$K''(0)=\sigma^2$。MGF の積は和の分布：独立なら $M_{X+Y}(t)=M_X(t)M_Y(t)$。

**2変量正規分布の条件付き分布**（→ 2変量正規分布）

$$
Y\mid X=x\ \sim\ N\!\left(\mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X),\ \sigma_Y^2(1-\rho^2)\right)
$$

**偏相関係数**：$r_{xy\cdot z}=\dfrac{r_{xy}-r_{xz}r_{yz}}{\sqrt{(1-r_{xz}^2)(1-r_{yz}^2)}}$。$z$ の影響を除いた $x,y$ の相関。→ 偏相関係数

## 変数変換

**1変数**：$Y=g(X)$、$g$ が単調なら

$$
f_Y(y)=f_X\big(g^{-1}(y)\big)\left|\frac{d}{dy}g^{-1}(y)\right|
$$

**多変数**：$\boldsymbol Y=\boldsymbol g(\boldsymbol X)$ が1対1なら $f_{\boldsymbol Y}(\boldsymbol y)=f_{\boldsymbol X}(\boldsymbol x(\boldsymbol y))\,\lvert\det(\partial\boldsymbol x/\partial\boldsymbol y)\rvert$。→ 変数変換とヤコビアン

**和の分布（畳み込み）**：独立なら $f_{X+Y}(z)=\displaystyle\int f_X(x)f_Y(z-x)\,dx$。

**再生性**（独立な和）→ 確率変数の和と再生性

| 分布 | 和の分布 |
|---|---|
| 二項（$p$ 共通） | $\mathrm{Bin}(n_1,p)+\mathrm{Bin}(n_2,p)=\mathrm{Bin}(n_1+n_2,p)$ |
| ポアソン | $\mathrm{Po}(\lambda_1)+\mathrm{Po}(\lambda_2)=\mathrm{Po}(\lambda_1+\lambda_2)$ |
| 負の二項（$p$ 共通） | $\mathrm{NB}(r_1,p)+\mathrm{NB}(r_2,p)=\mathrm{NB}(r_1+r_2,p)$ |
| 正規 | $N(\mu_1,\sigma_1^2)+N(\mu_2,\sigma_2^2)=N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$ |
| ガンマ（$\beta$ 共通） | $\mathrm{Ga}(\alpha_1,\beta)+\mathrm{Ga}(\alpha_2,\beta)=\mathrm{Ga}(\alpha_1+\alpha_2,\beta)$ |
| カイ二乗 | $\chi^2(k_1)+\chi^2(k_2)=\chi^2(k_1+k_2)$ |

**正規標本から作る分布**（→ カイ二乗分布、t分布、F分布）

$$
Z\sim N(0,1)\Rightarrow Z^2\sim\chi^2(1),\qquad \frac{(n-1)s^2}{\sigma^2}\sim\chi^2(n-1)\ \ (\bar X\text{ と独立})
$$

$$
T=\frac{Z}{\sqrt{W/k}}\sim t(k),\qquad F=\frac{W_1/m}{W_2/n}\sim F(m,n),\qquad T^2\sim F(1,k),\qquad \frac1F\sim F(n,m)
$$

**よく使う変換**

| 変換 | 式 | 使いどころ |
|---|---|---|
| 逆関数法 | $U\sim U(0,1)\Rightarrow F^{-1}(U)\sim F$。例：$-\frac1\lambda\log U\sim\mathrm{Exp}(\lambda)$ | 乱数の生成 → 逆関数法 |
| 対数正規 | $X\sim N(\mu,\sigma^2)\Rightarrow e^X\sim LN(\mu,\sigma^2)$ | 右に裾の長いデータ → 対数変換 |
| Box-Cox | $y^{(\lambda)}=\dfrac{y^\lambda-1}{\lambda}$（$\lambda\ne0$）、$\log y$（$\lambda=0$） | 正規性・等分散に近づける → Box-Cox変換 |
| ロジット | $\log\dfrac{p}{1-p}$、逆は $\dfrac{e^x}{1+e^x}$ | 確率を実数全体に広げる → ロジット変換・ロジスティック変換 |
| ボックス・ミュラー | $\sqrt{-2\log U_1}\cos(2\pi U_2)$、$\sqrt{-2\log U_1}\sin(2\pi U_2)$ は独立な $N(0,1)$ | 正規乱数 → ボックス・ミュラー法 |

**順序統計量**：連続分布 $F$ からの $n$ 個のうち $k$ 番目に小さい値 $X_{(k)}$ の密度

$$
f_{(k)}(x)=\frac{n!}{(k-1)!\,(n-k)!}F(x)^{k-1}\{1-F(x)\}^{n-k}f(x)
$$

- 最大値は $F(x)^n$、最小値は $1-\{1-F(x)\}^n$ が累積分布関数。$U(0,1)$ なら $X_{(k)}\sim\mathrm{Be}(k,n-k+1)$。→ 順序統計量

## 極限定理

| 名前 | 式 | 使いどころ |
|---|---|---|
| マルコフの不等式 | $P(X\ge a)\le\dfrac{E[X]}{a}$（$X\ge0$、$a>0$） | 期待値だけで裾の確率を抑える → マルコフの不等式 |
| チェビシェフの不等式 | $P(\lvert X-\mu\rvert\ge k\sigma)\le\dfrac{1}{k^2}$ | 分布の形を仮定しない評価 → チェビシェフの不等式 |
| 大数の弱法則 | $\bar X_n\xrightarrow{p}\mu$ | 標本平均の一致性 → 大数の弱法則 |
| 中心極限定理 | $\dfrac{\sqrt n(\bar X_n-\mu)}{\sigma}\xrightarrow{d}N(0,1)$ | 標本平均の近似分布 → 中心極限定理 |
| スラツキーの定理 | $X_n\xrightarrow{d}X$、$Y_n\xrightarrow{p}c$ なら $X_n+Y_n\xrightarrow{d}X+c$、$X_nY_n\xrightarrow{d}cX$ | $\sigma$ を $s$ に置き換えてよい理由 → スラツキーの定理 |
| デルタ法 | $\sqrt n\{g(\bar X_n)-g(\mu)\}\xrightarrow{d}N\big(0,\ g'(\mu)^2\sigma^2\big)$ | 推定量の関数の標準誤差 → デルタ法 |
| 少数の法則 | $np=\lambda$ を保って $n\to\infty$ なら $\mathrm{Bin}(n,p)\to\mathrm{Po}(\lambda)$ | まれな事象の件数 → 少数の法則 |

**二項分布の正規近似（連続修正つき）**：$X\sim\mathrm{Bin}(n,p)$ なら

$$
P(X\le k)\approx\Phi\!\left(\frac{k+0.5-np}{\sqrt{np(1-p)}}\right)
$$

- 整数値の確率を連続分布で近似するとき、境界を 0.5 ずらす。→ 連続修正、二項分布・ポアソン分布の正規近似

**分散安定化変換**（デルタ法の応用）→ 分散安定化変換、フィッシャーのz変換

| 対象 | 変換 | 近似分散 |
|---|---|---|
| 比率 $\hat p$ | $\arcsin\sqrt{\hat p}$ | $\dfrac{1}{4n}$ |
| ポアソン $X\sim\mathrm{Po}(\lambda)$ | $\sqrt X$ | $\dfrac14$ |
| 相関係数 $r$ | $z=\dfrac12\log\dfrac{1+r}{1-r}$ | $\dfrac{1}{n-3}$ |

## 推定量の性質

**平均二乗誤差の分解**：$\mathrm{MSE}(\hat\theta)=E[(\hat\theta-\theta)^2]=V[\hat\theta]+\{E[\hat\theta]-\theta\}^2$。→ 平均二乗誤差

- 不偏性：$E[\hat\theta]=\theta$。一致性：$\hat\theta_n\xrightarrow{p}\theta$（MSE→0 なら一致）。→ 不偏推定量、一致推定量

**スコア関数とフィッシャー情報量**（1個あたり）→ スコア関数、フィッシャー情報量

$$
U(\theta)=\frac{\partial}{\partial\theta}\log f(X;\theta),\qquad E[U]=0,\qquad I(\theta)=E[U^2]=-E\!\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\right]
$$

- 独立な $n$ 個では $I_n(\theta)=nI(\theta)$。

**クラメール・ラオの不等式**：不偏推定量 $T$ について（→ クラメール・ラオの不等式、有効推定量）

$$
V[T]\ge\frac{1}{nI(\theta)},\qquad g(\theta)\text{ の不偏推定量なら }V[T]\ge\frac{g'(\theta)^2}{nI(\theta)}
$$

**最尤推定量の漸近正規性**：$\sqrt n(\hat\theta-\theta)\xrightarrow{d}N\big(0,\,1/I(\theta)\big)$。標準誤差は $1/\sqrt{nI(\hat\theta)}$。不変性：$g(\theta)$ の最尤推定量は $g(\hat\theta)$。→ 最尤推定量の漸近正規性、最尤推定量の不変性

**代表的な分布の最尤推定量と情報量**

| 分布 | 最尤推定量 | $I(\theta)$（1個あたり） |
|---|---|---|
| $\mathrm{Bern}(p)$ | $\hat p=\bar x$ | $\dfrac{1}{p(1-p)}$ |
| $\mathrm{Po}(\lambda)$ | $\hat\lambda=\bar x$ | $\dfrac1\lambda$ |
| $\mathrm{Exp}(\lambda)$ | $\hat\lambda=1/\bar x$ | $\dfrac{1}{\lambda^2}$ |
| $N(\mu,\sigma^2)$（$\sigma^2$ 既知） | $\hat\mu=\bar x$ | $\dfrac{1}{\sigma^2}$ |
| $N(\mu,\sigma^2)$ の $\sigma^2$ | $\hat\sigma^2=\dfrac1n\sum(x_i-\bar x)^2$（不偏でない） | $\dfrac{1}{2\sigma^4}$（$\mu$ 既知） |
| $U(0,\theta)$ | $\hat\theta=\max x_i$（$E=\frac{n}{n+1}\theta$） | 正則条件を満たさない |

**十分統計量**：ネイマンの分解定理 $f(\boldsymbol x;\theta)=g(T(\boldsymbol x),\theta)\,h(\boldsymbol x)$。ラオ・ブラックウェル：$E[\tilde\theta\mid T]$ は分散を増やさない。→ ネイマンの分解定理、ラオ・ブラックウェルの定理

**モーメント法**：$E[X^k]=\frac1n\sum x_i^k$ を $\theta$ について解く。例：ガンマ分布 $\hat\alpha=\bar x^2/\hat\sigma^2$、$\hat\beta=\hat\sigma^2/\bar x$（$\hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$）。→ モーメント法

## 区間推定

信頼係数 $1-\alpha$ の両側区間。→ 信頼区間

| 対象 | 区間 | 条件・補足 |
|---|---|---|
| 母平均（$\sigma$ 既知） | $\bar x\pm z_{\alpha/2}\dfrac{\sigma}{\sqrt n}$ | 正規母集団、または $n$ が大きい → 母平均の区間推定（分散既知） |
| 母平均（$\sigma$ 未知） | $\bar x\pm t_{\alpha/2}(n-1)\dfrac{s}{\sqrt n}$ | 正規母集団 → 母平均の区間推定（分散未知） |
| 母分散 | $\left[\dfrac{(n-1)s^2}{\chi^2_{\alpha/2}(n-1)},\ \dfrac{(n-1)s^2}{\chi^2_{1-\alpha/2}(n-1)}\right]$ | 左右非対称 → 母分散の区間推定 |
| 母比率（ワルド） | $\hat p\pm z_{\alpha/2}\sqrt{\dfrac{\hat p(1-\hat p)}{n}}$ | $n\hat p$、$n(1-\hat p)$ が十分大きい → ワルド型信頼区間 |
| 母比率（ウィルソン） | $\dfrac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}$（$z=z_{\alpha/2}$） | $\hat p$ が 0 や 1 に近くても使える → ウィルソン区間（スコア区間） |
| ポアソン平均 | $\hat\lambda\pm z_{\alpha/2}\sqrt{\hat\lambda/n}$ | 件数データ、$n\hat\lambda$ が大きい |
| 平均の差（等分散） | $\bar x-\bar y\pm t_{\alpha/2}(m+n-2)\,s_p\sqrt{\dfrac1m+\dfrac1n}$ | $s_p^2=\dfrac{(m-1)s_1^2+(n-1)s_2^2}{m+n-2}$ → プールした分散 |
| 平均の差（不等分散） | $\bar x-\bar y\pm t_{\alpha/2}(\nu)\sqrt{\dfrac{s_1^2}{m}+\dfrac{s_2^2}{n}}$ | $\nu$ は次節のウェルチの自由度 → 2標本の母平均の差の区間推定 |
| 対応のある差 | $\bar d\pm t_{\alpha/2}(n-1)\dfrac{s_d}{\sqrt n}$ | 差 $d_i=x_i-y_i$ の1標本問題 → 対応のある場合の区間推定 |
| 比率の差 | $\hat p_1-\hat p_2\pm z_{\alpha/2}\sqrt{\dfrac{\hat p_1(1-\hat p_1)}{m}+\dfrac{\hat p_2(1-\hat p_2)}{n}}$ | 独立な2群 → 2標本の母比率の差の区間推定 |
| 分散比 $\sigma_1^2/\sigma_2^2$ | $\left[\dfrac{s_1^2/s_2^2}{F_{\alpha/2}(m-1,n-1)},\ \dfrac{s_1^2}{s_2^2}F_{\alpha/2}(n-1,m-1)\right]$ | $1/F_{\alpha/2}(m-1,n-1)=F_{1-\alpha/2}(n-1,m-1)$ → 母分散比の区間推定 |
| 母相関係数 | $\tanh\!\left(z\pm\dfrac{z_{\alpha/2}}{\sqrt{n-3}}\right)$、$z=\operatorname{artanh} r$ | 2変量正規 → フィッシャーのz変換 |

**必要な標本サイズ**（誤差幅 $E$ 以内）→ 必要な標本サイズ

$$
n\ge\left(\frac{z_{\alpha/2}\,\sigma}{E}\right)^2,\qquad \text{比率なら}\quad n\ge\frac{z_{\alpha/2}^2\,p(1-p)}{E^2}\le\frac{z_{\alpha/2}^2}{4E^2}
$$

- $p$ の見当がないときは $p=1/2$（最も大きい $n$）を使う。

## 検定統計量

帰無仮説のもとでの分布。→ 帰無仮説と対立仮説、棄却域、p値

| 検定 | 統計量 | 帰無分布 |
|---|---|---|
| 母平均（$\sigma$ 既知） | $Z=\dfrac{\bar x-\mu_0}{\sigma/\sqrt n}$ | $N(0,1)$ → 母平均の検定（分散既知） |
| 1標本 $t$ | $T=\dfrac{\bar x-\mu_0}{s/\sqrt n}$ | $t(n-1)$ → 1標本t検定 |
| 2標本 $t$（等分散） | $T=\dfrac{\bar x-\bar y}{s_p\sqrt{1/m+1/n}}$ | $t(m+n-2)$ → 2標本t検定 |
| ウェルチ | $T=\dfrac{\bar x-\bar y}{\sqrt{s_1^2/m+s_2^2/n}}$ | 近似 $t(\nu)$ → ウェルチの検定 |
| 対応のある $t$ | $T=\dfrac{\bar d}{s_d/\sqrt n}$ | $t(n-1)$ → 対応のあるt検定 |
| 母分散 | $\chi^2=\dfrac{(n-1)s^2}{\sigma_0^2}$ | $\chi^2(n-1)$ → 母分散のカイ二乗検定 |
| 等分散 | $F=\dfrac{s_1^2}{s_2^2}$ | $F(m-1,n-1)$ → 等分散のF検定 |
| 母比率 | $Z=\dfrac{\hat p-p_0}{\sqrt{p_0(1-p_0)/n}}$ | 近似 $N(0,1)$ → 母比率の検定 |
| 比率の差 | $Z=\dfrac{\hat p_1-\hat p_2}{\sqrt{\bar p(1-\bar p)(1/m+1/n)}}$、$\bar p=\dfrac{x_1+x_2}{m+n}$ | 近似 $N(0,1)$ → 2標本の比率の差の検定 |
| 無相関 | $T=\dfrac{r\sqrt{n-2}}{\sqrt{1-r^2}}$ | $t(n-2)$ → 母相関係数の検定 |
| ホテリングの $T^2$ | $T^2=n(\bar{\boldsymbol x}-\boldsymbol\mu_0)^\top S^{-1}(\bar{\boldsymbol x}-\boldsymbol\mu_0)$ | $\dfrac{n-p}{p(n-1)}T^2\sim F(p,n-p)$ → ホテリングのT二乗検定 |

**ウェルチの自由度**（ウェルチ・サタースウェイトの近似）

$$
\nu=\frac{\left(\dfrac{s_1^2}{m}+\dfrac{s_2^2}{n}\right)^2}{\dfrac{(s_1^2/m)^2}{m-1}+\dfrac{(s_2^2/n)^2}{n-1}}
$$

**検出力と標本サイズ**（片側 $z$ 検定、$H_0:\mu=\mu_0$、真の値 $\mu_1>\mu_0$）→ 検出力・検出力関数、検定のための標本サイズ設計

$$
1-\beta=\Phi\!\left(\frac{\sqrt n\,(\mu_1-\mu_0)}{\sigma}-z_\alpha\right),\qquad n=\left\{\frac{(z_\alpha+z_\beta)\sigma}{\mu_1-\mu_0}\right\}^2
$$

- 両側検定なら $z_\alpha$ を $z_{\alpha/2}$ に置き換える（反対側の小さな確率は無視）。

**多重検定**（$m$ 個の検定）→ ボンフェローニ補正、ホルム法、偽発見率

| 方法 | 棄却のルール |
|---|---|
| ボンフェローニ | $p_i\le\alpha/m$ |
| ホルム | $p$ 値を小さい順に $p_{(1)}\le\cdots$ と並べ、$p_{(i)}\le\alpha/(m-i+1)$ を満たす限り順に棄却 |
| ベンジャミニ・ホッホベルク（FDR） | $p_{(i)}\le iq/m$ を満たす最大の $i$ までを棄却 |

## 尤度による検定

**尤度比検定**：$\Lambda=\dfrac{\sup_{\theta\in\Theta_0}L(\theta)}{\sup_{\theta\in\Theta}L(\theta)}$、ウィルクスの定理より

$$
-2\log\Lambda\xrightarrow{d}\chi^2(r),\qquad r=\dim\Theta-\dim\Theta_0
$$

- 制約の個数が自由度。正規分布以外の検定を作るときの基本。→ 尤度比検定、ウィルクスの定理

**3つの漸近検定**（1パラメータ、$H_0:\theta=\theta_0$。どれも漸近的に $\chi^2(1)$）→ ワルド検定、スコア検定

| 検定 | 統計量 | 評価する点 |
|---|---|---|
| 尤度比 | $2\{\ell(\hat\theta)-\ell(\theta_0)\}$ | 両方 |
| ワルド | $(\hat\theta-\theta_0)^2\,nI(\hat\theta)$ | 最尤推定値 |
| スコア | $\dfrac{U_n(\theta_0)^2}{nI(\theta_0)}$ | 帰無仮説の値 |

- 例：二項の比率のスコア検定は $Z=(\hat p-p_0)/\sqrt{p_0(1-p_0)/n}$、ワルド検定は分母に $\hat p$ を使う。

**ネイマン・ピアソンの補題**：単純仮説どうしでは、$L(\theta_1)/L(\theta_0)>k$ で棄却する検定が最強力。→ ネイマン・ピアソンの補題

## カイ二乗検定

**ピアソンのカイ二乗統計量と G 統計量**（→ ピアソンのカイ二乗統計量、尤度比検定統計量（G統計量））

$$
\chi^2=\sum\frac{(O-E)^2}{E},\qquad G=2\sum O\log\frac{O}{E}
$$

| 検定 | 期待度数 $E$ | 自由度 |
|---|---|---|
| 適合度（$k$ カテゴリ） | $np_{0i}$ | $k-1-$（推定した母数の数） → 適合度検定の自由度（推定母数の影響） |
| 独立性・同質性（$r\times c$） | $E_{ij}=\dfrac{n_{i\cdot}\,n_{\cdot j}}{n}$ | $(r-1)(c-1)$ → 独立性の検定、同質性の検定 |

- 期待度数がおおむね 5 以上なら $\chi^2$ 近似が使える。→ カイ二乗近似の条件

**2×2 表の簡便式**（セル $a,b,c,d$、$n=a+b+c+d$）→ 2×2分割表、イェーツの連続修正

$$
\chi^2=\frac{n(ad-bc)^2}{(a+b)(c+d)(a+c)(b+d)},\qquad \chi^2_{\text{Yates}}=\frac{n\left(\lvert ad-bc\rvert-\frac n2\right)^2}{(a+b)(c+d)(a+c)(b+d)}
$$

**マクネマー検定**（対応のある2×2、不一致セル $b,c$）：$\chi^2=\dfrac{(b-c)^2}{b+c}\sim\chi^2(1)$。→ マクネマー検定

**過分散の検定**：ポアソンの仮定のもと $\sum(x_i-\bar x)^2/\bar x\sim\chi^2(n-1)$（近似）。→ 分散検定（過分散の検定）

## ノンパラメトリック

帰無仮説のもとでの期待値・分散（同順位なし）。$n$ が大きいときは正規近似で検定する。→ 順位統計量の正規近似

| 検定・統計量 | 定義 | 期待値 | 分散 |
|---|---|---|---|
| 符号検定 | 正の符号の数 $S$ | $\dfrac n2$ | $\dfrac n4$（$S\sim\mathrm{Bin}(n,\frac12)$） |
| ウィルコクソンの符号付き順位 | 正の差の順位和 $T^+$ | $\dfrac{n(n+1)}{4}$ | $\dfrac{n(n+1)(2n+1)}{24}$ |
| ウィルコクソンの順位和 | 第1群（$m$ 個）の順位和 $W$、$N=m+n$ | $\dfrac{m(N+1)}{2}$ | $\dfrac{mn(N+1)}{12}$ |
| マン・ホイットニーの $U$ | $U=W-\dfrac{m(m+1)}{2}$ | $\dfrac{mn}{2}$ | $\dfrac{mn(N+1)}{12}$ |
| ケンドールの $\tau$ | $\dfrac{P-Q}{n(n-1)/2}$（$P,Q$ は一致対・不一致対の数） | $0$ | $\dfrac{2(2n+5)}{9n(n-1)}$ |

**スピアマンの順位相関**：$r_s=1-\dfrac{6\sum d_i^2}{n(n^2-1)}$（$d_i$ は順位の差）。→ スピアマンの順位相関係数

**クラスカル・ウォリス検定**：$H=\dfrac{12}{N(N+1)}\displaystyle\sum_{i=1}^k\frac{R_i^2}{n_i}-3(N+1)$、近似 $\chi^2(k-1)$。$R_i$ は群 $i$ の順位和。→ クラスカル・ウォリス検定

## 回帰分析

モデル $\boldsymbol y=X\boldsymbol\beta+\boldsymbol\varepsilon$、$\varepsilon_i$ は独立に $N(0,\sigma^2)$、説明変数 $p$ 個・切片あり（係数 $p+1$ 個）。→ 重回帰モデル

**最小二乗推定量**（→ 最小二乗法、正規方程式、ガウス・マルコフの定理）

$$
\hat{\boldsymbol\beta}=(X^\top X)^{-1}X^\top\boldsymbol y,\qquad V[\hat{\boldsymbol\beta}]=\sigma^2(X^\top X)^{-1},\qquad \hat\sigma^2=\frac{S_e}{n-p-1}
$$

- 単回帰：$\hat\beta_1=\dfrac{S_{xy}}{S_{xx}}$、$\hat\beta_0=\bar y-\hat\beta_1\bar x$、$V[\hat\beta_1]=\dfrac{\sigma^2}{S_{xx}}$。

**平方和の分解と決定係数**（→ 決定係数、自由度調整済み決定係数）

$$
S_T=\sum(y_i-\bar y)^2=S_R+S_e,\qquad R^2=\frac{S_R}{S_T}=1-\frac{S_e}{S_T},\qquad \bar R^2=1-\frac{S_e/(n-p-1)}{S_T/(n-1)}
$$

- 単回帰では $R^2=r_{xy}^2$。$R^2$ は重相関係数（$y$ と $\hat y$ の相関）の2乗。

**係数の $t$ 検定と全体の $F$ 検定**（→ 回帰係数のt検定、回帰全体のF検定、部分F検定）

$$
t_j=\frac{\hat\beta_j}{\hat\sigma\sqrt{[(X^\top X)^{-1}]_{jj}}}\sim t(n-p-1),\qquad F=\frac{S_R/p}{S_e/(n-p-1)}=\frac{R^2/p}{(1-R^2)/(n-p-1)}\sim F(p,n-p-1)
$$

- 部分 $F$（$q$ 個の係数が 0 か）：$F=\dfrac{(S_{e0}-S_{e1})/q}{S_{e1}/(n-p-1)}\sim F(q,n-p-1)$。$q=1$ なら $F=t_j^2$。

**信頼区間と予測区間**（$\boldsymbol x_0$ での）→ 回帰の信頼区間と予測区間

$$
\hat y_0\pm t_{\alpha/2}(n-p-1)\,\hat\sigma\sqrt{\boldsymbol x_0^\top(X^\top X)^{-1}\boldsymbol x_0}\quad(\text{平均}),\qquad \hat y_0\pm t_{\alpha/2}(n-p-1)\,\hat\sigma\sqrt{1+\boldsymbol x_0^\top(X^\top X)^{-1}\boldsymbol x_0}\quad(\text{予測})
$$

**その他**

| 名前 | 式 | 補足 |
|---|---|---|
| 分散拡大要因 | $\mathrm{VIF}_j=\dfrac{1}{1-R_j^2}$ | $R_j^2$ は $x_j$ を他の説明変数で回帰した決定係数。10 超で多重共線性を疑う → 分散拡大要因 |
| 標準化偏回帰係数 | $\hat\beta_j\,\dfrac{s_{x_j}}{s_y}$ | 単位によらず効果を比べる → 標準化偏回帰係数 |
| 加重・一般化最小二乗 | $(X^\top WX)^{-1}X^\top W\boldsymbol y$（$W=\Sigma^{-1}$） | 不均一分散・相関のある誤差 → 一般化最小二乗法 |
| リッジ回帰 | $(X^\top X+\lambda I)^{-1}X^\top\boldsymbol y$ | 縮小推定、多重共線性に強い → リッジ回帰 |
| Lasso | $\sum(y_i-\boldsymbol x_i^\top\boldsymbol\beta)^2+\lambda\sum\lvert\beta_j\rvert$ を最小化 | 係数がちょうど 0 になり変数選択ができる → Lasso |
| ロジスティック回帰 | $\log\dfrac{p}{1-p}=\boldsymbol x^\top\boldsymbol\beta$ | $e^{\beta_j}$ は $x_j$ が1増えたときのオッズ比 → ロジスティック回帰 |
| ポアソン回帰 | $\log\mu=\boldsymbol x^\top\boldsymbol\beta$（+オフセット $\log t$） | 件数データ。$e^{\beta_j}$ は率の比 → ポアソン回帰 |

## 回帰診断

$H=X(X^\top X)^{-1}X^\top$（ハット行列）、残差 $\boldsymbol e=(I-H)\boldsymbol y$、$s^2=S_e/(n-p-1)$。→ ハット行列

| 名前 | 式 | 使いどころ |
|---|---|---|
| てこ比 | $h_{ii}$、$\sum h_{ii}=p+1$、$\frac1n\le h_{ii}\le1$ | $2(p+1)/n$ を超えると大きい → てこ比 |
| 残差の分散 | $V[e_i]=\sigma^2(1-h_{ii})$ | てこ比の大きい点ほど残差が小さくなりやすい → 残差 |
| 標準化残差 | $r_i=\dfrac{e_i}{s\sqrt{1-h_{ii}}}$ | 外れ値の目安（$\lvert r_i\rvert>2$〜$3$） → 標準化残差 |
| スチューデント化残差 | $t_i=\dfrac{e_i}{s_{(i)}\sqrt{1-h_{ii}}}\sim t(n-p-2)$ | $s_{(i)}$ は $i$ を除いた推定値 → スチューデント化残差 |
| クックの距離 | $D_i=\dfrac{r_i^2}{p+1}\cdot\dfrac{h_{ii}}{1-h_{ii}}$ | 1点を除いたときの当てはめの変化。1 前後以上で要注意 → クックの距離 |
| DFFITS | $t_i\sqrt{\dfrac{h_{ii}}{1-h_{ii}}}$ | 自分の予測値の変化 → DFFITS |
| 一個抜き残差 | $y_i-\hat y_{i(i)}=\dfrac{e_i}{1-h_{ii}}$ | 当てはめ直さずに LOOCV ができる → 一個抜き交差検証 |
| ダービン・ワトソン比 | $DW=\dfrac{\sum_{t=2}^n(e_t-e_{t-1})^2}{\sum_{t=1}^ne_t^2}\approx2(1-\hat\rho_1)$ | 2 に近ければ系列相関なし → ダービン・ワトソン比 |

## モデル選択

$\ell$ は最大対数尤度、$k$ はパラメータ数、$n$ は標本サイズ。小さいほど良い。→ AIC、BIC、補正AIC、マローズのCp

| 規準 | 式 | 補足 |
|---|---|---|
| AIC | $-2\ell+2k$ | 予測の良さ（KL 情報量）を目標にする |
| BIC | $-2\ell+k\log n$ | $n\ge8$ で AIC より罰則が重い。真のモデルを選ぶ一致性 |
| 補正AIC | $\mathrm{AIC}+\dfrac{2k(k+1)}{n-k-1}$ | 小標本で AIC を補正 |
| マローズの $C_p$ | $\dfrac{S_e(p)}{\hat\sigma^2}-n+2p$ | ここだけ $p$ は**切片を含む係数の数**。$\hat\sigma^2$ は最大モデルの不偏分散。良いモデルで $C_p\approx p$ |

- 正規線形回帰では $-2\ell=n\log(2\pi S_e/n)+n$ なので、$\mathrm{AIC}=n\log(S_e/n)+2k+$定数。
- カルバック・ライブラー情報量 $D(g\Vert f)=E_g\!\left[\log\dfrac{g(X)}{f(X)}\right]\ge0$（等号は $g=f$）。→ カルバック・ライブラー情報量
- 予測誤差の分解：$E[(Y-\hat f(x))^2]=\sigma^2+\{\text{バイアス}\}^2+\text{分散}$。→ バイアス・バリアンスのトレードオフ

## 分散分析

**一元配置**（$a$ 水準、水準 $i$ の繰り返し $n_i$、$N=\sum n_i$）→ 一元配置分散分析、平方和の分解

$$
\underbrace{\sum_{i,j}(y_{ij}-\bar y)^2}_{S_T}=\underbrace{\sum_i n_i(\bar y_{i\cdot}-\bar y)^2}_{S_A}+\underbrace{\sum_{i,j}(y_{ij}-\bar y_{i\cdot})^2}_{S_E},\qquad F=\frac{S_A/(a-1)}{S_E/(N-a)}\sim F(a-1,N-a)
$$

**自由度の分解**

| 計画 | 要因 | 自由度 |
|---|---|---|
| 一元配置 | A / 誤差 / 計 | $a-1$ / $N-a$ / $N-1$ |
| 二元配置（繰り返し $r$） | A / B / A×B / 誤差 | $a-1$ / $b-1$ / $(a-1)(b-1)$ / $ab(r-1)$ |
| 二元配置（繰り返しなし）・乱塊法 | A / B（ブロック）/ 誤差 | $a-1$ / $b-1$ / $(a-1)(b-1)$ |
| $k\times k$ ラテン方格 | 行 / 列 / 処理 / 誤差 | $k-1$ / $k-1$ / $k-1$ / $(k-1)(k-2)$ |

- 繰り返しなしの二元配置では交互作用と誤差を分けられない。→ 二元配置分散分析（繰り返しなし）、ラテン方格法

**期待平均平方**（一元配置、各水準 $n$ 個）→ 期待平均平方、母数模型と変量模型

$$
E[MS_E]=\sigma^2,\qquad E[MS_A]=\sigma^2+\frac{n\sum\alpha_i^2}{a-1}\ (\text{母数模型}),\qquad E[MS_A]=\sigma^2+n\sigma_A^2\ (\text{変量模型})
$$

- 変量模型の分散成分の推定：$\hat\sigma_A^2=(MS_A-MS_E)/n$。

**効果の大きさ**：$\eta^2=S_A/S_T$。テューキーの方法：$q=\dfrac{\lvert\bar y_i-\bar y_j\rvert}{\sqrt{MS_E/n}}$ をスチューデント化された範囲の分布と比べる。→ 効果量と寄与率、テューキーの多重比較

## 標本調査

母集団 $N$ 個、標本 $n$ 個、$f=n/N$、$S^2=\frac{1}{N-1}\sum(Y_i-\bar Y)^2$。→ 単純無作為抽出、有限母集団修正

| 量 | 式 | 補足 |
|---|---|---|
| 単純無作為抽出の標本平均 | $V(\bar y)=(1-f)\dfrac{S^2}{n}$ | $1-f$ が有限母集団修正。推定には $S^2$ を $s^2$ に |
| $N$ で割る $\sigma^2$ での形 | $V(\bar y)=\dfrac{N-n}{N-1}\cdot\dfrac{\sigma^2}{n}$ | 超幾何分布の分散と同じ形 |
| 母比率 | $V(\hat p)=\dfrac{N-n}{N-1}\cdot\dfrac{P(1-P)}{n}$ | 推定は $(1-f)\hat p(1-\hat p)/(n-1)$ |
| 母集団総計 | $V(N\bar y)=N^2(1-f)\dfrac{S^2}{n}$ | |
| 標本サイズ（有限母集団） | $n=\dfrac{n_0}{1+n_0/N}$、$n_0=\left(\dfrac{z_{\alpha/2}S}{E}\right)^2$ | 誤差幅 $E$ を達成する最小の $n$ |

**層化抽出**（層 $h$ の大きさ $N_h$、$W_h=N_h/N$、層内で単純無作為抽出）→ 層化抽出法、比例配分、ネイマン配分

$$
\bar y_{st}=\sum_h W_h\bar y_h,\qquad V(\bar y_{st})=\sum_h W_h^2\left(1-\frac{n_h}{N_h}\right)\frac{S_h^2}{n_h}
$$

| 配分 | $n_h$ | 分散 |
|---|---|---|
| 比例配分 | $n\dfrac{N_h}{N}$ | $\dfrac{1-f}{n}\sum_h W_hS_h^2$ |
| ネイマン配分 | $n\dfrac{N_hS_h}{\sum_k N_kS_k}$ | $\dfrac{(\sum_h W_hS_h)^2}{n}-\dfrac{\sum_h W_hS_h^2}{N}$ |

- 層間のばらつきが分散から消えるので、層内が均質なほど得をする。

**その他の推定量**

| 名前 | 式 | 補足 |
|---|---|---|
| ホルビッツ・トンプソン | $\hat Y=\sum_{i\in s}\dfrac{y_i}{\pi_i}$ | 包含確率 $\pi_i$ で割る。任意の抽出法で総計の不偏推定 → ホルビッツ・トンプソン推定量 |
| 比推定 | $\hat{\bar Y}_R=\dfrac{\bar y}{\bar x}\bar X$、$V\approx\dfrac{1-f}{n}(S_y^2-2RS_{xy}+R^2S_x^2)$ | 補助変数 $x$ の母平均 $\bar X$ が既知 → 比推定 |
| 回帰推定 | $\bar y+b(\bar X-\bar x)$、$V\approx\dfrac{1-f}{n}S_y^2(1-\rho^2)$ | → 回帰推定 |
| 計画効果 | $\mathrm{deff}=\dfrac{V_{\text{その抽出法}}}{V_{\text{単純無作為}}}$ | 集落抽出では級内相関 $\rho$、集落の大きさ $M$ で $\approx1+(M-1)\rho$ → 計画効果 |

## 多変量解析

**主成分分析**：分散共分散行列（または相関行列）$\Sigma$ の固有値 $\lambda_1\ge\cdots\ge\lambda_p$、固有ベクトル $\boldsymbol a_j$（$\lVert\boldsymbol a_j\rVert=1$）→ 主成分分析、寄与率、主成分負荷量

$$
Z_j=\boldsymbol a_j^\top\boldsymbol X,\qquad V[Z_j]=\lambda_j,\qquad \text{寄与率}=\frac{\lambda_j}{\sum_k\lambda_k}=\frac{\lambda_j}{\operatorname{tr}\Sigma},\qquad \mathrm{Corr}(Z_j,X_k)=\frac{\sqrt{\lambda_j}\,a_{jk}}{\sigma_k}
$$

- 相関行列から求めると固有値の和は $p$。カイザー基準は固有値 1 以上を残す。→ 相関行列からの主成分分析、カイザー基準

**マハラノビス距離**：$D^2=(\boldsymbol x-\boldsymbol\mu)^\top\Sigma^{-1}(\boldsymbol x-\boldsymbol\mu)$。$\boldsymbol X\sim N_p(\boldsymbol\mu,\Sigma)$ なら $D^2\sim\chi^2(p)$。→ マハラノビス距離

**線形判別**（共通の $\Sigma$、事前確率 $\pi_1,\pi_2$）→ 線形判別分析、フィッシャーの線形判別関数、誤判別率

$$
h(\boldsymbol x)=(\boldsymbol\mu_1-\boldsymbol\mu_2)^\top\Sigma^{-1}\left(\boldsymbol x-\frac{\boldsymbol\mu_1+\boldsymbol\mu_2}{2}\right)+\log\frac{\pi_1}{\pi_2}>0\ \Rightarrow\ \text{群1}
$$

- フィッシャーの判別方向は $\boldsymbol w\propto S_W^{-1}(\bar{\boldsymbol x}_1-\bar{\boldsymbol x}_2)$。正規分布・等事前確率のときの誤判別率は $\Phi(-\Delta/2)$（$\Delta$ は群平均間のマハラノビス距離）。
- 共分散が群で違うと $\log$ 密度の差が2次式になる。→ 二次判別分析

**分類の評価**（TP, FP, FN, TN）→ 感度と特異度、適合率・再現率・F値

| 指標 | 式 |
|---|---|
| 感度（再現率） | $\dfrac{TP}{TP+FN}$ |
| 特異度 | $\dfrac{TN}{TN+FP}$ |
| 適合率 | $\dfrac{TP}{TP+FP}$ |
| F値 | $\dfrac{2\cdot\text{適合率}\cdot\text{再現率}}{\text{適合率}+\text{再現率}}$ |

**因子分析**：$\Sigma=\Lambda\Lambda^\top+\Psi$、共通性 $h_i^2=\sum_k\lambda_{ik}^2$、標準化変数なら独自分散 $\psi_i=1-h_i^2$。→ 因子分析モデル、共通性

**ウォード法**：クラスター $A,B$ を併合したときの群内平方和の増加 $\Delta=\dfrac{n_An_B}{n_A+n_B}\lVert\bar{\boldsymbol x}_A-\bar{\boldsymbol x}_B\rVert^2$。→ ウォード法

## 時系列解析

自己共分散 $\gamma(h)=\mathrm{Cov}(X_t,X_{t+h})$、自己相関 $\rho(h)=\gamma(h)/\gamma(0)$、$\varepsilon_t\sim WN(0,\sigma^2)$。→ 自己共分散・自己相関、弱定常性

| モデル | 式 | 平均・分散 | 自己相関 | 条件 |
|---|---|---|---|---|
| AR(1) | $X_t=c+\phi X_{t-1}+\varepsilon_t$ | $\mu=\dfrac{c}{1-\phi}$、$\gamma(0)=\dfrac{\sigma^2}{1-\phi^2}$ | $\rho(h)=\phi^{\lvert h\rvert}$ | 定常 $\lvert\phi\rvert<1$ |
| AR(2) | $X_t=c+\phi_1X_{t-1}+\phi_2X_{t-2}+\varepsilon_t$ | $\mu=\dfrac{c}{1-\phi_1-\phi_2}$ | $\rho(1)=\dfrac{\phi_1}{1-\phi_2}$、$\rho(h)=\phi_1\rho(h-1)+\phi_2\rho(h-2)$ | 定常 $\phi_1+\phi_2<1$、$\phi_2-\phi_1<1$、$\lvert\phi_2\rvert<1$ |
| MA(1) | $X_t=\mu+\varepsilon_t+\theta\varepsilon_{t-1}$ | $\gamma(0)=\sigma^2(1+\theta^2)$ | $\rho(1)=\dfrac{\theta}{1+\theta^2}$、$\rho(h)=0$（$h\ge2$） | 常に定常。反転可能 $\lvert\theta\rvert<1$ |
| ランダムウォーク | $X_t=X_{t-1}+\varepsilon_t$（$X_0=0$） | $V[X_t]=t\sigma^2$ | — | 非定常（単位根） |

- AR($p$) の定常条件：$\phi(z)=1-\phi_1z-\cdots-\phi_pz^p=0$ の根がすべて単位円の外。MA($q$) の反転可能条件も同様。→ 定常条件、反転可能性
- MA(1) の $\lvert\rho(1)\rvert\le1/2$。MA($q$) の自己相関はラグ $q$ で切れ、AR($p$) の偏自己相関はラグ $p$ で切れる。→ 偏自己相関
- ユール・ウォーカー方程式：$\rho(h)=\sum_{j=1}^p\phi_j\rho(h-j)$（$h\ge1$）。AR(1) なら $\hat\phi=\hat\rho(1)$。→ ユール・ウォーカー方程式
- MA の符号は $+\theta$ で書いています（$-\theta$ で書く本では $\theta$ を $-\theta$ に読み替え）。

**AR(1) の予測**：$h$ 期先予測 $\hat X_{T+h}=\mu+\phi^h(X_T-\mu)$、予測誤差の分散 $\sigma^2\dfrac{1-\phi^{2h}}{1-\phi^2}$。→ 予測

## 分割表

2×2 表 $\begin{pmatrix}a&b\\c&d\end{pmatrix}$（行が曝露あり・なし、列がイベントあり・なし）。→ オッズ比の推定と区間推定、相対リスク

| 指標 | 推定値 | 対数（または値）の標準誤差 |
|---|---|---|
| オッズ比 | $\widehat{OR}=\dfrac{ad}{bc}$ | $\mathrm{SE}(\log\widehat{OR})=\sqrt{\dfrac1a+\dfrac1b+\dfrac1c+\dfrac1d}$ |
| 相対リスク | $\widehat{RR}=\dfrac{a/(a+b)}{c/(c+d)}$ | $\mathrm{SE}(\log\widehat{RR})=\sqrt{\dfrac1a-\dfrac1{a+b}+\dfrac1c-\dfrac1{c+d}}$ |
| リスク差 | $\hat p_1-\hat p_2$ | $\sqrt{\dfrac{\hat p_1(1-\hat p_1)}{a+b}+\dfrac{\hat p_2(1-\hat p_2)}{c+d}}$ |

- 信頼区間は $\exp\{\log\widehat{OR}\pm z_{\alpha/2}\mathrm{SE}\}$。ケースコントロール研究ではリスクは推定できないがオッズ比は推定できる。→ コホート研究とケースコントロール研究

**マンテル・ヘンツェル共通オッズ比**（層 $k$ の表 $a_k,b_k,c_k,d_k$、合計 $n_k$）：$\widehat{OR}_{MH}=\dfrac{\sum_k a_kd_k/n_k}{\sum_k b_kc_k/n_k}$。→ マンテル・ヘンツェル共通オッズ比推定量

**連関の強さ**：$\phi=\dfrac{ad-bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}$、$\phi^2=\chi^2/n$。クラメールの $V=\sqrt{\dfrac{\chi^2}{n(\min(r,c)-1)}}$。→ ファイ係数とクラメールの連関係数

**残差**：ピアソン残差 $\dfrac{O_{ij}-E_{ij}}{\sqrt{E_{ij}}}$、調整済み残差 $\dfrac{O_{ij}-E_{ij}}{\sqrt{E_{ij}(1-n_{i\cdot}/n)(1-n_{\cdot j}/n)}}$（近似 $N(0,1)$）。→ 分割表の残差分析

**対数線形モデル**：独立モデル $\log\mu_{ij}=\lambda+\lambda_i^A+\lambda_j^B$、飽和モデルは交互作用項 $\lambda_{ij}^{AB}$ を加える。2×2 では $\lambda_{11}^{AB}$ などの対比が $\log OR$ に対応。→ 対数線形モデル

## ベイズ法

事後分布 $\propto$ 尤度 $\times$ 事前分布。ここでのガンマ分布は辞典のベイズの項目に合わせて**レート表記** $\mathrm{Ga}(\alpha,\beta)$（平均 $\alpha/\beta$）です。→ 共役事前分布、事後分布

| 観測 | 事前分布 | 事後分布 | 事後平均 |
|---|---|---|---|
| 二項：$n$ 回中 $x$ 回成功 | $\mathrm{Be}(a,b)$ | $\mathrm{Be}(a+x,\ b+n-x)$ | $\dfrac{a+x}{a+b+n}$ |
| ポアソン：$n$ 個、合計 $\sum x_i$ | $\mathrm{Ga}(\alpha,\beta)$ | $\mathrm{Ga}(\alpha+\sum x_i,\ \beta+n)$ | $\dfrac{\alpha+\sum x_i}{\beta+n}$ |
| 指数：$n$ 個、合計 $\sum x_i$ | $\mathrm{Ga}(\alpha,\beta)$ | $\mathrm{Ga}(\alpha+n,\ \beta+\sum x_i)$ | $\dfrac{\alpha+n}{\beta+\sum x_i}$ |
| 幾何（失敗回数）：$n$ 個 | $\mathrm{Be}(a,b)$ | $\mathrm{Be}(a+n,\ b+\sum x_i)$ | $\dfrac{a+n}{a+b+n+\sum x_i}$ |
| 正規（$\sigma^2$ 既知）：$n$ 個、平均 $\bar x$ | $N(\mu_0,\tau_0^2)$ | $N(\mu_n,\tau_n^2)$、$\dfrac{1}{\tau_n^2}=\dfrac{1}{\tau_0^2}+\dfrac{n}{\sigma^2}$ | $\mu_n=\tau_n^2\left(\dfrac{\mu_0}{\tau_0^2}+\dfrac{n\bar x}{\sigma^2}\right)$ |

- 事後平均は「事前平均」と「標本平均」の重み付き平均になる（重みは精度＝分散の逆数）。→ 正規・正規モデル、ベータ・二項モデル、ガンマ・ポアソンモデル
- ベータ分布 $\mathrm{Be}(a,b)$ の最頻値（MAP）は $\dfrac{a-1}{a+b-2}$（$a,b>1$）。→ 事後平均とMAP推定
- ガンマ・ポアソンの予測分布は負の二項分布 $\mathrm{NB}\!\left(\alpha',\ \dfrac{\beta'}{\beta'+1}\right)$（$\alpha',\beta'$ は事後のパラメータ）。→ 予測分布

**ジェフリーズの事前分布**：$\pi(\theta)\propto\sqrt{I(\theta)}$。ベルヌーイなら $\mathrm{Be}(\frac12,\frac12)$、ポアソンなら $\lambda^{-1/2}$。→ ジェフリーズの事前分布

**ベイズファクター**：$B_{10}=\dfrac{p(\boldsymbol x\mid M_1)}{p(\boldsymbol x\mid M_0)}$、事後オッズ $=B_{10}\times$ 事前オッズ。→ ベイズファクター

## マルコフ連鎖

推移確率行列 $P=(p_{ij})$。→ 推移確率行列、n段階推移確率、チャップマン・コルモゴロフ方程式

$$
P^{(n)}=P^n,\qquad P^{(m+n)}=P^{(m)}P^{(n)},\qquad \boldsymbol\pi^{(n)}=\boldsymbol\pi^{(0)}P^n
$$

**定常分布**：$\boldsymbol\pi P=\boldsymbol\pi$、$\sum_i\pi_i=1$。既約・非周期・正再帰なら $P^n$ の各行が $\boldsymbol\pi$ に収束する。→ 定常分布、エルゴード性

- 2状態 $P=\begin{pmatrix}1-a&a\\b&1-b\end{pmatrix}$ の定常分布は $\left(\dfrac{b}{a+b},\ \dfrac{a}{a+b}\right)$。
- 平均再帰時間 $m_i=1/\pi_i$。→ 初到達時間・平均再帰時間
- 詳細つり合い $\pi_ip_{ij}=\pi_jp_{ji}$ を満たす $\boldsymbol\pi$ は定常分布。→ 詳細つり合い条件

**吸収確率と平均吸収時間**（一時的な状態 $i$ から）→ 吸収確率・平均吸収時間

$$
h_i=\sum_j p_{ij}h_j\ (\text{吸収先で }h=1\text{ または }0),\qquad t_i=1+\sum_{j\in\text{一時的}}p_{ij}t_j
$$

**破産問題**（所持金 $k$、目標 $N$、1回で $+1$ の確率 $p$、$-1$ の確率 $q$）→ 破産問題

$$
P(\text{先に }N\text{ に達する})=\begin{cases}\dfrac{1-(q/p)^k}{1-(q/p)^N}&(p\ne q)\\[2mm]\dfrac kN&(p=q=\frac12)\end{cases},\qquad p=q\text{ のとき平均時間 }k(N-k)
$$

## 確率過程

**ポアソン過程**（強度 $\lambda$）→ ポアソン過程、到着時間間隔、ポアソン過程の合成と間引き

| 性質 | 式 |
|---|---|
| 件数 | $N(t)\sim\mathrm{Po}(\lambda t)$、増分は独立・定常 |
| 到着間隔 | 独立に $\mathrm{Exp}(\lambda)$（平均 $1/\lambda$） |
| $n$ 番目の到着時刻 | $S_n\sim\mathrm{Ga}(n,1/\lambda)$（形状・尺度）、$E[S_n]=n/\lambda$ |
| 合成・間引き | 独立な強度 $\lambda_1,\lambda_2$ の合成は強度 $\lambda_1+\lambda_2$。確率 $p$ で残す間引きは強度 $\lambda p$ |
| 条件付き | $N(t)=n$ のもとで到着時刻は $U(0,t)$ の $n$ 個の順序統計量 |
| 複合ポアソン | $X(t)=\sum_{i=1}^{N(t)}Y_i$：$E=\lambda tE[Y]$、$V=\lambda tE[Y^2]$ → 複合ポアソン過程 |

**ブラウン運動**：$B(t)\sim N(0,t)$、$\mathrm{Cov}(B(s),B(t))=\min(s,t)$。反射原理 $P\left(\max_{0\le s\le t}B(s)\ge a\right)=2P(B(t)\ge a)$（$a>0$）。→ ブラウン運動、反射原理

**M/M/1 待ち行列**（到着率 $\lambda$、サービス率 $\mu$、$\rho=\lambda/\mu<1$）→ M/M/1待ち行列、リトルの公式

| 量 | 式 |
|---|---|
| 系内人数の分布 | $P(N=n)=(1-\rho)\rho^n$（幾何分布） |
| 平均系内人数 | $L=\dfrac{\rho}{1-\rho}$、待ち行列の長さ $L_q=\dfrac{\rho^2}{1-\rho}$ |
| 平均滞在時間 | $W=\dfrac{1}{\mu-\lambda}$、平均待ち時間 $W_q=\dfrac{\rho}{\mu-\lambda}$ |
| リトルの公式 | $L=\lambda W$、$L_q=\lambda W_q$ |

**出生死滅過程の定常分布**：$\pi_n=\pi_0\prod_{k=1}^n\dfrac{\lambda_{k-1}}{\mu_k}$。→ 出生死滅過程

## 生存・欠測

**生存時間**（→ 生存関数、ハザード関数、カプラン・マイヤー推定量、グリーンウッドの公式）

$$
S(t)=P(T>t),\qquad h(t)=\frac{f(t)}{S(t)},\qquad S(t)=\exp\!\left\{-\int_0^th(u)\,du\right\}
$$

$$
\hat S(t)=\prod_{t_i\le t}\left(1-\frac{d_i}{n_i}\right),\qquad \widehat{V}[\hat S(t)]=\hat S(t)^2\sum_{t_i\le t}\frac{d_i}{n_i(n_i-d_i)}
$$

- $n_i$ は時刻 $t_i$ 直前のリスク集合の大きさ、$d_i$ はイベント数。打ち切りがなければ $\hat S$ は経験生存関数と一致する。
- 比例ハザードモデル $h(t\mid\boldsymbol x)=h_0(t)\exp(\boldsymbol x^\top\boldsymbol\beta)$、$e^{\beta_j}$ がハザード比。→ 比例ハザードモデル

**多重代入のルービンの規則**（$m$ 個の代入、推定値 $\hat Q_l$、分散 $U_l$）→ ルービンの規則

$$
\bar Q=\frac1m\sum_l\hat Q_l,\qquad T=\bar U+\left(1+\frac1m\right)B,\qquad B=\frac{1}{m-1}\sum_l(\hat Q_l-\bar Q)^2
$$

## シミュレーション

| 名前 | 式 | 補足 |
|---|---|---|
| モンテカルロ積分 | $\int h(x)f(x)dx\approx\dfrac1n\sum h(X_i)$、標準誤差 $\sigma_h/\sqrt n$ | 精度は $1/\sqrt n$ でしか良くならない → モンテカルロ積分 |
| 重点サンプリング | $E_f[h(X)]=E_g\!\left[h(X)\dfrac{f(X)}{g(X)}\right]$ | まれな事象の確率を効率よく求める → 重点サンプリング |
| 棄却法 | $f(x)\le cg(x)$ のとき、$g$ から $X$、$U<\dfrac{f(X)}{cg(X)}$ なら採択。採択率 $1/c$ | → 棄却法 |
| メトロポリス・ヘイスティングス | 採択確率 $\min\!\left\{1,\ \dfrac{\pi(y)q(x\mid y)}{\pi(x)q(y\mid x)}\right\}$ | 対称な提案なら $\min\{1,\pi(y)/\pi(x)\}$ → メトロポリス・ヘイスティングス法 |
| 線形合同法 | $x_{n+1}=(ax_n+c)\bmod m$ | 周期は最大 $m$ → 線形合同法 |
| ブートストラップ | 標本から復元抽出で $B$ 回推定し、その標準偏差を標準誤差に | → ブートストラップ法 |
| ジャックナイフ | 偏り $\approx(n-1)(\bar\theta_{(\cdot)}-\hat\theta)$、分散 $\dfrac{n-1}{n}\sum(\hat\theta_{(i)}-\bar\theta_{(\cdot)})^2$ | → ジャックナイフ法 |
