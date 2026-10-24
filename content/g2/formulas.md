2級でよく使う公式を分野ごとにまとめた早見表です。くわしい説明や例は「→」の後に書いた用語のページを見てください。
記号の約束：$z_\alpha$、$t_\alpha(k)$、$\chi^2_\alpha(k)$、$F_\alpha(m,n)$ はそれぞれの分布の**上側 $\alpha$ 点**、$q=1-p$ です。データの分散は、ことわりがなければ $n-1$ で割ったもの（不偏分散）です。

## データの要約

| 指標 | 式 | メモ |
|---|---|---|
| 平均値 | $\bar x=\dfrac{1}{n}\sum x_i$ | 度数分布表では $\dfrac{1}{n}\sum m_i f_i$（$m_i$ は階級値） |
| 分散・標準偏差 | $s^2=\dfrac{1}{n-1}\sum (x_i-\bar x)^2$、$s=\sqrt{s^2}$ | $\sum(x_i-\bar x)^2=\sum x_i^2-n\bar x^{\,2}$ |
| 四分位範囲 | $Q_3-Q_1$ | 外れ値の目安：$Q_1-1.5(Q_3-Q_1)$ 未満、$Q_3+1.5(Q_3-Q_1)$ 超 |
| 変動係数 | $s/\bar x$ | 平均の違う量のばらつきの比較 |
| 標準得点・偏差値 | $z=\dfrac{x-\bar x}{s}$、偏差値 $=50+10z$ | 標準化したデータは平均 0・標準偏差 1 |
| 幾何平均 | $\sqrt[n]{x_1x_2\cdots x_n}$ | 比・成長率の平均 |
| 調和平均 | $\dfrac{n}{1/x_1+\cdots+1/x_n}$ | 同じ距離を違う速さで進んだときの平均の速さ |
| ジニ係数 | $1-2\times(\text{ローレンツ曲線の下の面積})$ | 面積は台形の和。完全平等で 0 |

- $y=a+bx$ と変換すると、平均は $a+b\bar x$、分散は $b^2s^2$、標準偏差は $|b|s$。→ データの分散と標準偏差
- 2 つの群を合わせた分散（$n$ で割る形）は「群の中の分散の加重平均 ＋ 群の平均のずれの加重平均」。→ 平均値・中央値・最頻値

## 2 変数のデータと単回帰

| 指標 | 式 |
|---|---|
| 共分散 | $s_{xy}=\dfrac{1}{n-1}\sum (x_i-\bar x)(y_i-\bar y)$ |
| 相関係数 | $r=\dfrac{s_{xy}}{s_x s_y}=\dfrac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}$ |
| 回帰直線の傾き・切片 | $b=\dfrac{S_{xy}}{S_{xx}}=r\dfrac{s_y}{s_x}$、$a=\bar y-b\bar x$ |
| 残差 | $e_i=y_i-\hat y_i$（合計は 0） |
| 決定係数（単回帰） | $R^2=1-\dfrac{\sum e_i^2}{S_{yy}}=r^2$ |
| 偏相関係数 | $r_{xy\cdot z}=\dfrac{r_{xy}-r_{xz}r_{yz}}{\sqrt{(1-r_{xz}^2)(1-r_{yz}^2)}}$ |

- $S_{xx}=\sum(x_i-\bar x)^2$、$S_{xy}=\sum(x_i-\bar x)(y_i-\bar y)$。回帰直線は $(\bar x,\bar y)$ を通る。→ 単回帰と回帰直線
- $x$ を $c$ 倍すると、相関係数は変わらず、傾きは $1/c$ 倍。→ 共分散と相関係数

## 時系列と指数

| 指標 | 式 |
|---|---|
| 成長率 | $g_t=\dfrac{y_t}{y_{t-1}}-1$ |
| 平均成長率（$n$ 期間） | $\sqrt[n]{y_n/y_0}-1$ |
| 寄与度 | $\dfrac{\text{部分の増加分}}{\text{全体の前期の値}}$（合計は全体の成長率） |
| 指数（基準時点 0） | $\dfrac{y_t}{y_0}\times 100$ |
| ラスパイレス価格指数 | $\dfrac{\sum p_1q_0}{\sum p_0q_0}\times 100$ |
| パーシェ価格指数 | $\dfrac{\sum p_1q_1}{\sum p_0q_1}\times 100$ |
| 実質値 | 名目値 ÷（物価指数 / 100） |
| 4 項中心化移動平均 | $\dfrac14\left(\dfrac12y_{t-2}+y_{t-1}+y_t+y_{t+1}+\dfrac12y_{t+2}\right)$ |

- 指数の差は「ポイント」、比の変化は「%」。→ 指数化と指数、成長率と平均成長率、物価指数（ラスパイレス・パーシェ）、移動平均

## 確率

| 公式 | 式 |
|---|---|
| 加法定理 | $P(A\cup B)=P(A)+P(B)-P(A\cap B)$ |
| 余事象 | $P(A^c)=1-P(A)$ |
| 条件付き確率・乗法定理 | $P(A\mid B)=\dfrac{P(A\cap B)}{P(B)}$、$P(A\cap B)=P(B)P(A\mid B)$ |
| 独立 | $P(A\cap B)=P(A)P(B)$ |
| 全確率の公式 | $P(B)=\sum_i P(A_i)P(B\mid A_i)$ |
| ベイズの定理 | $P(A_i\mid B)=\dfrac{P(A_i)P(B\mid A_i)}{\sum_j P(A_j)P(B\mid A_j)}$ |
| 順列・組合せ | ${}_n\mathrm{P}_r=\dfrac{n!}{(n-r)!}$、$\binom{n}{r}=\dfrac{n!}{r!(n-r)!}$ |

→ 事象と確率（加法定理）、条件付き確率と乗法定理、ベイズの定理、順列と組合せ

## 期待値と分散

- $E[aX+b]=aE[X]+b$、$V[aX+b]=a^2V[X]$、$V[X]=E[X^2]-(E[X])^2$
- $E[X+Y]=E[X]+E[Y]$、$V[X\pm Y]=V[X]+V[Y]\pm 2\,\mathrm{Cov}[X,Y]$（独立なら共分散は 0）
- 相関係数 $\rho=\dfrac{\mathrm{Cov}[X,Y]}{\sqrt{V[X]V[Y]}}$
- 独立な正規分布の和は正規分布：$X+Y\sim N(\mu_X+\mu_Y,\ \sigma_X^2+\sigma_Y^2)$

→ 期待値と分散の性質、確率変数の和と再生性

## 主な確率分布

| 分布 | 確率（密度）関数 | 期待値 | 分散 |
|---|---|---|---|
| 二項 $B(n,p)$ | $\binom{n}{x}p^xq^{n-x}$ | $np$ | $npq$ |
| ポアソン $Po(\lambda)$ | $\dfrac{\lambda^xe^{-\lambda}}{x!}$ | $\lambda$ | $\lambda$ |
| 幾何（試行の回数 $x=1,2,\dots$） | $q^{x-1}p$ | $\dfrac1p$ | $\dfrac{q}{p^2}$ |
| 超幾何（$N$ 個中 $K$ 個、$n$ 個取る） | $\dfrac{\binom{K}{x}\binom{N-K}{n-x}}{\binom{N}{n}}$ | $n\dfrac{K}{N}$ | $n\dfrac{K}{N}\left(1-\dfrac{K}{N}\right)\dfrac{N-n}{N-1}$ |
| 一様 $U(a,b)$ | $\dfrac{1}{b-a}$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ |
| 指数（レート $\lambda$） | $\lambda e^{-\lambda x}$ | $\dfrac1\lambda$ | $\dfrac1{\lambda^2}$ |
| 正規 $N(\mu,\sigma^2)$ | $\dfrac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ |

- 指数分布：$P(X>t)=e^{-\lambda t}$。平均 $\theta$ で書くときは $\lambda=1/\theta$。
- 幾何分布は「失敗の回数（0 から）」で数える流儀もあり、そのときの期待値は $q/p$（辞書の 幾何分布と無記憶性 はこちらの流儀）。

## 標本分布

| 内容 | 式 |
|---|---|
| 標本平均 | $E[\bar X]=\mu$、$V[\bar X]=\dfrac{\sigma^2}{n}$、正規母集団なら $\bar X\sim N(\mu,\sigma^2/n)$ |
| 中心極限定理 | $n$ が大きいとき $\dfrac{\bar X-\mu}{\sigma/\sqrt n}$ は近似的に $N(0,1)$ |
| 二項分布の正規近似 | $X\sim B(n,p)$ は近似的に $N(np,\ npq)$。「$X\ge k$」は連続修正で $k-0.5$ から |
| 標本比率 | $E[\hat p]=p$、$V[\hat p]=\dfrac{pq}{n}$ |
| チェビシェフの不等式 | $P(\lvert X-\mu\rvert\ge k\sigma)\le\dfrac{1}{k^2}$ |
| カイ二乗分布 | $Z_1^2+\cdots+Z_k^2\sim\chi^2(k)$、期待値 $k$、分散 $2k$。$\dfrac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)$ |
| t 分布 | $\dfrac{\bar X-\mu}{S/\sqrt n}\sim t(n-1)$ |
| F 分布 | $\dfrac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}\sim F(n_1-1,\ n_2-1)$、$F_{1-\alpha}(m,n)=\dfrac{1}{F_\alpha(n,m)}$ |

→ 標本平均の分布、中心極限定理、分布表の読み方（上側確率点）

## 区間推定（信頼係数 95%）

| 対象 | 区間 |
|---|---|
| 母平均（分散既知） | $\bar x\pm z_{0.025}\dfrac{\sigma}{\sqrt n}$ |
| 母平均（分散未知） | $\bar x\pm t_{0.025}(n-1)\dfrac{s}{\sqrt n}$ |
| 母分散 | $\left(\dfrac{(n-1)s^2}{\chi^2_{0.025}(n-1)},\ \dfrac{(n-1)s^2}{\chi^2_{0.975}(n-1)}\right)$ |
| 母比率 | $\hat p\pm z_{0.025}\sqrt{\dfrac{\hat p(1-\hat p)}{n}}$ |
| 母平均の差（大標本） | $(\bar x_1-\bar x_2)\pm z_{0.025}\sqrt{\dfrac{s_1^2}{n_1}+\dfrac{s_2^2}{n_2}}$ |
| 母比率の差 | $(\hat p_1-\hat p_2)\pm z_{0.025}\sqrt{\dfrac{\hat p_1(1-\hat p_1)}{n_1}+\dfrac{\hat p_2(1-\hat p_2)}{n_2}}$ |
| 母相関係数 | $z=\tanh^{-1}r$ として $z\pm\dfrac{z_{0.025}}{\sqrt{n-3}}$ を求め、$\tanh$ で戻す |
| 必要な標本の大きさ（比率） | 誤差の範囲を $d$ 以下にするには $n\ge\left(\dfrac{z_{0.025}}{d}\right)^2p(1-p)$（見当がなければ $p=0.5$） |

→ 信頼区間、母平均の区間推定（分散未知）、母比率の区間推定、母分散の区間推定、必要な標本サイズ

## 検定統計量

| 検定 | 統計量 | 帰無仮説のもとの分布 |
|---|---|---|
| 母平均（分散既知） | $\dfrac{\bar x-\mu_0}{\sigma/\sqrt n}$ | $N(0,1)$ |
| 母平均（分散未知） | $\dfrac{\bar x-\mu_0}{s/\sqrt n}$ | $t(n-1)$ |
| 母分散 | $\dfrac{(n-1)s^2}{\sigma_0^2}$ | $\chi^2(n-1)$ |
| 母比率 | $\dfrac{\hat p-p_0}{\sqrt{p_0(1-p_0)/n}}$ | 近似的に $N(0,1)$ |
| 2 標本（等分散） | $\dfrac{\bar x_1-\bar x_2}{\sqrt{s_p^2(1/n_1+1/n_2)}}$、$s_p^2=\dfrac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}$ | $t(n_1+n_2-2)$ |
| ウェルチ | $\dfrac{\bar x_1-\bar x_2}{\sqrt{s_1^2/n_1+s_2^2/n_2}}$ | 近似的に t 分布 |
| 対応のある差 | $\dfrac{\bar d}{s_d/\sqrt n}$ | $t(n-1)$ |
| 等分散 | $\dfrac{s_1^2}{s_2^2}$ | $F(n_1-1,\ n_2-1)$ |
| 比率の差 | $\dfrac{\hat p_1-\hat p_2}{\sqrt{\hat p(1-\hat p)(1/n_1+1/n_2)}}$（$\hat p$ は合わせた比率） | 近似的に $N(0,1)$ |
| 適合度 | $\sum\dfrac{(O-E)^2}{E}$ | $\chi^2(\text{カテゴリ数}-1-\text{推定した母数の数})$ |
| 独立性（$r\times c$ 表） | $\sum\dfrac{(O-E)^2}{E}$、$E=\dfrac{\text{行の計}\times\text{列の計}}{n}$ | $\chi^2((r-1)(c-1))$ |

- 両側検定は上側 $\alpha/2$ 点、片側検定は上側 $\alpha$ 点と比べる。→ 片側検定と両側検定、p値

## 回帰分析と分散分析

| 内容 | 式 |
|---|---|
| 係数の t 値 | 推定値 ÷ 標準誤差（自由度 $n-p-1$） |
| 係数の 95% 区間 | 推定値 $\pm\,t_{0.025}(n-p-1)\times$ 標準誤差 |
| 自由度調整済み決定係数 | $1-(1-R^2)\dfrac{n-1}{n-p-1}$ |
| 回帰全体の F | $\dfrac{R^2/p}{(1-R^2)/(n-p-1)}\sim F(p,\ n-p-1)$ |
| 一元配置（$k$ 群・$N$ 個） | $S_T=S_A+S_E$、自由度 $N-1=(k-1)+(N-k)$、$F=\dfrac{S_A/(k-1)}{S_E/(N-k)}$ |

（$p$ は切片を除く説明変数の数）→ 回帰分析の出力の読み方、自由度調整済み決定係数、分散分析表の読み方
