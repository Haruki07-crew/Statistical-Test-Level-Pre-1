修士1・2年（第45〜57章）の主な公式の早見表です。くわしい説明・導出・例は「→」の後に書いた用語のページを見てください。学部の範囲（第1〜44章）の公式は、2級の辞典と準1級の辞典の「公式まとめ」にあります（ホームの上のリンクから移れます）。
記号の約束：$\xrightarrow{p}$ は確率収束、$\xrightarrow{d}$ は分布収束、$\phi$ は標準正規分布の密度、$e(x)=P(T=1\mid X=x)$ は傾向スコア、$\mu_t(x)=E[Y\mid T=t,X=x]$ です。

## 測度論的確率論

| 内容 | 式・条件 |
|---|---|
| 確率測度の可算加法性 | 互いに素な $A_n$ で $P(\bigcup A_n)=\sum P(A_n)$ |
| 単調収束定理 | $0\le X_n\uparrow X$ なら $E[X_n]\uparrow E[X]$ |
| 優収束定理 | $X_n\to X$（a.s.）、$\lvert X_n\rvert\le Y$、$E[Y]<\infty$ なら $E[X_n]\to E[X]$ |
| 裾の公式 | $X\ge0$ なら $E[X]=\int_0^\infty P(X>t)\,dt$ |
| 測度の取り換え | $Q\ll P$ なら $E_Q[g(X)]=E_P\big[g(X)\,\tfrac{dQ}{dP}(X)\big]$ |
| ボレル・カンテリ | $\sum P(A_n)<\infty\Rightarrow P(A_n\ \text{i.o.})=0$／独立で $\sum P(A_n)=\infty\Rightarrow P(A_n\ \text{i.o.})=1$ |

- 正規分布どうしの尤度比：$P=N(0,1)$、$Q=N(\mu,1)$ なら $dQ/dP(x)=e^{\mu x-\mu^2/2}$。→ ラドン・ニコディムの定理と密度
- 期待値と極限は、優収束定理などの条件がないと交換できない（例：$n\mathbf 1\{U\le1/n\}$）。→ ルベーグ積分と期待値（収束定理）

## 漸近理論

| 内容 | 式 |
|---|---|
| 確率のオーダー | $O_p(a_n)O_p(b_n)=O_p(a_nb_n)$、$O_p(1)\,o_p(1)=o_p(1)$、$\bar X_n-\mu=O_p(n^{-1/2})$ |
| 多変量中心極限定理 | $\sqrt n(\bar X_n-\mu)\xrightarrow{d}N_k(0,\Sigma)$ |
| 多変量デルタ法 | $\sqrt n\{g(\bar X_n)-g(\mu)\}\xrightarrow{d}N(0,\nabla g^\top\Sigma\nabla g)$ |
| DKW の不等式 | $P(\sup_x\lvert F_n(x)-F(x)\rvert>\varepsilon)\le2e^{-2n\varepsilon^2}$ |
| リアプノフ条件 | $\sum_iE\lvert X_{n,i}\rvert^{2+\delta}/s_n^{2+\delta}\to0$ |
| ブラウン橋の共分散 | $\mathrm{Cov}(B(s),B(t))=\min(s,t)-st$（経験過程 $\sqrt n\{F_n(t)-t\}$ の極限） |
| 局所対立仮説での漸近検出力 | $\theta_n=\theta_0+h/\sqrt n$ で $\Phi(-z_{\alpha/2}+h\sqrt I)+\Phi(-z_{\alpha/2}-h\sqrt I)$ |

- 多項分布の共分散：$\mathrm{Cov}(\hat p_j,\hat p_k)=-p_jp_k/n$。同じ調査の 2 つの割合の差の分散は $\{p_1+p_2-(p_1-p_2)^2\}/n$。→ 多変量中心極限定理とクラメール・ウォルドの方法
- DKW から、分布関数の一様な信頼帯 $F_n(x)\pm\sqrt{\log(2/\alpha)/(2n)}$。→ グリベンコ・カンテリの定理
- ブートストラップの片側の被覆率の誤差は、パーセンタイル法で $O(n^{-1/2})$、ブートストラップ t 法で $O(n^{-1})$。最大値では $1-(1-1/n)^n\to1-e^{-1}$ の確率で元の最大値に一致し、失敗する。→ ブートストラップの理論（一致性と高次の精度）

## 推定の理論・決定理論

| 内容 | 式 |
|---|---|
| M 推定量の漸近分散 | $A^{-1}BA^{-\top}$、$A=E[\partial\psi/\partial\theta^\top]$、$B=E[\psi\psi^\top]$（1 次元なら $E[\psi^2]/(E[\psi'])^2$） |
| 中央値の漸近分散 | $1/\{4f(\theta)^2\}$ |
| ホワイト（HC0）の分散 | $(X^\top X)^{-1}\big(\sum e_i^2x_ix_i^\top\big)(X^\top X)^{-1}$（HC3 は $e_i^2/(1-h_{ii})^2$） |
| リスク関数 | $R(\theta,\delta)=E_\theta[L(\theta,\delta(X))]$、ベイズリスク $r(\pi,\delta)=\int R\,d\pi$ |
| 二項分布の推定量 $\tfrac{X+\alpha}{n+\beta}$ のリスク | $\{np(1-p)+(\alpha-\beta p)^2\}/(n+\beta)^2$ |
| 二項分布のミニマックス推定量 | $\tfrac{X+\sqrt n/2}{n+\sqrt n}$、リスク $\tfrac1{4(\sqrt n+1)^2}$ |
| ジェームズ・スタイン推定量 | $\big(1-\tfrac{p-2}{\lVert X\rVert^2}\big)X$、リスク $p-(p-2)^2E[1/\lVert X\rVert^2]$（$\theta=0$ で 2） |

- 一様分布 $U(0,\theta)$ の UMVUE は $\frac{n+1}nX_{(n)}$（分散 $\theta^2/\{n(n+2)\}$）。→ レーマン・シェフェの定理
- 正規分布の分散の推定：$SS/d$ の MSE は $\sigma^4\{2(n-1)+(n-1-d)^2\}/d^2$ で、$d=n+1$ で最小。→ 許容性
- 2 乗損失のベイズ規則は事後平均、絶対損失は事後中央値。→ 損失関数とリスク関数

## ノンパラメトリック推定と頑健統計

| 内容 | 式 |
|---|---|
| カーネル密度推定 | $\hat f_h(x)=\frac1{nh}\sum K\big(\frac{x-X_i}h\big)$ |
| 偏りと分散 | 偏り $\approx\frac{h^2}2\mu_2f''(x)$、分散 $\approx\frac{R(K)f(x)}{nh}$ |
| 最適なバンド幅 | $h^*\propto n^{-1/5}$、MISE $\propto n^{-4/5}$。正規基準 $h=1.06\hat\sigma n^{-1/5}$ |
| ナダラヤ・ワトソンの偏り | $h^2\mu_2\{m''/2+m'f'/f\}$（局所線形は $h^2\mu_2m''/2$） |
| 影響関数 | 平均 $x-\theta$、中央値 $\mathrm{sign}(x-\theta)/\{2f(\theta)\}$、M 推定量 $\psi(x-\theta)/E[\psi']$ |
| フーバーの ψ | $\psi_k(r)=\max(-k,\min(k,r))$、重み $\min(1,k/\lvert r\rvert)$ |

- 漸近分散は影響関数の 2 乗の期待値 $E[\mathrm{IF}^2]$。$k=1.345$ のフーバー推定量は正規分布で効率 95%。→ 影響関数と破綻点

## 混合モデルと相関のあるデータ

| 内容 | 式 |
|---|---|
| 線形混合モデル | $y=X\beta+Zb+\varepsilon$、$V=ZGZ^\top+R$ |
| 級内相関（ランダム切片） | $\rho=\sigma_u^2/(\sigma_u^2+\sigma^2)$ |
| BLUP の縮小係数 | $\lambda_i=n_i\sigma_u^2/(n_i\sigma_u^2+\sigma^2)$ |
| デザイン効果 | $1+(m-1)\rho$ |
| 一元配置の分散成分 | REML $(MS_A-MS_E)/n$、ML $\{(1-1/a)MS_A-MS_E\}/n$ |
| GEE の推定方程式 | $\sum_iD_i^\top V_i^{-1}(y_i-\mu_i)=0$（分散はサンドイッチ型） |
| GLMM の母集団平均の効果（ロジット、近似） | $\beta/\sqrt{1+c^2\sigma^2}$、$c=16\sqrt3/(15\pi)\approx0.588$、潜在変数の級内相関 $\sigma^2/(\sigma^2+\pi^2/3)$ |

- 線形混合モデルの BLUP は $\hat b=GZ^\top V^{-1}(y-X\hat\beta)$。→ 線形混合モデル

## 時系列と状態空間モデル

| 内容 | 式 |
|---|---|
| カルマンフィルタ（予測） | $a_{t\mid t-1}=Ta_{t-1}$、$P_{t\mid t-1}=TP_{t-1}T^\top+Q$ |
| カルマンフィルタ（更新） | $K_t=P_{t\mid t-1}Z^\top F_t^{-1}$、$a_t=a_{t\mid t-1}+K_tv_t$、$P_t=(I-K_tZ)P_{t\mid t-1}$ |
| 対数尤度（予測誤差分解） | $-\frac12\sum_t\{\log\lvert2\pi F_t\rvert+v_t^\top F_t^{-1}v_t\}$ |
| ローカルレベルの定常ゲイン | $K=(-q+\sqrt{q^2+4q})/2$（$q$ は分散の比） |
| GARCH(1,1) | $\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta\sigma_{t-1}^2$、無条件分散 $\omega/(1-\alpha-\beta)$ |
| GARCH の予測・半減期 | $\bar\sigma^2+(\alpha+\beta)^{h-1}(\sigma_{t+1}^2-\bar\sigma^2)$、$\log0.5/\log(\alpha+\beta)$ |
| VAR(1) | 定常 ⇔ $A$ の固有値の絶対値 $<1$、平均 $(I-A)^{-1}c$、インパルス応答 $A^h$ |

- GARCH(1,1) の $r_t^2$ の自己相関は $\rho_k=\rho_1(\alpha+\beta)^{k-1}$。→ ARCH・GARCH モデル

## 高次元統計と学習理論

| 内容 | 式 |
|---|---|
| ヘフディングの不等式 | $P(\lvert\bar X-E\bar X\rvert\ge t)\le2\exp\{-2nt^2/(b-a)^2\}$ |
| 最大値の上界 | $E\max_{j\le p}Z_j\le\sqrt{2\log p}$（準ガウス） |
| 最小二乗の誤差 | $p<n-1$：$\sigma^2p/(n-p-1)$、$p>n+1$（最小ノルム解）：$\approx\lVert\beta\rVert^2(1-n/p)+\sigma^2n/(p-n-1)$ |
| ソフト閾値 | $\mathrm{sign}(z)\max(\lvert z\rvert-\lambda,0)$ |
| ラッソの誤差の速さ | $\lVert\hat\beta-\beta\rVert^2\lesssim\sigma^2s\log p/n$（$\lambda\asymp\sigma\sqrt{\log p/n}$） |
| 超過リスクの分解 | 推定誤差 ＋ 近似誤差 |
| 有限個の関数の一様なずれ | $\sqrt{\{\log\lvert\mathcal F\rvert+\log(2/\delta)\}/(2n)}$ |
| ザウアーの補題 | $S_{\mathcal F}(n)\le\sum_{i=0}^d\binom ni\le(en/d)^d$ |

- ERM の基本の不等式 $R(\hat f)-R(f^*)\le2\sup_f\lvert R_n(f)-R(f)\rvert$。→ 経験リスク最小化と汎化誤差の上界

## ベイズ計算

| 内容 | 式 |
|---|---|
| ELBO | $\log p(x)=\mathrm{ELBO}(q)+\mathrm{KL}(q\Vert p(\theta\mid x))$ |
| 平均場の更新 | $\log q_j^*(\theta_j)=E_{q_{-j}}[\log p(x,\theta)]+\text{定数}$ |
| ハミルトニアン | $H(\theta,p)=U(\theta)+\frac12p^\top M^{-1}p$、採択確率 $\min\{1,e^{H-H^*}\}$ |
| ガウス過程の事後平均・分散 | $k_*^\top(K+\sigma_n^2I)^{-1}y$、$k(x_*,x_*)-k_*^\top(K+\sigma_n^2I)^{-1}k_*$ |
| WAIC | $-2\{\mathrm{lppd}-p_{\mathrm{WAIC}}\}$、$p_{\mathrm{WAIC}}=\sum_iV_{\theta\mid y}[\log p(y_i\mid\theta)]$ |
| ディリクレ過程のクラスターの数 | $E[K_n]=\sum_{i=0}^{n-1}\frac{\alpha}{\alpha+i}=\alpha\{\psi(\alpha+n)-\psi(\alpha)\}$（$\alpha\log n$ 程度で増える）、棒折り $w_j=V_j\prod_{l<j}(1-V_l)$、$V_j\sim\mathrm{Beta}(1,\alpha)$ |

- 相関 $\rho$ の 2 変量正規の事後分布の平均場近似の分散は $1-\rho^2$。→ 変分推論

## 因果推論と予測

| 内容 | 式 |
|---|---|
| 識別（無交絡性・正値性） | $E[Y(t)]=E[\mu_t(X)]$ |
| ネイマンの分散 | $V(\hat\tau)=\frac{S_1^2}{N_1}+\frac{S_0^2}{N_0}-\frac{S_\tau^2}N$（推定は $s_1^2/N_1+s_0^2/N_0$） |
| IPW（ホルビッツ・トンプソン型） | $\frac1n\sum\{T_iY_i/e(X_i)-(1-T_i)Y_i/(1-e(X_i))\}$ |
| AIPW（二重頑健） | $\frac1n\sum\big[\hat\mu_1-\hat\mu_0+\frac{T(Y-\hat\mu_1)}{\hat e}-\frac{(1-T)(Y-\hat\mu_0)}{1-\hat e}\big]$ |
| 差の差 | $(\bar Y_{11}-\bar Y_{10})-(\bar Y_{01}-\bar Y_{00})$ |
| 回帰不連続 | $\lim_{x\downarrow c}E[Y\mid x]-\lim_{x\uparrow c}E[Y\mid x]$ |
| 平均処置効果の効率限界 | $E\big[\frac{\sigma_1^2(X)}{e(X)}+\frac{\sigma_0^2(X)}{1-e(X)}+\{\tau(X)-\tau\}^2\big]$ |
| ダブル機械学習（部分線形） | $\hat\theta=\sum\hat v_i\hat u_i/\sum\hat v_i^2$、$\hat u=Y-\hat\ell(X)$、$\hat v=D-\hat m(X)$ |
| 分割コンフォーマル | $k=\lceil(n+1)(1-\alpha)\rceil$ 番目のスコアを幅に、被覆率 $\in[1-\alpha,\,1-\alpha+\tfrac1{n+1}]$ |

- 標準化平均差 $(\bar x_1-\bar x_0)/\sqrt{(s_1^2+s_0^2)/2}$ が 0.1 未満ならバランスがとれている目安。→ 傾向スコア
- 二重頑健推定量の偏りは $E[(e^\circ-e)(\mu^\circ-\mu)/e]$ で、どちらかのモデルが正しければ 0。→ 二重頑健推定量

## 生存時間解析の理論

| 内容 | 式 |
|---|---|
| 計数過程のマルチンゲール | $M(t)=N(t)-\int_0^tY(s)\lambda(s)\,ds$（独立な打ち切りのもとで平均 0） |
| ネルソン・アーレン推定量 | $\hat\Lambda(t)=\sum_{t_j\le t}d_j/Y(t_j)$、分散の推定 $\sum_{t_j\le t}d_j/Y(t_j)^2$ |
| カプラン・マイヤーとの関係 | $\hat S_{KM}(t)=\prod(1-d_j/Y_j)\le e^{-\hat\Lambda(t)}$ |
| 累積発生関数 | $F_k(t)=\int_0^tS(s^-)\lambda_k(s)\,ds$、ハザード一定なら $\frac{\lambda_k}{\sum_l\lambda_l}\{1-e^{-\sum_l\lambda_lt}\}$ |

- 他の原因を打ち切りとみなした 1 − KM は、原因 $k$ の実際の発生確率を過大に見積もる。→ 競合リスクと累積発生関数
