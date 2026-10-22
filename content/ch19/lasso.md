---
id: lasso
title: Lasso
reading: らっそ
en: Lasso (least absolute shrinkage and selection operator)
aliases: [ラッソ, L1正則化, Lasso回帰, ソフト閾値]
chapter: 19
importance: A
tags: [正則化, 罰則付き推定, 変数選択, スパース推定]
related: [ridge-regression, regularization, elastic-net, variable-selection, bias-in-penalized-estimation, cross-validation]
charts:
  - type: custom
    id: ridge-lasso-path
    params: { method: lasso }
    caption: 4変数のデータで λ を大きくしたときの Lasso 推定値の推移。係数は折れ線的に縮み、ある λ でちょうど0に張りついて、それ以降は0のままになる。
  - type: custom
    id: line-data
    params:
      x: [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
      series:
        - { label: "OLS（縮小なし）", y: [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3] }
        - { label: "Lasso（ソフト閾値, λ=1）", y: [-2, -1.5, -1, -0.5, 0, 0, 0, 0, 0, 0.5, 1, 1.5, 2] }
        - { label: "リッジ（λ=1）", y: [-1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5] }
      xlabel: "OLS 推定値 b_j"
      ylabel: "縮小後の推定値"
    caption: 直交計画での縮小の仕方。Lasso は |b_j|≤λ を0にし、それより外側は一律に λ だけ0へ寄せる。リッジは全体を同じ割合で縮める。
---

## 一言でいうと

残差平方和に「係数の絶対値の和」の罰則を足して推定する回帰です。
罰則の形のおかげで、効きの弱い変数の係数がちょうど0になり、推定と変数選択を同時に行えます。

## 定義・数式

説明変数を標準化し、目的変数を中心化した線形モデルで、**Lasso 推定量**は

$$
\hat{\boldsymbol{\beta}}_L(\lambda)=\arg\min_{\boldsymbol{\beta}}\left\{\frac{1}{2}\|\boldsymbol{y}-X\boldsymbol{\beta}\|^2+\lambda\sum_{j=1}^{p}|\beta_j|\right\}\qquad(\lambda\ge0)
$$

です（切片は罰則に含めない）。制約 $\sum_j|\beta_j|\le t$ のもとで残差平方和を最小にする問題と同値です。

罰則項は $\beta_j=0$ で微分できないので、一般には閉じた形の解はなく、座標降下法や LARS などで数値的に解きます。
ただし**直交計画** $X^\top X=I$ のときは、OLS 推定値 $b_j=\boldsymbol{x}_j^\top\boldsymbol{y}$ を使って

$$
\hat\beta_{L,j}=\mathrm{sign}(b_j)\,(|b_j|-\lambda)_+=
\begin{cases}
b_j-\lambda & (b_j>\lambda)\\
0 & (|b_j|\le\lambda)\\
b_j+\lambda & (b_j<-\lambda)
\end{cases}
$$

と書けます。これを**ソフト閾値関数**といいます。$(a)_+=\max(a,0)$ です。

目的関数の係数の付け方には流儀があり、$\frac{1}{2n}\|\boldsymbol{y}-X\boldsymbol{\beta}\|^2+\lambda\|\boldsymbol{\beta}\|_1$ と書くソフトウェアもあります。その場合は同じ解に対応する $\lambda$ の値が $n$ 倍ずれます。

## 数式の読み解き

**ソフト閾値の導出（直交計画）**：$X^\top X=I$ なら

$$
\frac{1}{2}\|\boldsymbol{y}-X\boldsymbol{\beta}\|^2=\frac{1}{2}\|\boldsymbol{y}\|^2-\boldsymbol{\beta}^\top\boldsymbol{b}+\frac{1}{2}\|\boldsymbol{\beta}\|^2
=\text{定数}+\sum_{j}\left\{\frac{1}{2}(\beta_j-b_j)^2\right\}
$$

（$\boldsymbol{b}=X^\top\boldsymbol{y}$、定数は $\boldsymbol{\beta}$ によらない部分）なので、問題は係数ごとの1変数問題に分かれます。

$$
g(\beta)=\frac{1}{2}(\beta-b)^2+\lambda|\beta|
$$

- $\beta>0$ の範囲：$g'(\beta)=\beta-b+\lambda=0$ より $\beta=b-\lambda$。これが正であるには $b>\lambda$。
- $\beta<0$ の範囲：$g'(\beta)=\beta-b-\lambda=0$ より $\beta=b+\lambda$。これが負であるには $b<-\lambda$。
- どちらでもない $|b|\le\lambda$ のとき：$\beta=0$ での劣微分 $\{-b+\lambda s: s\in[-1,1]\}$ が0を含むので、$\beta=0$ が最小点です。

**なぜ0になるのか**：リッジの罰則 $\lambda\beta^2$ は0の近くで傾きが0なので、少しでもデータが係数を引っ張れば0から離れます。
絶対値の罰則は0でも傾き $\lambda$ をもつため、データの引っ張り $|b|$ が $\lambda$ を超えない限り0に留まります。
図形で言えば、制約領域 $\sum|\beta_j|\le t$ はひし形（高次元では角のある多面体）で、残差平方和の等高線が角（座標軸上）で接しやすいためです。

**偏り**：生き残った係数も一律に $\lambda$ だけ0へ寄せられるので、大きな係数ほど相対的な偏りは小さいものの、偏りは残ります。

## 具体例

**例1：直交計画でのソフト閾値**

$n=4$ で、説明変数の列が直交するように $x_1=\tfrac12(1,1,-1,-1)$、$x_2=\tfrac12(1,-1,1,-1)$、$x_3=\tfrac12(1,-1,-1,1)$ とし、$y=(3.0,\ 0.6,\ -0.4,\ -2.2)$ とします。$X^\top X=I$ で、切片は罰則なしに $\bar{y}=0.25$ と推定され、傾きには影響しません。

$$
\boldsymbol{b}=X^\top\boldsymbol{y}=\left(\tfrac{3.0+0.6+0.4+2.2}{2},\ \tfrac{3.0-0.6-0.4+2.2}{2},\ \tfrac{3.0-0.6+0.4-2.2}{2}\right)=(3.1,\ 2.1,\ 0.3)
$$

$\lambda=1$ では

$$
\hat{\boldsymbol{\beta}}_L=(3.1-1,\ 2.1-1,\ 0)=(2.1,\ 1.1,\ 0)
$$

で、$x_3$ が除かれます。同じ $\lambda=1$ のリッジは $\boldsymbol{b}/2=(1.55,\ 1.05,\ 0.15)$ で、0にはなりません。$\lambda=2$ の Lasso は $(1.1,\ 0.1,\ 0)$ です。座標降下法で数値的に解いても同じ値になります。

**例2：相関の強い2変数では片方が先に落ちる**

リッジ回帰の項の例と同じデータ（$X^\top X$ の対角10・非対角9、$X^\top\boldsymbol{y}=(18.6,\ 18.0)$、OLS $=(1.2632,\ 0.6632)$）に、上の定義の Lasso を当てはめます。
両方の係数が正のまま残る範囲では、最適性の条件 $X^\top X\boldsymbol{\beta}=X^\top\boldsymbol{y}-\lambda(1,1)^\top$ より

$$
\hat{\boldsymbol{\beta}}_L=\hat{\boldsymbol{\beta}}_{OLS}-\lambda(X^\top X)^{-1}\begin{pmatrix}1\\1\end{pmatrix}=\begin{pmatrix}1.2632-\lambda/19\\ 0.6632-\lambda/19\end{pmatrix}
$$

$\hat\beta_2$ は $\lambda=0.6632\times19=12.6$ で0になります。それ以降は $x_1$ だけで $\hat\beta_1=(18.6-\lambda)/10$ となり、$\lambda=18.6$ ですべて0です。
たとえば $\lambda=10$ では $(0.7368,\ 0.1368)$ です。相関の強い変数の組からは、片方だけが残りやすいことがわかります。

## 現場での使われ方

- **医療**：数千の遺伝子マーカーの中から、病気の予測に効く少数のマーカーを選びます。
- **マーケティング**：購買履歴から作った多数の特徴量のうち、離反の予測に効くものだけを残し、説明しやすいモデルにします。
- **金融・保険**：信用スコアのモデルで、説明変数を絞り込んで運用・監査しやすくします。
- **IT・Webサービス**：高次元の特徴量をもつ予測モデルで、交差検証で $\lambda$ を選び、疎な（0の多い）モデルを作ります。

## グラフ・イメージ

上の図は解の経路で、$\lambda$ を大きくすると係数が直線的に縮み、ある所で0に張りついて消えていきます。
下の図は直交計画での縮小の仕方で、Lasso は $|b_j|\le1$ の範囲を0にし、外側は1だけ平行移動します。リッジは原点を通る直線で全体を半分にします。

## 試験のポイント

- 直交計画でのソフト閾値 $\mathrm{sign}(b)(|b|-\lambda)_+$ を、場合分けで導けるようにします。リッジの $b/(1+\lambda)$ と並べて比較させる問題が典型です。
- 「Lasso は係数がちょうど0になる（変数選択）、リッジはならない」という違いを、罰則の形（0で尖っているか）で説明できるようにします。
- 目的関数の $\frac12$ や $\frac{1}{2n}$ の付け方で、閾値に当たる量が変わる点に注意します。
- ベイズ的には、各係数にラプラス分布（両側指数分布）の事前分布を置いたときの事後モード（MAP 推定値）です。
- 相関の強い変数群からは1つだけ選びやすい、$p>n$ では選ばれる変数が高々 $n$ 個、という弱点を補うのがエラスティックネットです。

## よくある間違い

- × 直交計画の Lasso は OLS を $1/(1+\lambda)$ 倍する → ○ それはリッジです。Lasso は $\lambda$ を引いて、符号が変わるなら0にします。
- × Lasso で0にならなかった係数は不偏 → ○ 残った係数も $\lambda$ だけ0側へ縮められていて、偏りがあります。
- × 標準化しなくてよい → ○ 単位を変えると罰則の効き方が変わり、選ばれる変数も変わります。
- × Lasso に閉じた形の解は一般にある → ○ 閉じた形で書けるのは直交計画など特別な場合だけです。
