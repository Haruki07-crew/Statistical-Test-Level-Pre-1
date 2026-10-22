---
id: conjugate-prior
title: 共役事前分布
reading: きょうやくじぜんぶんぷ
en: Conjugate prior
aliases: [自然共役事前分布, 共役分布族]
chapter: 31
importance: A
tags: [ベイズ推論, 事前分布, 指数型分布族]
related: [prior-distribution, posterior-distribution, beta-binomial-model, gamma-poisson-model, normal-normal-model, exponential-family]
charts:
  - type: custom
    id: prior-posterior
    params: { model: gamma-poisson, shape: 3, rate: 1, n: 5, sum: 20 }
    sliders:
      - { param: shape, label: "事前の形状 α", min: 0.5, max: 30, step: 0.5 }
      - { param: rate, label: "事前のレート β", min: 0.5, max: 20, step: 0.5 }
    caption: ガンマ事前分布とポアソン観測（5日で合計20件）。事後分布もガンマ分布 Ga(α+20, β+5) で、事前分布と同じ族にとどまる。
  - type: custom
    id: prior-posterior
    params: { model: beta-binomial, a: 2, b: 2, n: 20, x: 14 }
    sliders:
      - { param: n, label: "試行回数 n", min: 1, max: 100, step: 1 }
      - { param: x, label: "成功回数 x", min: 0, max: 100, step: 1 }
    caption: ベータ事前分布と二項データ。事後分布もベータ分布で、パラメータに成功数と失敗数を足すだけで求まる。
---

## 一言でいうと

尤度と掛け合わせたとき、事後分布が事前分布と**同じ分布族**になるような事前分布です。
事後分布の計算が「ハイパーパラメータの足し算」で済むので、手計算でベイズ推論ができます。

## 定義・数式

データの分布の族 $\{f(x \mid \theta)\}$ と、事前分布の族 $\mathcal{P}=\{\pi(\theta \mid \eta)\}$（$\eta$ はハイパーパラメータ）があるとします。
任意の $\pi \in \mathcal{P}$ と任意のデータ $\boldsymbol{x}$ について

$$
\pi(\theta \mid \boldsymbol{x}) \propto f(\boldsymbol{x} \mid \theta)\,\pi(\theta \mid \eta) \in \mathcal{P}
$$

となるとき、$\mathcal{P}$ を $f$ に対する**共役事前分布族**といいます。事後分布は $\pi(\theta \mid \eta')$ の形で、データによって $\eta \to \eta'$ と更新されます。

代表的な組み合わせ（ガンマ分布はレート表記 $\mathrm{Ga}(\alpha,\beta)$：密度 $\propto \theta^{\alpha-1}e^{-\beta\theta}$、平均 $\alpha/\beta$。尺度表記では尺度 $=1/\beta$）：

| データの分布 | パラメータ | 共役事前分布 | 事後分布（$n$ 個観測） |
|---|---|---|---|
| 二項 $\mathrm{Bin}(m,\theta)$ | 成功確率 $\theta$ | $\mathrm{Be}(a,b)$ | $\mathrm{Be}(a+\sum x_i,\ b+nm-\sum x_i)$ |
| ポアソン $\mathrm{Po}(\lambda)$ | 平均 $\lambda$ | $\mathrm{Ga}(\alpha,\beta)$ | $\mathrm{Ga}(\alpha+\sum x_i,\ \beta+n)$ |
| 指数 $\mathrm{Exp}(\lambda)$ | レート $\lambda$ | $\mathrm{Ga}(\alpha,\beta)$ | $\mathrm{Ga}(\alpha+n,\ \beta+\sum x_i)$ |
| 正規 $N(\mu,\sigma^2)$（$\sigma^2$ 既知） | 平均 $\mu$ | $N(\mu_0,\tau_0^2)$ | $N(\mu_n,\tau_n^2)$（精度が加算） |
| 正規 $N(\mu,\sigma^2)$（$\mu$ 既知） | 精度 $1/\sigma^2$ | $\mathrm{Ga}(\alpha,\beta)$ | $\mathrm{Ga}(\alpha+n/2,\ \beta+\sum(x_i-\mu)^2/2)$ |
| 幾何 $P(X=k)=\theta(1-\theta)^k$ | $\theta$ | $\mathrm{Be}(a,b)$ | $\mathrm{Be}(a+n,\ b+\sum x_i)$ |
| 多項 | 確率ベクトル | ディリクレ | ディリクレ（各カテゴリに度数を加算） |

## 数式の読み解き

**なぜ同じ族に戻るのか**：尤度を $\theta$ の関数として見たときの形と、事前分布の核の形が同じだからです。
ポアソン分布で確かめます。$x_1,\dots,x_n \sim \mathrm{Po}(\lambda)$ 独立、$\lambda \sim \mathrm{Ga}(\alpha,\beta)$ とすると、

$$
\begin{aligned}
\pi(\lambda \mid \boldsymbol{x}) &\propto \prod_{i=1}^{n} \frac{\lambda^{x_i}e^{-\lambda}}{x_i!} \cdot \frac{\beta^\alpha}{\Gamma(\alpha)}\lambda^{\alpha-1}e^{-\beta\lambda} \\
&\propto \lambda^{\sum x_i} e^{-n\lambda} \cdot \lambda^{\alpha-1}e^{-\beta\lambda}
= \lambda^{(\alpha+\sum x_i)-1}\,e^{-(\beta+n)\lambda}
\end{aligned}
$$

で、$\mathrm{Ga}(\alpha+\sum x_i,\ \beta+n)$ の核です。

**一般論（指数型分布族）**：データの密度が $f(x \mid \theta)=h(x)\exp\{\eta(\theta)T(x)-A(\theta)\}$ の形なら、

$$
\pi(\theta) \propto \exp\{\eta(\theta)\,\nu_1 - A(\theta)\,\nu_0\}
$$

が共役事前分布になり、事後分布は $\nu_1 \to \nu_1+\sum T(x_i)$、$\nu_0 \to \nu_0+n$ と更新されます。
ここから、ハイパーパラメータは「**事前の仮想データ**」（仮想的な十分統計量の和 $\nu_1$ と仮想標本サイズ $\nu_0$）と解釈できます。

- ベータ・二項：$a$ = 仮想の成功数、$b$ = 仮想の失敗数。
- ガンマ・ポアソン：$\alpha$ = 仮想の総件数、$\beta$ = 仮想の観測期間数。

## 具体例

**例1：部品の故障率（指数分布とガンマ事前分布）**

部品の寿命（単位：千時間）が指数分布 $\mathrm{Exp}(\lambda)$ に従い、故障率 $\lambda$ の事前分布を $\mathrm{Ga}(2, 10)$（レート表記、平均 0.2）とします。
5 個を試験したところ、寿命の合計が 30 でした。

$$
\pi(\lambda \mid \boldsymbol{x}) \propto \lambda^{5}e^{-30\lambda}\cdot \lambda^{2-1}e^{-10\lambda} = \lambda^{7-1}e^{-40\lambda}
$$

事後分布は $\mathrm{Ga}(7, 40)$ です。

- 事後平均：$7/40 = 0.175$（最尤推定値 $5/30 \approx 0.1667$ と事前平均 0.2 の間）
- 事後最頻値：$(7-1)/40 = 0.15$
- 事後標準偏差：$\sqrt{7}/40 \approx 0.0661$
- 95% 等裾信用区間：$[0.0704,\ 0.3265]$

**例2：ベータ事前分布と二項データ**

$\mathrm{Be}(2,2)$ の事前分布で、20 回中 14 回成功したとき、事後分布は $\mathrm{Be}(16, 8)$、事後平均は $16/24 \approx 0.6667$ です。
計算は「$a$ に成功数、$b$ に失敗数を足す」だけです（詳しくはベータ・二項モデル）。

## 現場での使われ方

- **製造・品質管理**：故障率を指数・ガンマの共役モデルで更新し、試験データが届くたびに信頼性の見積もりを即座に更新します。
- **IT・Webサービス**：多数の広告や記事のクリック率をベータ・二項モデルで管理し、表示のたびに $a, b$ に1を足すだけでオンライン更新します（トンプソン抽出などの基礎）。
- **金融・保険**：保険事故の件数をポアソン、事故率をガンマで表し、契約者ごとの事故率を経験に応じて更新します。

## グラフ・イメージ

上の図はガンマ事前分布とポアソン観測（5日で合計20件）、下の図はベータ事前分布と二項観測です。
どちらも事前分布（点線）と事後分布（太線）が同じ形の分布族で、パラメータだけが変わっていることを確認してください。
下の図で $n$ と $x$ を動かすと、事後分布の山の位置と幅がパラメータの足し算どおりに動きます。

## 試験のポイント

- 表の組み合わせ（ベータ・二項、ガンマ・ポアソン、ガンマ・指数、正規・正規）と更新式は必ず覚えます。
- 導出は「尤度 × 事前の核をまとめる」だけです。途中で定数を捨ててよいことを使います。
- ガンマ分布のレート表記と尺度表記の取り違えが最大の落とし穴です。問題文の密度の式で確認します。
- 事後平均が「事前平均と最尤推定値の重み付き平均」になることは、どの共役モデルでも問われます。
- ハイパーパラメータを仮想データと読む見方を使うと、更新式を忘れても再現できます。

## よくある間違い

- × 共役事前分布は「正しい」事前分布 → ○ 計算の便利さで選ぶもので、事前知識を表すのに十分かは別問題です。
- × ガンマ・ポアソンで $\beta$ に $\sum x_i$ を足す → ○ ポアソンでは形状 $\alpha$ に $\sum x_i$、レート $\beta$ に $n$ を足します。指数分布では逆で、$\alpha$ に $n$、$\beta$ に $\sum x_i$ です。
- × 正規分布の分散も未知なのに、平均の事前分布に正規分布を使えば共役になる → ○ 平均と分散が両方未知なら、正規・逆ガンマ（正規・ガンマ）型の同時事前分布が共役です。
