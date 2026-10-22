---
id: gamma-poisson-model
title: ガンマ・ポアソンモデル
reading: がんま・ぽあそんもでる
en: Gamma-Poisson model
aliases: [ポアソン・ガンマモデル, ガンマポアソンモデル]
chapter: 31
importance: A
tags: [共役事前分布, ガンマ分布, ポアソン分布, 件数データ]
related: [gamma-distribution, poisson-distribution, negative-binomial-distribution, conjugate-prior, predictive-distribution, overdispersion]
charts:
  - type: custom
    id: prior-posterior
    params: { model: gamma-poisson, shape: 3, rate: 1, n: 5, sum: 20 }
    sliders:
      - { param: shape, label: "事前の形状 α", min: 0.5, max: 40, step: 0.5 }
      - { param: rate, label: "事前のレート β", min: 0.5, max: 20, step: 0.5 }
      - { param: n, label: "観測数 n", min: 1, max: 60, step: 1 }
      - { param: sum, label: "件数の合計 Σx", min: 0, max: 300, step: 1 }
    caption: 事前 Ga(α, β) とポアソン観測 n 個（合計 Σx）から事後 Ga(α+Σx, β+n) を描く。n と Σx を同じ比で増やすと、事後分布は標本平均 Σx/n の位置に細く集まる。
---

## 一言でいうと

ポアソン分布の平均 $\lambda$ にガンマ分布の事前分布をおき、件数データで更新するモデルです。
事後分布もガンマ分布になり、予測分布は負の二項分布になります。

## 定義・数式

ガンマ分布は**レート表記** $\mathrm{Ga}(\alpha, \beta)$ を使います。

$$
\pi(\lambda) = \frac{\beta^\alpha}{\Gamma(\alpha)}\lambda^{\alpha-1}e^{-\beta\lambda}\quad(\lambda>0), \qquad E[\lambda]=\frac{\alpha}{\beta},\quad V[\lambda]=\frac{\alpha}{\beta^2}
$$

（尺度表記では尺度 $\theta=1/\beta$ を使い、平均 $\alpha\theta$、分散 $\alpha\theta^2$ と書きます。）

モデルは

$$
x_1,\dots,x_n \mid \lambda \overset{\text{iid}}{\sim} \mathrm{Po}(\lambda), \qquad \lambda \sim \mathrm{Ga}(\alpha, \beta)
$$

で、事後分布は

$$
\lambda \mid \boldsymbol{x} \sim \mathrm{Ga}\!\left(\alpha + \sum_{i=1}^n x_i,\ \beta + n\right)
$$

です。事後平均は $\dfrac{\alpha+\sum x_i}{\beta+n}$、事後分散は $\dfrac{\alpha+\sum x_i}{(\beta+n)^2}$、MAP は $\dfrac{\alpha+\sum x_i-1}{\beta+n}$（$\alpha+\sum x_i>1$ のとき）です。

観測期間の長さ $t_i$ が異なり $x_i \sim \mathrm{Po}(\lambda t_i)$ のときは、$\beta + n$ を $\beta + \sum t_i$ に置き換えます。

## 数式の読み解き

**事後分布の導出**：$x_i!$ や $\beta^\alpha/\Gamma(\alpha)$ は $\lambda$ を含まないので捨てます。

$$
\pi(\lambda \mid \boldsymbol{x}) \propto \prod_{i=1}^n \lambda^{x_i} e^{-\lambda} \cdot \lambda^{\alpha-1}e^{-\beta\lambda} = \lambda^{(\alpha+\sum x_i)-1}\,e^{-(\beta+n)\lambda}
$$

**ハイパーパラメータの解釈**：$\beta$ は「事前に観測した期間の数」、$\alpha$ は「その期間に観測した件数の合計」と読めます。データは $\alpha$ に件数を、$\beta$ に期間数を加えます。

**事後平均は重み付き平均**：事前平均 $\alpha/\beta$ と標本平均 $\bar{x}=\sum x_i/n$ を使うと、

$$
\frac{\alpha+n\bar{x}}{\beta+n} = \frac{\beta}{\beta+n}\cdot\frac{\alpha}{\beta} + \frac{n}{\beta+n}\cdot\bar{x}
$$

**予測分布は負の二項分布**：事後分布 $\mathrm{Ga}(\alpha', \beta')$ のもとで、次の1期間の件数 $Y$ は

$$
\begin{aligned}
P(Y=k \mid \boldsymbol{x}) &= \int_0^\infty \frac{\lambda^k e^{-\lambda}}{k!}\cdot\frac{\beta'^{\alpha'}}{\Gamma(\alpha')}\lambda^{\alpha'-1}e^{-\beta'\lambda}\,d\lambda
= \frac{\beta'^{\alpha'}}{k!\,\Gamma(\alpha')}\cdot\frac{\Gamma(\alpha'+k)}{(\beta'+1)^{\alpha'+k}} \\
&= \frac{\Gamma(\alpha'+k)}{k!\,\Gamma(\alpha')}\left(\frac{\beta'}{\beta'+1}\right)^{\alpha'}\left(\frac{1}{\beta'+1}\right)^{k}
\end{aligned}
$$

これは $r=\alpha'$、$p=\beta'/(\beta'+1)$ の負の二項分布（成功 $r$ 回までの失敗回数の分布）です。平均 $\alpha'/\beta'$、分散 $\dfrac{\alpha'}{\beta'}\left(1+\dfrac{1}{\beta'}\right)$ で、ポアソン分布より分散が大きくなります。

## 具体例

**例1：コールセンターの1時間あたりの問い合わせ件数**

過去の経験から平均 3 件程度と考え、事前分布を $\mathrm{Ga}(3, 1)$ とします（平均 3、分散 3）。5 時間の件数が 3, 5, 2, 6, 4（合計 20）でした。

$$
\lambda \mid \boldsymbol{x} \sim \mathrm{Ga}(3+20,\ 1+5) = \mathrm{Ga}(23, 6)
$$

- 事後平均：$23/6 \approx 3.8333$（$\frac{1}{6}\times 3 + \frac{5}{6}\times 4 \approx 3.8333$）
- 事後分散：$23/36 \approx 0.6389$（標準偏差 $\approx 0.7993$）
- MAP：$22/6 \approx 3.6667$
- 95% 等裾信用区間：$[2.4300,\ 5.5514]$
- $P(\lambda > 3 \mid \boldsymbol{x}) \approx 0.8551$（事前には $\approx 0.4232$）

**例2：次の1時間の件数を予測する**

予測分布は $r=23$、$p=6/7$ の負の二項分布です。

| $k$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| $P(Y=k)$ | 0.0289 | 0.0948 | 0.1625 | 0.1935 | 0.1797 | 0.1386 | 0.0924 |

$P(Y=0) = (6/7)^{23} \approx 0.0289$、平均 $\approx 3.8333$、分散 $\frac{23}{6}\times\frac{7}{6} \approx 4.4722$ です。
事後平均 3.8333 を代入したポアソン分布では $P(Y=0) \approx 0.0216$、分散 3.8333 なので、予測分布の方が裾が重くなっています。
6 件以上来る確率は予測分布で $\approx 0.2020$、ポアソン近似では $\approx 0.1894$ です。

## 現場での使われ方

- **製造・品質管理**：設備ごとの月間故障件数をガンマ・ポアソンで推定し、稼働月数が短い設備でも極端な故障率にならないように見積もります。
- **医療**：病院ごとの院内感染件数や有害事象の発生率を推定します。患者日数を期間 $t_i$ として扱います。
- **金融・保険**：契約者の年間事故件数をポアソン、事故率をガンマとし、事故歴に応じて保険料を調整する（経験料率）根拠になります。全体としての件数の分布は負の二項分布になります。
- **IT・Webサービス**：サーバーの障害件数やエラー件数の発生率を逐次更新し、異常の判定に使います。

## グラフ・イメージ

ガンマ事前分布（点線）、ポアソン尤度（細線）、ガンマ事後分布（太線）を重ねています。
観測数 $n$ と合計を増やすと、事後分布は標本平均の位置に細く集まり、事前分布の影響が薄れます。
レート $\beta$ を大きくすると「事前に観測した期間」が長いことになり、事後平均が事前平均 $\alpha/\beta$ に引き寄せられます。

## 試験のポイント

- 事後分布 $\mathrm{Ga}(\alpha+\sum x_i, \beta+n)$ の導出と、事後平均・分散・MAP の計算が典型です。
- レート表記か尺度表記かを問題文の密度で必ず確認します。尺度表記なら事後の尺度は $\theta/(1+n\theta)$ です。
- 予測分布が負の二項分布になる積分は、ガンマ関数の積分 $\int_0^\infty \lambda^{s-1}e^{-c\lambda}d\lambda = \Gamma(s)/c^s$ を使えば一行で済みます。
- ポアソン分布をガンマ分布で混合すると負の二項分布になる事実は、過分散のモデルとしても問われます。

## よくある間違い

- × 事後分布を $\mathrm{Ga}(\alpha+n, \beta+\sum x_i)$ とする → ○ それは指数分布（待ち時間）データの場合です。ポアソン（件数）では $\alpha$ に件数の合計、$\beta$ に観測数を足します。
- × 事後平均を代入したポアソン分布を予測分布とする → ○ $\lambda$ の不確かさを積分すると負の二項分布になり、分散が $\alpha'/\beta'^2$ だけ大きくなります。
- × 尺度表記の $\mathrm{Ga}(\alpha, \theta)$ で「$\theta$ に $n$ を足す」 → ○ 足し算が成り立つのはレート $\beta = 1/\theta$ の方です。
