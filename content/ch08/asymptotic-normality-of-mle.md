---
id: asymptotic-normality-of-mle
title: 最尤推定量の漸近正規性
reading: さいゆうすいていりょうのぜんきんせいきせい
en: Asymptotic normality of the maximum likelihood estimator
aliases: [漸近正規性, 漸近有効性, 漸近分散]
chapter: 8
importance: A
tags: [最尤法, 大標本理論, フィッシャー情報量, 中心極限定理]
related: [maximum-likelihood-estimator, fisher-information, central-limit-theorem, delta-method, wald-confidence-interval, wald-test]
charts:
  - type: custom
    id: clt-sim
    params: { base: bernoulli, n: 20, reps: 3000, p: 0.2 }
    sliders:
      - { param: n, label: "標本サイズ n", min: 1, max: 200, step: 1 }
      - { param: p, label: "成功確率 p", min: 0.05, max: 0.95, step: 0.05 }
    caption: ベルヌーイ分布の最尤推定量（標本比率）を何度も計算したときの分布。n が小さいと飛び飛びで歪んでいるが、n を増やすと平均 p、分散 p(1−p)/n の正規分布に近づく。
---

## 一言でいうと

標本サイズが大きいとき、最尤推定量は「真の値を中心とし、分散がフィッシャー情報量の逆数」の正規分布で近似できる、という性質です。
最尤推定量の標準誤差・信頼区間・ワルド検定は、すべてこの性質の上に成り立っています。

## 定義・数式

$X_1,\dots,X_n$ が独立に $f(x;\theta)$ に従い、正則条件（台が $\theta$ に依存しない、$\log f$ が $\theta$ について3回微分可能で微分と積分を交換できる、$\theta$ が母数空間の内点、$0<I(\theta)<\infty$ など）が成り立つとします。
このとき、最尤推定量 $\hat\theta_n$ は

$$
\sqrt n\,(\hat\theta_n - \theta) \xrightarrow{d} N\Big(0,\ \frac{1}{I(\theta)}\Big)\qquad(n\to\infty)
$$

を満たします。$I(\theta)$ は1個あたりのフィッシャー情報量です。大標本での近似として

$$
\hat\theta_n \overset{\text{近似}}{\sim} N\Big(\theta,\ \frac{1}{nI(\theta)}\Big)
$$

と書けます。漸近分散がクラメール・ラオの下限に一致するので、最尤推定量は**漸近有効**です。
母数がベクトルのときは、分散が情報行列の逆行列 $I(\boldsymbol\theta)^{-1}/n$ になります。

## 数式の読み解き

- $\sqrt n$ を掛けるのは、$\hat\theta_n-\theta$ が $1/\sqrt n$ の速さで 0 に縮むので、その倍率で拡大してちょうどよい大きさの分布を見るためです。
- 分散 $1/\{nI(\theta)\}$ は未知の $\theta$ を含むので、実際には $\theta$ に $\hat\theta$ を代入した $1/\{nI(\hat\theta)\}$ を使います（観測情報量 $-\ell''(\hat\theta)$ を使う方法もあります）。スラツキーの定理により、代入しても漸近分布は変わりません。

**証明の筋道（スコアのテイラー展開）**

尤度方程式 $U(\hat\theta)=0$ の左辺を真の $\theta$ のまわりで1次まで展開します。

$$
0 = U(\hat\theta) \approx U(\theta) + (\hat\theta-\theta)\,U'(\theta)
\quad\Longrightarrow\quad
\sqrt n(\hat\theta-\theta) \approx \frac{U(\theta)/\sqrt n}{-U'(\theta)/n}
$$

- 分子：$U(\theta)$ は平均0・分散 $I(\theta)$ の独立な項 $n$ 個の和なので、中心極限定理により $U(\theta)/\sqrt n\xrightarrow{d}N(0,I(\theta))$。
- 分母：$-U'(\theta)/n$ は $-\frac{\partial^2}{\partial\theta^2}\log f(X_i;\theta)$ の平均なので、大数の法則により $I(\theta)$ に確率収束。

スラツキーの定理から、比は $N(0,I)/I = N(0,1/I)$ に分布収束します。

**各分布での確認**

- ベルヌーイ：$\hat p=\bar X$、漸近分散 $p(1-p)/n$（中心極限定理そのもの）。
- ポアソン：$\hat\lambda=\bar X$、漸近分散 $\lambda/n$。
- 指数：$\hat\lambda=1/\bar X$、$1/\{nI(\lambda)\}=\lambda^2/n$。デルタ法で確かめると、$V[\bar X]=\frac{1}{n\lambda^2}$、$g(\mu)=1/\mu$、$g'(1/\lambda)=-\lambda^2$ より $\lambda^4\cdot\frac{1}{n\lambda^2}=\frac{\lambda^2}{n}$ で一致します。

## 具体例

**例1：故障率の信頼区間（指数分布）**

部品50個の寿命を観測し、平均寿命 $\bar x=2$（千時間）だったとします。故障率の最尤推定値は $\hat\lambda=1/2=0.5$。標準誤差は

$$
\mathrm{SE}(\hat\lambda) = \sqrt{\frac{\hat\lambda^2}{n}} = \frac{0.5}{\sqrt{50}} \approx 0.0707
$$

近似 95% 信頼区間は $0.5\pm1.96\times0.0707$、すなわち $(0.3614,\ 0.6386)$ です。

**例2：支持率（ベルヌーイ）**

200人中70人が支持：$\hat p=0.35$、$\mathrm{SE}=\sqrt{0.35\times0.65/200}\approx0.0337$。近似 95% 信頼区間は $(0.2839,\ 0.4161)$ です。

**例3：関数の推定（デルタ法との組み合わせ）**

例1で平均寿命 $1/\lambda$ を推定するなら、不変性から $1/\hat\lambda=2$、デルタ法から標準誤差 $\frac{1}{\hat\lambda^2}\times0.0707 = 4\times0.0707\approx0.2828$ です（$=\bar x/\sqrt n = 2/\sqrt{50}$ と一致します）。

## 現場での使われ方

- **医療**：ロジスティック回帰や比例ハザードモデルの係数の標準誤差は、情報行列の逆行列から計算されます。論文の「95% 信頼区間」の多くはこの漸近正規近似によるものです。
- **金融・保険**：リスクモデルの母数推定値に誤差の幅を付け、資本の見積もりの不確かさを評価します。
- **IT・Webサービス**：A/B テストでクリック率の差やその対数オッズ比を推定し、漸近正規性に基づく z 検定で判定します。

## グラフ・イメージ

ベルヌーイ分布の最尤推定量（標本比率）をシミュレーションで何度も計算したヒストグラムです。
$n$ が小さく $p$ が 0 や 1 に近いと分布は歪んでいますが、$n$ を大きくすると正規分布の釣鐘形に近づきます。漸近正規性は「$n$ が十分大きいとき」の近似であることがわかります。

## 試験のポイント

- 「最尤推定量を求める → 情報量を計算する → $\hat\theta\pm1.96/\sqrt{nI(\hat\theta)}$ で信頼区間」の一連の流れが典型です。
- 関数 $g(\theta)$ の推定にはデルタ法 $\{g'(\theta)\}^2/\{nI(\theta)\}$ を組み合わせます。
- 漸近分散 ＝ CR 下限なので、最尤推定量は漸近有効です。
- ワルド検定・スコア検定・尤度比検定は、いずれもこの性質から $\chi^2$ 近似を導きます（第10章）。
- 正則条件が崩れる一様分布 $U(0,\theta)$ の $X_{(n)}$ は、正規分布に近づかず、$n(\theta-X_{(n)})$ が指数分布に収束します。

## よくある間違い

- × 漸近分散は $1/I(\theta)$ → ○ $\hat\theta$ そのものの近似分散は $1/\{nI(\theta)\}$ です。$1/I(\theta)$ は $\sqrt n(\hat\theta-\theta)$ の分散です。
- × 漸近正規性があるので小標本でも正規近似でよい → ○ 小標本や母数が境界に近いとき（$p\approx0$ など）は近似が悪く、正確な方法や変換を検討します。
- × どの推定量も漸近分散が CR 下限に一致する → ○ 一致するのは最尤推定量などの漸近有効な推定量です。モーメント推定量は一般に大きくなります。
