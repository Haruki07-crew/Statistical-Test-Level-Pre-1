---
id: ma-model
title: MAモデル
reading: えむえーもでる
en: Moving average model
aliases: [移動平均モデル, MA過程, "MA(q)", "MA(1)", 移動平均過程]
chapter: 27
importance: A
tags: [時系列解析, MAモデル, 自己共分散, 反転可能性]
related: [white-noise, invertibility, ar-model, arma-model, autocovariance-autocorrelation, correlogram]
charts:
  - type: custom
    id: acf-plot
    params: { model: ma1, theta: 0.6, n: 300, lags: 15, pacf: 1 }
    sliders:
      - { param: theta, label: "θ", min: -0.95, max: 0.95, step: 0.05 }
    caption: MA(1) 過程 X_t = ε_t + θε_{t-1}。自己相関（中段）はラグ1だけが0でなく、理論値 θ/(1+θ²) は |θ| を大きくしても 0.5 を超えない。偏自己相関（下段）は減衰する。
---

## 一言でいうと

今の値を、現在と過去 $q$ 期分の雑音（ショック）の重み付き和で表すモデルです。
ショックの影響が $q$ 期でぴたりと消えるので、自己相関がラグ $q$ の後で0になります。

## 定義・数式

$\varepsilon_t\sim WN(0,\sigma^2)$ として、

$$
X_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\cdots+\theta_q\varepsilon_{t-q}
$$

を次数 $q$ の**移動平均モデル** MA($q$) といいます。後退作用素で書くと $X_t-\mu=\theta(B)\varepsilon_t$、$\theta(z)=1+\theta_1z+\cdots+\theta_qz^q$ です。

> 符号の流儀：本サイトは $+\theta$ で書きます。$X_t=\varepsilon_t-\theta_1\varepsilon_{t-1}-\cdots$ と $-\theta$ で書く教科書もあり、その場合は以下の式で $\theta$ を $-\theta$ に読み替えます。

$\theta_0=1$ とおくと、自己共分散は

$$
\gamma(h)=\begin{cases}\sigma^2\displaystyle\sum_{j=0}^{q-|h|}\theta_j\theta_{j+|h|} & (|h|\le q)\\[2mm] 0 & (|h|>q)\end{cases}
$$

です。MA($q$) は係数によらず**常に弱定常**です。

**MA(1) のまとめ** $X_t=\mu+\varepsilon_t+\theta\varepsilon_{t-1}$

| 項目 | 値 |
|---|---|
| 平均 | $\mu$ |
| 分散 $\gamma(0)$ | $(1+\theta^2)\sigma^2$ |
| $\gamma(1)$ | $\theta\sigma^2$ |
| 自己相関 | $\rho(1)=\dfrac{\theta}{1+\theta^2}$、$\lvert h\rvert\ge2$ で0 |
| 反転可能条件 | $\lvert\theta\rvert<1$ |
| 偏自己相関 | $\phi_{kk}=-\dfrac{(-\theta)^k(1-\theta^2)}{1-\theta^{2(k+1)}}$（減衰） |

## 数式の読み解き

**MA(1) の導出**（$\mu=0$ としてよい）

$$
\gamma(0)=V[\varepsilon_t+\theta\varepsilon_{t-1}]=\sigma^2+\theta^2\sigma^2=(1+\theta^2)\sigma^2
$$

$$
\gamma(1)=\operatorname{Cov}(\varepsilon_t+\theta\varepsilon_{t-1},\ \varepsilon_{t+1}+\theta\varepsilon_t)=\theta\,V[\varepsilon_t]=\theta\sigma^2
$$

共通に含まれる $\varepsilon$ は $\varepsilon_t$ だけなので、その係数の積 $1\times\theta$ が残ります。
$h\ge2$ では共通の $\varepsilon$ がないので $\gamma(h)=0$ です。一般の MA($q$) でも「同じ $\varepsilon$ の係数どうしを掛けて足す」と覚えれば上の公式になります。

- 平均・分散・自己共分散のどれにも $t$ が入らないので、どんな $\theta$ でも弱定常です（AR と違い定常条件は不要）。
- $\rho(1)=\theta/(1+\theta^2)$ は $\theta=\pm1$ で最大の $\pm0.5$ をとります。$(1-\theta)^2\ge0$ から $1+\theta^2\ge2\theta$ となるためです。標本で $|r_1|>0.5$ なら MA(1) は不適切です。
- $\theta$ と $1/\theta$ は同じ自己相関を与えます（$\dfrac{1/\theta}{1+1/\theta^2}=\dfrac{\theta}{\theta^2+1}$）。どちらを選ぶかを決めるのが**反転可能性** $|\theta|<1$ です。
- 反転可能なら $\varepsilon_t=\sum_{j\ge0}(-\theta)^jX_{t-j}$ と AR(∞) で表せるので、偏自己相関は切断せずに減衰します。

## 具体例

**例1：MA(1) の自己相関**

$X_t=10+\varepsilon_t+0.6\varepsilon_{t-1}$、$\sigma^2=1$ とします。

$$
E[X_t]=10,\quad \gamma(0)=1.36,\quad \gamma(1)=0.6,\quad \rho(1)=\frac{0.6}{1.36}\approx0.4412
$$

$\theta=1/0.6\approx1.6667$ でも $\rho(1)=0.4412$ になりますが、反転可能なのは $\theta=0.6$ のほうです。

**例2：自己相関から係数を逆算する（モーメント法）**

MA(1) で $\rho(1)=0.4$ とわかったとき、$\theta/(1+\theta^2)=0.4$ を整理すると $0.4\theta^2-\theta+0.4=0$ で、

$$
\theta=\frac{1\pm\sqrt{1-4\times0.4^2}}{2\times0.4}=\frac{1\pm0.6}{0.8}=0.5\ \text{または}\ 2
$$

反転可能な $\theta=0.5$ を採用します。

**例3：MA(2)**

$X_t=\varepsilon_t+0.5\varepsilon_{t-1}-0.24\varepsilon_{t-2}$、$\sigma^2=1$ では、

$$
\gamma(0)=1+0.25+0.0576=1.3076,\quad
\gamma(1)=0.5+0.5\times(-0.24)=0.38,\quad
\gamma(2)=-0.24
$$

$$
\rho(1)=\frac{0.38}{1.3076}\approx0.2906,\qquad \rho(2)=\frac{-0.24}{1.3076}\approx-0.1835,\qquad \rho(h)=0\ (h\ge3)
$$

## 現場での使われ方

- **製造・品質管理**：ある時点の段取り替えや原料ロットの影響が数期だけ残る、といった「ショックの短い余韻」を MA モデルで表します。
- **金融**：売買の気配値のはね返り（ビッド・アスク・バウンス）により、高頻度の収益率にはラグ1だけ負の自己相関が出ることがあり、MA(1) で表されます。
- **需要予測**：差分をとった系列に MA(1) を当てはめた ARIMA(0,1,1) は、指数平滑法と同じ予測を与え、在庫管理で広く使われます。

## グラフ・イメージ

中段の標本自己相関は、ラグ1だけが帯の外に出て、ラグ2以降は帯の中に収まります。
θ を ±0.95 まで動かしても、ラグ1の理論値（点）は ±0.5 付近で頭打ちになります。
下段の偏自己相関は、θ>0 では符号を交互に変えながら減衰します。

## 試験のポイント

- MA(1)・MA(2) の自己共分散を「共通の $\varepsilon$ の係数の積の和」で計算させる問題が典型です。
- $|\rho(1)|\le0.5$ と、その理由（$1+\theta^2\ge2|\theta|$）は重要です。
- $\rho(1)$ から $\theta$ を求めると2つの解が出るので、反転可能な方（$|\theta|<1$）を選びます。
- 符号の流儀（$+\theta$ か $-\theta$ か）を問題文で必ず確認します。

## よくある間違い

- × MA(1) の $\rho(1)$ を $\theta$ とする → ○ $\theta/(1+\theta^2)$ です。
- × MA モデルにも定常条件が必要 → ○ 有限次の MA は常に定常です。MA に課すのは反転可能条件です。
- × $-\theta$ の流儀の式に $+\theta$ の公式をそのまま当てはめる → ○ $\rho(1)$ の符号が逆になります。
- × MA の「移動平均」を、傾向を取り出す移動平均（平滑化）と同じものと考える → ○ 名前は似ていますが、MA モデルは雑音の重み付き和で表す確率モデルです。
