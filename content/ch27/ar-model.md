---
id: ar-model
title: ARモデル
reading: えーあーるもでる
en: Autoregressive model
aliases: [自己回帰モデル, AR過程, "AR(p)", "AR(1)", "AR(2)", 自己回帰過程]
chapter: 27
importance: A
tags: [時系列解析, ARモデル, 定常性, 自己相関]
related: [stationarity-condition, yule-walker-equations, partial-autocorrelation, ma-model, arma-model, serial-correlation]
charts:
  - type: custom
    id: ar-sim
    params: { phi: 0.5, phi2: 0.3, n: 200 }
    sliders:
      - { param: phi, label: "φ1", min: -2, max: 2, step: 0.05 }
      - { param: phi2, label: "φ2", min: -1, max: 1, step: 0.05 }
    caption: AR(2) 過程 X_t = φ1 X_{t-1} + φ2 X_{t-2} + ε_t の1本の系列。φ2=0 で AR(1)。定常域の外に出ると系列が発散する。φ1=0.4, φ2=-0.5 のように φ1²+4φ2<0 だと、周期的に揺れる系列になる。
  - type: custom
    id: acf-plot
    params: { model: ar2, phi: 0.4, phi2: -0.5, n: 300, lags: 20, pacf: 1 }
    sliders:
      - { param: phi, label: "φ1", min: -1, max: 1.5, step: 0.05 }
      - { param: phi2, label: "φ2", min: -0.9, max: 0.9, step: 0.05 }
    caption: AR(2) の自己相関（減衰振動）と偏自己相関（ラグ2の後で切断）。
---

## 一言でいうと

今の値を、自分自身の過去の値の線形結合と新しい雑音で表すモデルです。
「昨日が高ければ今日も高い（または低い）」という記憶を、少数の係数で表します。

## 定義・数式

$\varepsilon_t\sim WN(0,\sigma^2)$ として、

$$
X_t = c + \phi_1X_{t-1} + \phi_2X_{t-2} + \cdots + \phi_pX_{t-p} + \varepsilon_t
$$

を次数 $p$ の**自己回帰モデル** AR($p$) といいます。
後退作用素 $B$（$BX_t=X_{t-1}$）を使うと

$$
\phi(B)X_t = c + \varepsilon_t,\qquad \phi(z)=1-\phi_1z-\phi_2z^2-\cdots-\phi_pz^p
$$

と書けます。**定常条件**は「特性方程式 $\phi(z)=0$ の根がすべて単位円の外（$|z|>1$）」です。

**AR(1)・AR(2) のまとめ**（定常のとき）

| | AR(1) $X_t=c+\phi X_{t-1}+\varepsilon_t$ | AR(2) $X_t=c+\phi_1X_{t-1}+\phi_2X_{t-2}+\varepsilon_t$ |
|---|---|---|
| 定常条件 | $\lvert\phi\rvert<1$ | $\phi_1+\phi_2<1,\ \phi_2-\phi_1<1,\ \lvert\phi_2\rvert<1$ |
| 平均 | $\mu=\dfrac{c}{1-\phi}$ | $\mu=\dfrac{c}{1-\phi_1-\phi_2}$ |
| 分散 $\gamma(0)$ | $\dfrac{\sigma^2}{1-\phi^2}$ | $\dfrac{\sigma^2}{1-\phi_1\rho(1)-\phi_2\rho(2)}$ |
| 自己相関 | $\rho(h)=\phi^{h}$ | $\rho(1)=\dfrac{\phi_1}{1-\phi_2}$、$\rho(h)=\phi_1\rho(h-1)+\phi_2\rho(h-2)$ |
| 偏自己相関 | $\phi_{11}=\phi$、$k\ge2$ で0 | $\phi_{22}=\phi_2$、$k\ge3$ で0 |

## 数式の読み解き

**AR(1) の導出**

(1) 平均：両辺の期待値をとり、定常性から $E[X_t]=E[X_{t-1}]=\mu$ とおくと $\mu=c+\phi\mu$、よって $\mu=c/(1-\phi)$。

(2) MA(∞) 表現：$Y_t=X_t-\mu$ とおくと $Y_t=\phi Y_{t-1}+\varepsilon_t$。これをくり返し代入すると

$$
Y_t=\varepsilon_t+\phi\varepsilon_{t-1}+\phi^2Y_{t-2}=\cdots=\sum_{j=0}^{\infty}\phi^j\varepsilon_{t-j}\qquad(|\phi|<1)
$$

(3) 分散：$\varepsilon$ どうしが無相関なので

$$
\gamma(0)=\sigma^2\sum_{j=0}^{\infty}\phi^{2j}=\frac{\sigma^2}{1-\phi^2}
$$

(4) 自己共分散：$Y_t=\phi Y_{t-1}+\varepsilon_t$ に $Y_{t-h}$（$h\ge1$）を掛けて期待値をとると、$E[\varepsilon_tY_{t-h}]=0$ なので $\gamma(h)=\phi\gamma(h-1)$。よって $\gamma(h)=\phi^h\sigma^2/(1-\phi^2)$、$\rho(h)=\phi^h$ です。

**AR(2) の導出**

$Y_t=\phi_1Y_{t-1}+\phi_2Y_{t-2}+\varepsilon_t$ に $Y_{t-h}$ を掛けて期待値をとります。

- $h\ge1$：$\gamma(h)=\phi_1\gamma(h-1)+\phi_2\gamma(h-2)$。$\gamma(0)$ で割ると $\rho(h)=\phi_1\rho(h-1)+\phi_2\rho(h-2)$。
- $h=1$：$\rho(1)=\phi_1+\phi_2\rho(1)$（$\rho(-1)=\rho(1)$ を使う）なので $\rho(1)=\phi_1/(1-\phi_2)$。
- $h=0$：$Y_t$ を掛けると $E[\varepsilon_tY_t]=\sigma^2$ が残り、$\gamma(0)=\phi_1\gamma(1)+\phi_2\gamma(2)+\sigma^2$。

最後の式を $\gamma(0)$ についてまとめたものが表の分散です。
特性方程式の根が複素数（$\phi_1^2+4\phi_2<0$）のとき、自己相関は**減衰振動**し、系列は周期的に揺れます。

**パラメータの推定**

AR モデルは「説明変数が自分の過去の値である回帰モデル」なので、$X_t$ を $X_{t-1},\dots,X_{t-p}$ に回帰する最小二乗法で係数を推定できます。
最初の $p$ 個の観測値を条件とした正規尤度の最大化（条件付き最尤法）は、この最小二乗法と一致します。
標本自己相関から連立方程式を解くユール・ウォーカー推定もあり、標本が大きければどの方法もほぼ同じ値になります。
ただし説明変数 $X_{t-1}$ は前の時点の誤差を含んでおり、誤差の列全体とは独立でない（強外生性がない）ため、通常の回帰と違って推定量は小標本では偏ります（一致性は持ちます）。

## 具体例

**例1：AR(1)**

$X_t=2+0.6X_{t-1}+\varepsilon_t$、$\sigma^2=1$ とします。

$$
\mu=\frac{2}{1-0.6}=5,\qquad \gamma(0)=\frac{1}{1-0.36}=1.5625,\qquad \rho(1)=0.6,\ \rho(2)=0.36,\ \rho(3)=0.216
$$

**例2：AR(2)**

$X_t=0.5X_{t-1}+0.3X_{t-2}+\varepsilon_t$、$\sigma^2=1$ とします。
$\phi_1+\phi_2=0.8<1$、$\phi_2-\phi_1=-0.2<1$、$|\phi_2|=0.3<1$ で定常です。

$$
\rho(1)=\frac{0.5}{0.7}\approx0.7143,\quad
\rho(2)=0.5\times0.7143+0.3\approx0.6571,\quad
\rho(3)=0.5\times0.6571+0.3\times0.7143\approx0.5429
$$

$$
\gamma(0)=\frac{1}{1-0.5\times0.7143-0.3\times0.6571}=\frac{1}{0.4457}\approx2.2436
$$

特性方程式 $1-0.5z-0.3z^2=0$ の根は $z=\dfrac{-0.5\pm\sqrt{1.45}}{0.6}$ すなわち $1.1736$ と $-2.8403$ で、どちらも絶対値が1より大きいことも確認できます。

**例3：周期的に揺れる AR(2)**

$\phi_1=0.4,\ \phi_2=-0.5$ では $\phi_1^2+4\phi_2=0.16-2=-1.84<0$ で、根は複素数 $0.4\pm1.3565i$（絶対値 $\sqrt{2}\approx1.4142$）です。
自己相関は $\rho(h)\propto(\sqrt{0.5})^h\cos(\omega h+\text{定数})$、$\cos\omega=\dfrac{\phi_1}{2\sqrt{-\phi_2}}=0.2828$ の形で、$\omega\approx1.2840$、周期 $2\pi/\omega\approx4.89$ 期の減衰振動になります。

## 現場での使われ方

- **経済統計**：GDP 成長率やインフレ率を AR モデルで表し、数期先を予測します。次数は PACF や AIC で決めます。
- **製造・品質管理**：連続生産の測定値が前の値を引きずる（自己相関がある）とき、AR(1) を当てはめた残差で管理図を作ります。
- **回帰分析**：誤差が AR(1) に従う回帰（系列相関）では、一般化最小二乗法で係数を推定します。
- **信号処理**：音声を AR モデルで表す線形予測は、音声の圧縮・合成の基本です。

## グラフ・イメージ

1つめの図で φ1, φ2 を動かすと、定常域の中では0のまわりを行き来し、外に出ると発散します。
φ1=0.4, φ2=-0.5 のような値では、系列が周期的に上下します。
2つめの図では、自己相関が減衰振動し、偏自己相関がラグ2の後で切れることを確認できます。

## 試験のポイント

- AR(1) の平均・分散・自己相関は、上の (1)〜(4) の手順で導出できるようにします。
- AR(2) の $\rho(1)=\phi_1/(1-\phi_2)$ と漸化式、$\gamma(0)$ の求め方は重要です。
- 定常条件は「特性方程式 $1-\phi_1z-\cdots-\phi_pz^p=0$ の根が単位円の外」。AR(2) では3つの不等式（三角形の領域）と同値です。
- 偏自己相関はラグ $p$ の後で切断、自己相関は減衰。次数の同定と結びつけて問われます。

## よくある間違い

- × 平均を $c$ とする → ○ 定数項 $c$ と平均 $\mu=c/(1-\phi_1-\cdots-\phi_p)$ は違います。
- × AR(1) の分散を $\sigma^2/(1-\phi)$ とする → ○ $\sigma^2/(1-\phi^2)$ です。
- × AR(2) の定常条件を $|\phi_1|<1,\ |\phi_2|<1$ とする → ○ $\phi_1=0.9,\ \phi_2=0.2$ は両方とも絶対値1未満ですが、$\phi_1+\phi_2=1.1>1$ で非定常です（根の1つが $0.9221$ で単位円の内側）。
