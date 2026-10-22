---
id: compound-poisson-process
title: 複合ポアソン過程
reading: ふくごうぽあそんかてい
en: Compound Poisson process
aliases: [複合ポアソン分布, "compound Poisson", 集合的リスクモデル]
chapter: 15
importance: C
tags: [確率過程, ポアソン過程, 条件付き期待値, 保険数理]
related: [poisson-process, independent-stationary-increments, conditional-expectation, conditional-variance, superposition-thinning-poisson, central-limit-theorem]
---

## 一言でいうと

ポアソン過程で起こる出来事のそれぞれに、ランダムな「大きさ」（金額・量）がついていて、その合計を時間の関数として表したものです。
保険金の支払総額や、一定期間の総注文量のモデルに使います。

## 定義・数式

$\{N(t)\}$ を強度 $\lambda$ のポアソン過程、$Y_1,Y_2,\dots$ を $N$ と独立で同じ分布に従う独立な確率変数とします。

$$
S(t)=\sum_{i=1}^{N(t)}Y_i\qquad(N(t)=0\text{ なら }S(t)=0)
$$

を**複合ポアソン過程**といいます。$E[Y^2]<\infty$ のとき

$$
E[S(t)]=\lambda t\,E[Y],\qquad V[S(t)]=\lambda t\,E[Y^2],\qquad
M_{S(t)}(u)=\exp\bigl\{\lambda t\,(M_Y(u)-1)\bigr\}
$$

です。独立増分性・定常増分性をもちます。

## 数式の読み解き

$N(t)=n$ を条件とすると $S(t)$ は $n$ 個の和なので、$E[S\mid N]=NE[Y]$、$V[S\mid N]=NV[Y]$ です。

$$
E[S]=E[N]E[Y]=\lambda tE[Y]
$$

$$
V[S]=E[V[S\mid N]]+V[E[S\mid N]]=\lambda tV[Y]+\lambda t(E[Y])^2=\lambda tE[Y^2]
$$

- 分散は「件数のばらつき」と「1件の大きさのばらつき」の両方から生まれます。$V[N]=E[N]$ というポアソン分布の性質で、2つがきれいに $E[Y^2]$ にまとまります。
- $Y\equiv1$ なら $S(t)=N(t)$ で、ふつうのポアソン過程に戻ります。

## 具体例

**1年間の保険金の支払総額**

請求が月に平均3件（ポアソン過程）、1件の支払額は平均20万円、標準偏差10万円とします。1年（12か月）では $\lambda t=36$、$E[Y^2]=10^2+20^2=500$ なので

$$
E[S]=36\times20=720\ \text{万円},\qquad V[S]=36\times500=18000,\qquad \sqrt{V[S]}=134.16\ \text{万円}
$$

正規近似で、支払総額が900万円を超える確率は

$$
P(S>900)\approx P\Bigl(Z>\frac{900-720}{134.16}\Bigr)=P(Z>1.342)=0.0899
$$

です。支払額をガンマ分布としたシミュレーションでは約 $0.094$ で、右の裾が長いぶん正規近似より少し大きくなります。

## 現場での使われ方

- **金融・保険**：一定期間の保険金支払総額の平均と分散から、保険料や責任準備金を設定します。
- **製造・在庫管理**：注文がポアソン的に届き、1件の注文量がばらつくとき、期間中の総需要の分布から安全在庫を決めます。

## 試験のポイント

- $E[S]=\lambda tE[Y]$、$V[S]=\lambda tE[Y^2]$ を、条件付き期待値・全分散の公式から導けることが重要です。
- 分散に入るのが $V[Y]$ ではなく $E[Y^2]$ である点がよく問われます。
- 期間を $k$ 倍すると平均も分散も $k$ 倍になる（定常増分）ことを使って計算します。

## よくある間違い

- × $V[S]=\lambda tV[Y]$ とする → ○ 件数のばらつきの分 $\lambda t(E[Y])^2$ を加えて $\lambda tE[Y^2]$ です。
- × $V[S]=(\lambda t)^2V[Y]$ のように件数の二乗を掛ける → ○ 独立な $N$ 個の和なので、件数については1乗です。
- × 件数 $N$ と大きさ $Y$ が相関していても同じ式が使える → ○ $N$ と $Y_i$ の独立性が前提です。
