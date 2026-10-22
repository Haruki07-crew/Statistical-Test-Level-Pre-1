---
id: tic-ebic
title: TICと一般化情報量規準
reading: てぃーあいしーといっぱんかじょうほうりょうきじゅん
en: Takeuchi information criterion and generalized information criterion
aliases: [TIC, 竹内情報量規準, GIC, "generalized information criterion", EIC, ブートストラップ情報量規準, WAIC, 広く使える情報量規準]
chapter: 30
importance: C
tags: [モデル選択, 情報量規準, モデルの誤特定]
related: [aic, expected-log-likelihood, kullback-leibler-divergence, fisher-information, aicc, bootstrap]
---

## 一言でいうと

AIC は「モデルが正しい」ことを前提に罰則を $2k$ としています。
TIC はモデルが正しくなくても使えるように罰則をデータから推定し、GIC はさらに最尤法以外の推定法にも使えるよう一般化した規準です。

## 定義・数式

**TIC**（竹内の情報量規準）は、最尤推定量 $\hat{\boldsymbol\theta}$ について

$$
\mathrm{TIC}=-2\ell(\hat{\boldsymbol\theta})+2\,\mathrm{tr}\bigl(\hat I\,\hat J^{-1}\bigr)
$$

です。$\hat I,\hat J$ は1観測あたりの次の行列の推定値です。

$$
I=E_g\!\left[\frac{\partial\log f}{\partial\boldsymbol\theta}\frac{\partial\log f}{\partial\boldsymbol\theta^\top}\right],\qquad
J=-E_g\!\left[\frac{\partial^2\log f}{\partial\boldsymbol\theta\,\partial\boldsymbol\theta^\top}\right]
$$

$g$ は真の分布です。モデルが正しければ $I=J$（フィッシャー情報量）なので $\mathrm{tr}(IJ^{-1})=k$ となり、AIC に一致します。

**GIC**（一般化情報量規準、小西・北川）は、推定量を統計的汎関数 $\hat{\boldsymbol\theta}=T(\hat G)$ として表し、その影響関数を使って偏りを $\mathrm{tr}(\hat R^{-1}\hat Q)$ の形で評価します。最尤推定なら TIC に一致し、正則化付きの推定やロバスト推定にも使えます。

## 数式の読み解き

- 最大対数尤度は、期待対数尤度を上向きに偏って推定しています。その偏りが一般には $\mathrm{tr}(IJ^{-1})$ で、モデルが正しいときだけ $k$ になります。
- $I$ はスコアのばらつき、$J$ は対数尤度の曲がり具合です。データの裾がモデルより重いと $I$ が大きくなり、罰則も大きくなります。
- 偏りを解析的に求めずに、ブートストラップで数値的に推定する規準を **EIC** といいます。ベイズ推測で事後分布を使って汎化損失を推定する規準には **WAIC**（渡辺）があります。

## 具体例

**例：裾の重いデータに正規分布を当てはめる**

10個のデータ $-3.1,-0.8,-0.4,-0.2,0.0,0.1,0.3,0.5,0.9,2.7$ に $N(\mu,\sigma^2)$（$k=2$）を当てはめます。
最尤推定値は $\hat\mu=0$、$\hat\sigma^2=1.89$ で、$-2\ell(\hat{\boldsymbol\theta})=34.745$、$\mathrm{AIC}=38.745$ です。

正規モデルでは、標本尖度 $\hat\kappa=\hat\mu_4/\hat\sigma^4$ を使って

$$
\mathrm{tr}(\hat I\hat J^{-1})=1+\frac{\hat\kappa-1}{2}
$$

となります（$\mu$ の分が1、$\sigma^2$ の分が $(\hat\kappa-1)/2$）。このデータでは $\hat\kappa=4.106$ なので $\mathrm{tr}=2.553$、

$$
\mathrm{TIC}=34.745+2\times2.553=39.850
$$

です。正規分布なら $\kappa=3$ で罰則は $4$ ですが、裾が重いぶん罰則が $5.106$ に増えています。

## 現場での使われ方

- **金融**：収益率のように裾の重いデータに正規モデルを当てはめるとき、モデルの誤特定を考慮した比較に使います。
- **IT・Webサービス**：正則化付きの回帰やロバスト推定で作ったモデルを比べるとき、GIC で罰則を評価します。ベイズモデルの比較では WAIC が広く使われます。

## 試験のポイント

- TIC の罰則 $2\,\mathrm{tr}(IJ^{-1})$ と、モデルが正しいときに $2k$（AIC）になる理由（$I=J$）を押さえます。
- AIC は「真の分布がモデルに含まれる」ことを前提にした近似であり、TIC・GIC はその拡張であることを説明できるようにします。
- TIC は $\hat I,\hat J$ の推定が不安定で、小標本ではかえってばらつくため、実際には AIC が使われることが多い点も重要です。

## よくある間違い

- × TIC の罰則はパラメータ数 $k$ だけで決まる → ○ データ（真の分布）に依存します。$k$ になるのはモデルが正しいときです。
- × GIC は BIC を一般化したもの → ○ GIC は AIC・TIC と同じく期待対数尤度の推定から導かれる規準で、最尤法以外の推定量に広げたものです。
- × TIC のほうが常に AIC より良い → ○ 罰則の推定誤差が大きく、標本が少ないと不安定になります。
