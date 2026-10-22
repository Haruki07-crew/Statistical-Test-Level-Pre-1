---
id: goodness-of-fit-indices
title: 適合度指標
reading: てきごうどしひょう
en: Goodness-of-fit indices
aliases: [RMSEA, CFI, GFI, AGFI, TLI, SRMR, モデル適合度]
chapter: 26
importance: C
tags: [構造方程式モデリング, 因子分析, モデル評価]
related: [structural-equation-modeling, path-analysis, maximum-likelihood-factor-analysis, likelihood-ratio-test, aic, chi-squared-distribution]
---

## 一言でいうと

構造方程式モデリング（SEM）や確認的因子分析で、モデルがデータの共分散行列をどれだけうまく再現しているかを、1つの数で表す指標の総称です。
カイ二乗検定だけでは標本が大きいと何でも棄却されてしまうため、「どの程度ずれているか」を測る指標を組み合わせて使います。

## 定義・数式

観測変数 $p$ 個、標本の大きさ $n$、モデルのカイ二乗統計量を $\chi^2=(n-1)F_{\mathrm{ML}}(\hat{\boldsymbol\theta})$、自由度を $\mathit{df}$ とします。
比較の基準として、すべての変数が無相関とする**独立モデル**の $\chi^2_0$、$\mathit{df}_0=p(p-1)/2$ も使います。代表的な指標は次のとおりです。

$$
\begin{aligned}
\mathrm{RMSEA}&=\sqrt{\frac{\max(\chi^2-\mathit{df},\,0)}{\mathit{df}\,(n-1)}}\\
\mathrm{CFI}&=1-\frac{\max(\chi^2-\mathit{df},\,0)}{\max(\chi^2_0-\mathit{df}_0,\ \chi^2-\mathit{df},\ 0)}\\
\mathrm{TLI}&=\frac{\chi^2_0/\mathit{df}_0-\chi^2/\mathit{df}}{\chi^2_0/\mathit{df}_0-1}\\
\mathrm{GFI}&=1-\frac{\mathrm{tr}\{(\hat\Sigma^{-1}S-I)^2\}}{\mathrm{tr}\{(\hat\Sigma^{-1}S)^2\}},\qquad \mathrm{AGFI}=1-\frac{p(p+1)}{2\,\mathit{df}}(1-\mathrm{GFI})
\end{aligned}
$$

SRMR は、標準化した残差 $S-\hat\Sigma$ の下三角（対角を含む $p(p+1)/2$ 個）の2乗平均の平方根です。GFI は最尤法の場合の式で、推定法によって式が変わります。

## 数式の読み解き

- **RMSEA**：$\chi^2-\mathit{df}$ は、モデルが正しいときの期待値 $\mathit{df}$ を超えた「ずれの量」の推定です。これを自由度と $n-1$ で割るので、標本の大きさやモデルの複雑さに左右されにくくなります。小さいほどよい指標です。
- **CFI・TLI**：独立モデル（最悪の基準）からどれだけ改善したかの割合です。1に近いほどよい指標です。TLI は $\chi^2/\mathit{df}$ を使うので、自由度を使いすぎる複雑なモデルに罰則がかかります。
- **GFI・AGFI**：回帰の決定係数に似た「共分散行列の何割を再現したか」です。AGFI は自由度で調整した版で、自由度調整済み決定係数にあたります。
- よく使われる目安は、RMSEA 0.05 以下（0.1 以上は不適）、CFI・TLI 0.95 以上、SRMR 0.08 以下、GFI 0.9 以上です。ただし目安にすぎず、1つの指標だけで判断しません。
- 入れ子でないモデルどうしの比較には、AIC や BIC も使います。

## 具体例

**構造方程式モデリングの項目の4指標・1因子モデル（$n=200$）**

最尤推定で $\chi^2=2.845$、$\mathit{df}=2$ でした。標本相関行列の行列式から独立モデルは $\chi^2_0=199\times(-\log|S|)=187.105$、$\mathit{df}_0=6$ です（numpy・scipy で計算）。

$$
\mathrm{RMSEA}=\sqrt{\frac{2.845-2}{2\times199}}\approx0.046,\qquad
\mathrm{CFI}=1-\frac{0.845}{181.105}\approx0.995
$$

$$
\mathrm{TLI}=\frac{187.105/6-2.845/2}{187.105/6-1}=\frac{31.184-1.423}{30.184}\approx0.986
$$

GFI は $0.993$、AGFI は $1-\frac{20}{4}(1-0.993)\approx0.963$、SRMR は $0.023$ でした。
どの指標も目安を満たしており、カイ二乗検定（$p=0.241$）の結論とも合っています。

## 現場での使われ方

- **行政・調査**：心理尺度や満足度尺度の因子構造を確認的因子分析で検証し、RMSEA・CFI・SRMR を並べて報告します。
- **マーケティング**：顧客満足 → ロイヤルティ → 再購入のような因果モデルを SEM で当てはめ、複数の候補モデルを適合度指標と AIC で比べます。
- **医療**：生活の質（QOL）の質問票を新しい言語に翻訳したとき、元の因子構造が保たれているかを適合度指標で確認します。

## 試験のポイント

- 大標本ではカイ二乗検定が小さなずれでも棄却するため、適合度指標を併用する、という理由が重要です。
- RMSEA は $\chi^2$、$\mathit{df}$、$n$ から計算できるので、数値を与えて計算させる形に備えます。
- 「小さいほどよい（RMSEA・SRMR）」と「1に近いほどよい（CFI・TLI・GFI）」を区別して覚えます。

## よくある間違い

- × RMSEA は大きいほど当てはまりがよい → ○ RMSEA はずれの大きさなので、小さいほどよい指標です。
- × 適合度指標がよければモデルが正しいと証明された → ○ 同じくらい当てはまる別のモデル（同値モデル）がありえます。理論的な妥当性と合わせて判断します。
- × 自由度0のモデルでも適合度指標で評価できる → ○ 丁度識別のモデルは必ず完全に当てはまるので、適合度は評価できません。
