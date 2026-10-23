---
id: box-cox-transformation
title: Box-Cox変換
reading: ぼっくすこっくすへんかん
en: Box-Cox transformation
aliases: [ボックス・コックス変換, ボックスコックス変換, Box–Cox変換]
chapter: 4
importance: A
tags: [データ変換, 最尤法, 正規化, べき乗変換]
related: [power-transformation, log-transformation, change-of-variables, maximum-likelihood-estimator, likelihood-function, normal-qq-plot]
charts:
  - type: custom
    id: box-cox
    params: { lambda: 0.5 }
    sliders:
      - { param: lambda, label: "λ", min: -2, max: 2, step: 0.1 }
    caption: 横軸を元の値 y（0 < y ≤ 5 程度）、縦軸を変換後の値として、Box-Cox 変換の曲線を描いた図。λ=1 は直線（形は変わらない）、λ を下げるほど大きい y が強く縮められ、λ=0 で対数の曲線に一致する。どの λ でも y=1 で値 0・傾き 1 を通る。
---

## 一言でいうと

べき乗変換と対数変換を、1つのパラメータ $\lambda$ でつないだ変換です。
データが正規分布に最も近くなる $\lambda$ を最尤法で選べるのが特長です。

## 定義・数式

$y>0$ に対して、

$$
y^{(\lambda)}=
\begin{cases}
\dfrac{y^{\lambda}-1}{\lambda} & (\lambda\ne 0)\\[3mm]
\log y & (\lambda=0)
\end{cases}
$$

と定めます。変換後のデータ $y_i^{(\lambda)}$ $(i=1,\dots,n)$ が独立に $N(\mu,\sigma^2)$ に従う（回帰なら $N(\boldsymbol{x}_i^{\top}\boldsymbol{\beta},\sigma^2)$）と仮定すると、元のデータ $y_i$ に対する対数尤度は

$$
\ell(\lambda,\mu,\sigma^2)=-\frac{n}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^{n}\left(y_i^{(\lambda)}-\mu\right)^2+(\lambda-1)\sum_{i=1}^{n}\log y_i
$$

です。$\mu,\sigma^2$ について最大化すると（$\hat\sigma^2(\lambda)$ は変換後のデータの偏差平方和を $n$ で割ったもの）、**プロファイル対数尤度**

$$
\ell_p(\lambda)=-\frac{n}{2}\log\hat\sigma^2(\lambda)+(\lambda-1)\sum_{i=1}^{n}\log y_i+\text{定数}
$$

が得られ、これを最大にする $\hat\lambda$ を使います。

## 数式の読み解き

- $(y^{\lambda}-1)/\lambda$：$y^{\lambda}$ から1を引いて $\lambda$ で割るのは、$\lambda\to 0$ で連続にするためです。$y^{\lambda}=e^{\lambda\log y}=1+\lambda\log y+O(\lambda^2)$ なので、$\lambda\to 0$ の極限が $\log y$ になります。
- $\lambda<0$ でも $\lambda$ で割ることで、変換が常に増加関数になり、大小の順序が保たれます。
- $(\lambda-1)\sum\log y_i$：変数変換のヤコビアンの項です。$dy^{(\lambda)}/dy=y^{\lambda-1}$ なので、変換後の正規密度に $\prod y_i^{\lambda-1}$ を掛けると元のデータの密度になります。この項がないと、$\lambda$ を変えるだけで尺度が変わり、尤度が比べられません。
- 代表的な $\lambda$：$1$（ほぼ変換なし）、$1/2$（平方根）、$0$（対数）、$-1$（逆数）。実務では $\hat\lambda$ に近い「きりのよい値」を選ぶことも多いです。

## 具体例

**例1：1つの値を変換する**

$y=4$ のとき、

$$
\lambda=1:\ 3,\qquad \lambda=\tfrac12:\ \frac{\sqrt4-1}{1/2}=2,\qquad \lambda=0:\ \log 4\approx 1.3863,\qquad \lambda=-1:\ \frac{4^{-1}-1}{-1}=0.75
$$

$\lambda$ が小さいほど、大きな値が強く縮められます。

**例2：最尤法で $\lambda$ を選ぶ**

右に歪んだデータ $1, 2, 3, 5, 8, 13, 21, 34$（$n=8$）を考えます。$\sum\log y_i\approx 14.6165$ です。各 $\lambda$ でプロファイル対数尤度（定数を除く）を計算すると、

| $\lambda$ | $\hat\sigma^2(\lambda)$ | $\ell_p(\lambda)$ |
|---|---|---|
| $-1$ | 0.0947 | $-19.804$ |
| $-0.5$ | 0.2762 | $-16.778$ |
| $0$ | 1.2725 | $-15.580$ |
| $0.5$ | 9.7734 | $-16.427$ |
| $1$ | 115.3594 | $-18.992$ |

最大は $\lambda=0$ 付近で、数値的に最大化すると $\hat\lambda\approx 0.039$ です。このデータはほぼ等比的に増えているので、対数変換が適していると判断できます。

$\hat\sigma^2(\lambda)$ だけを見ると $\lambda=-1$ が最小ですが、それは逆数変換で値の尺度自体が小さくなったためです。ヤコビアンの項を加えて初めて公平に比べられます。

## 現場での使われ方

- **製造・品質管理**：工程能力指数を計算する前に、正規分布でない測定値を Box-Cox 変換して正規性を確保します。
- **医療・生物**：検査値や薬物濃度を回帰分析する際、残差の正規性と等分散性を改善する変換を、試行錯誤ではなく最尤法で選びます。
- **IT・Webサービス**：応答時間や購入金額のような右に歪んだ特徴量を、予測モデルに入れる前に変換します。

## グラフ・イメージ

元の値 $y$ と変換後の値 $y^{(\lambda)}$ の関係を曲線で描いています。$\lambda$ を 1 から下げていくと、大きな $y$ ほど強く押しつぶされ、$\lambda=0$ で対数曲線になります。どの曲線も $y=1$ で値0・傾き1を通るので、$\lambda$ を変えても曲線が連続的に変わることを確認してください。

## 試験のポイント

- 定義式と、$\lambda\to0$ の極限が $\log y$ になる理由（$y^{\lambda}=e^{\lambda\log y}$ のテイラー展開、またはロピタルの定理）を説明できるようにしておきます。
- 尤度に $(\lambda-1)\sum\log y_i$ のヤコビアン項が入ることが要点です。変数変換の公式とのつながりで問われます。
- $\lambda$ の信頼区間は、尤度比 $2\{\ell_p(\hat\lambda)-\ell_p(\lambda)\}\le\chi^2_{1}(0.05)=3.841$ を満たす $\lambda$ の範囲として求められます。
- データに0や負の値があるときは、そのままでは使えません。定数を足してずらす方法や、負の値も扱える Yeo-Johnson 変換があります。

## よくある間違い

- × $\hat\sigma^2(\lambda)$ が最小になる $\lambda$ を選ぶ → ○ ヤコビアン項を含めたプロファイル対数尤度を最大化します。
- × $\lambda=0$ のときも $(y^0-1)/0$ で計算する → ○ $\lambda=0$ は極限として $\log y$ と定義します。
- × Box-Cox 変換をすれば必ず正規分布になる → ○ 「この族の中で最も正規分布に近い」変換が選ばれるだけです。変換後も Q-Q プロットなどで確認します。
