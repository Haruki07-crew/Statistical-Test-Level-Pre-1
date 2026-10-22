---
id: bic
title: BIC
reading: びーあいしー
en: Bayesian information criterion
aliases: [ベイズ情報量規準, シュワルツ規準, SBC, シュワルツのベイズ情報量規準]
chapter: 30
importance: A
tags: [モデル選択, 情報量規準, ベイズ, ラプラス近似]
related: [aic, marginal-likelihood, bayes-factor, information-criteria-comparison, consistent-estimator, likelihood-ratio-test]
charts:
  - type: custom
    id: line-data
    params:
      x: [5, 8, 10, 20, 50, 100, 200, 500, 1000]
      series:
        - { label: "AIC の罰則（1パラメータあたり 2）", y: [2, 2, 2, 2, 2, 2, 2, 2, 2] }
        - { label: "BIC の罰則（1パラメータあたり log n）", y: [1.609, 2.079, 2.303, 2.996, 3.912, 4.605, 5.298, 6.215, 6.908] }
      xlabel: "標本サイズ n"
      ylabel: "パラメータ1個あたりの罰則"
    caption: BIC の罰則 log n は n が e²≈7.39 を超えると AIC の 2 より大きくなり、n とともに増え続ける。このため BIC は大標本ほど小さなモデルを好む。
  - type: custom
    id: line-data
    params:
      x: [0, 1, 2, 3]
      series:
        - { label: "AIC", y: [23.727, 17.256, 9.621, 11.618] }
        - { label: "BIC", y: [23.886, 17.495, 9.938, 12.016] }
      xlabel: "多項式回帰の次数"
      ylabel: "規準の値"
    caption: 例1（n=8）の AIC と BIC。n=8 では log 8 ≈ 2.079 で罰則の差が小さく、どちらも2次を選ぶ。
---

## 一言でいうと

最大対数尤度から「パラメータ数 × 標本サイズの対数」の分だけ割り引いてモデルを比べる規準です。
ベイズの立場で「データが得られる確率（周辺尤度）が最も高いモデル」を近似的に選ぶもので、値が小さいほど良いモデルです。

## 定義・数式

パラメータ数 $k$ のモデルを大きさ $n$ のデータに当てはめ、最大対数尤度を $\ell(\hat{\boldsymbol\theta})$ とすると、

$$
\mathrm{BIC} = -2\,\ell(\hat{\boldsymbol\theta}) + k\log n
$$

です（$\log$ は自然対数）。候補のうち **BIC が最小のモデルを選びます**。

正規線形回帰（係数 $p$ 個、$\sigma^2$ 未知）なら、定数を省いて

$$
\mathrm{BIC} = n\log\frac{S_e}{n} + (p+1)\log n
$$

と書けます。$S_e$ は残差平方和です。

BIC は、モデル $M$ の**周辺尤度**

$$
p(\boldsymbol x\mid M)=\int f(\boldsymbol x\mid\boldsymbol\theta)\,\pi(\boldsymbol\theta)\,d\boldsymbol\theta
$$

（$\pi$ は事前分布）について、$-2\log p(\boldsymbol x\mid M)$ を $n$ が大きいときに近似したものです。

## 数式の読み解き

- $-2\ell(\hat{\boldsymbol\theta})$：当てはまりの悪さ（AIC と共通）。
- $k\log n$：複雑さへの罰則。AIC の $2k$ と違い、$n$ とともに大きくなります。$\log n>2$ となるのは $n\ge 8$ のときです。
- 周辺尤度は「事前分布に従ってパラメータをばらまいたとき、平均してどれだけデータを説明できるか」です。パラメータが多いと、データに合わない値にも事前確率を割くので、自然に罰則がかかります。

**導出の筋道（ラプラス近似）**

対数尤度を最尤推定値 $\hat{\boldsymbol\theta}$ のまわりで2次までテイラー展開します。$\hat{\boldsymbol\theta}$ では1次の項が 0 なので、

$$
\ell(\boldsymbol\theta) \approx \ell(\hat{\boldsymbol\theta}) - \frac{n}{2}(\boldsymbol\theta-\hat{\boldsymbol\theta})^\top \hat J\,(\boldsymbol\theta-\hat{\boldsymbol\theta})
$$

です（$\hat J$ は1観測あたりの観測情報行列）。$n$ が大きいと被積分関数は $\hat{\boldsymbol\theta}$ の近くに集中するので、$\pi(\boldsymbol\theta)\approx\pi(\hat{\boldsymbol\theta})$ とおき、残りを多変量正規分布の密度の積分として計算します。

$$
\begin{aligned}
p(\boldsymbol x\mid M) &\approx e^{\ell(\hat{\boldsymbol\theta})}\,\pi(\hat{\boldsymbol\theta}) \int \exp\Big\{-\frac{n}{2}(\boldsymbol\theta-\hat{\boldsymbol\theta})^\top \hat J(\boldsymbol\theta-\hat{\boldsymbol\theta})\Big\}d\boldsymbol\theta\\
&= e^{\ell(\hat{\boldsymbol\theta})}\,\pi(\hat{\boldsymbol\theta})\,(2\pi)^{k/2}\,|n\hat J|^{-1/2}
\end{aligned}
$$

$|n\hat J|=n^k|\hat J|$ なので、対数をとって $-2$ を掛けると

$$
-2\log p(\boldsymbol x\mid M) \approx -2\ell(\hat{\boldsymbol\theta}) + k\log n \;\underbrace{-\,2\log\pi(\hat{\boldsymbol\theta}) - k\log(2\pi) + \log|\hat J|}_{n\text{ によらない }O(1)\text{ の項}}
$$

$n\to\infty$ で増えていく項だけを残したものが BIC です。

## 具体例

**例1：多項式回帰の次数（$n=8$）**

月数 $x=1,\dots,8$、売上 $y=(2.2,3.3,3.8,3.7,4.9,4.8,4.3,4.4)$ に0〜3次の多項式を当てはめます。$-2\ell$ に罰則 $(p+1)\log 8$（$\log 8=2.0794$）を足すと次のとおりです。

| 次数 | $-2\ell(\hat{\boldsymbol\theta})$ | AIC | BIC |
|---|---|---|---|
| 0 | 19.727 | 23.727 | 23.886 |
| 1 | 11.256 | 17.256 | 17.495 |
| 2 | 1.621 | 9.621 | 9.938 |
| 3 | 1.618 | 11.618 | 12.016 |

2次の BIC は $1.621+4\times2.0794=9.938$ です。BIC も2次を選びます。

**例2：AIC と BIC で結論が分かれる（$n=100$）**

コインを100回投げて表が60回。モデル0「表の確率 0.5」とモデル1「表の確率は未知」を比べます。$\ell_0=-69.315$、$\ell_1=-67.301$ です。

$$
\begin{aligned}
\mathrm{AIC}_0 &= 138.629, & \mathrm{AIC}_1 &= 134.602+2 = 136.602,\\
\mathrm{BIC}_0 &= 138.629, & \mathrm{BIC}_1 &= 134.602+\log 100 = 139.208
\end{aligned}
$$

AIC はモデル1、BIC はモデル0 を選びます。尤度比統計量は $2(\ell_1-\ell_0)=4.027$ で、AIC の境目 2 は超えますが、BIC の境目 $\log 100=4.605$ には届きません。

ラプラス近似の精度も確かめます。モデル1で事前分布を一様分布 $U(0,1)$ にすると、周辺尤度は厳密に $B(61,41)$（ベータ関数）で、$\log p(\boldsymbol x\mid M_1)=-69.406$ です。ラプラス近似（$O(1)$ の項も含めた形）では $-69.398$ で、ほぼ一致します。
厳密なベイズファクターは $p(\boldsymbol x\mid M_1)/p(\boldsymbol x\mid M_0)=e^{-69.406+69.315}=0.913$、BIC の差から求めた近似値は $e^{-(139.208-138.629)/2}=0.749$ です。どちらも1未満で、モデル0 をわずかに支持します。

## 現場での使われ方

- **マーケティング**：顧客のクラスタリング（混合分布モデル）で、クラスタの数を BIC 最小で決めます。AIC より少ないクラスタ数になりやすく、解釈しやすい分け方になります。
- **IT・Webサービス**：大量のログデータでモデルを比べるとき、$n$ が非常に大きいと AIC ではほとんどの変数が「必要」と判定されます。BIC を使うと本当に効いている変数に絞り込めます。
- **金融・保険**：時系列モデルの次数選択で、AIC と BIC の両方を計算し、BIC の選ぶ簡潔なモデルを基本にします。
- **医療**：疫学で、多数の候補因子から「真に関係する因子」を見つけたいときに、一致性のある BIC が使われます。

## グラフ・イメージ

1つ目は、パラメータ1個あたりの罰則を $n$ に対して描いたものです。$n=8$ 付近で AIC と BIC が入れ替わり、それ以降は BIC の罰則が対数的に大きくなります。
2つ目は例1の AIC と BIC です。$n=8$ では両者の差はわずかで、同じ次数を選びます。

## 試験のポイント

- 定義 $\mathrm{BIC}=-2\ell(\hat{\boldsymbol\theta})+k\log n$ と、AIC との罰則の違いを確実に覚えます。$n\ge8$ で BIC の罰則のほうが重くなります。
- **一致性**：候補の中に真のモデルがあれば、BIC が真のモデルを選ぶ確率は $n\to\infty$ で 1 になります。AIC は一致性をもたず、大きすぎるモデルを選ぶ確率が残ります。逆に予測の良さでは AIC が有利な場面が多い、という対比が重要です。
- BIC は周辺尤度のラプラス近似です。2つのモデルの BIC の差は、ベイズファクターの対数の $-2$ 倍の近似になります：$\mathrm{BIC}_0-\mathrm{BIC}_1 \approx 2\log\mathrm{BF}_{10}$。
- 入れ子のモデルでパラメータが1個違うとき、BIC で大きいモデルが選ばれる条件は尤度比統計量 $>\log n$ です。

## よくある間違い

- × BIC の $\log$ を常用対数で計算する → ○ 自然対数です（$\log 100=4.605$）。
- × BIC は事前分布を細かく決めないと計算できない → ○ 事前分布の影響は $O(1)$ の項に入り、BIC では無視されます（そのぶん近似は粗くなります）。
- × BIC も AIC も「予測の良さ」を測っている → ○ AIC は予測（KL 情報量）、BIC は周辺尤度（真のモデルの特定）を狙っていて、目的が違います。
- × $n$ が大きいほど AIC と BIC の結論は近づく → ○ 罰則の差 $k(\log n-2)$ は $n$ とともに広がるので、大標本ほど結論が分かれやすくなります。
