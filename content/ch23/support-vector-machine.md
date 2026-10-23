---
id: support-vector-machine
title: サポートベクターマシン
reading: さぽーとべくたーましん
en: Support vector machine
aliases: [SVM, サポートベクトルマシン, マージン最大化]
chapter: 23
importance: A
tags: [判別分析, 機械学習, マージン, カーネル法, 二次計画]
related: [kernel-method, linear-discriminant-analysis, logistic-regression, regularization, k-nearest-neighbors, cross-validation]
charts:
  - type: custom
    id: svm-margin
    params: { gap: 1, n: 30 }
    sliders:
      - { param: gap, label: "2群の間隔", min: -0.5, max: 2, step: 0.25 }
    caption: 2群の点と、マージン最大の分離直線（実線）、マージンの境界（破線）、サポートベクトル（丸で囲んだ点）。境界を決めるのはマージン上の少数の点だけである。
---

## 一言でいうと

2群を分ける直線（超平面）のうち、**両側の最も近い点までの余白（マージン）が最大**になるものを選ぶ判別方法です。
境界の近くにある少数の点（サポートベクトル）だけで境界が決まり、カーネル法と組み合わせると曲がった境界も作れます。

## 定義・数式

ラベル $y_i \in \{-1, +1\}$ のついたデータ $(\boldsymbol{x}_i, y_i)$, $i = 1, \dots, n$ について、判別関数 $f(\boldsymbol{x}) = \boldsymbol{w}^\top \boldsymbol{x} + b$ の符号で群を判別します。

**ハードマージン**（2群が直線で完全に分けられる場合）は、次の最適化問題です。

$$
\min_{\boldsymbol{w}, b} \; \frac{1}{2}\|\boldsymbol{w}\|^2 \quad \text{subject to} \quad y_i(\boldsymbol{w}^\top \boldsymbol{x}_i + b) \ge 1 \quad (i = 1, \dots, n)
$$

このときマージン（境界 $f = 0$ から最も近い点までの距離）は $1/\|\boldsymbol{w}\|$、両側を合わせた幅は $2/\|\boldsymbol{w}\|$ です。

**ソフトマージン**（重なりがある場合）は、はみ出しの量 $\xi_i \ge 0$ を許して

$$
\min_{\boldsymbol{w}, b, \boldsymbol{\xi}} \; \frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum_{i=1}^{n} \xi_i \quad \text{subject to} \quad y_i(\boldsymbol{w}^\top \boldsymbol{x}_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0
$$

とします。$C > 0$ ははみ出しへの罰の強さです。これは**ヒンジ損失** $\max\{0,\ 1 - y_i f(\boldsymbol{x}_i)\}$ の和に $L_2$ 正則化を加えたものと同じです。

**双対問題**による解は

$$
\boldsymbol{w} = \sum_{i=1}^{n} \alpha_i y_i \boldsymbol{x}_i, \qquad \sum_{i=1}^{n} \alpha_i y_i = 0, \qquad 0 \le \alpha_i \le C
$$

の形になり（ハードマージンでは上限なし）、$\alpha_i > 0$ となる点を**サポートベクトル**といいます。

## 数式の読み解き

- **マージンが $1/\|\boldsymbol{w}\|$ になる理由**：点 $\boldsymbol{x}_i$ から超平面 $\boldsymbol{w}^\top\boldsymbol{x} + b = 0$ までの距離は $|\boldsymbol{w}^\top\boldsymbol{x}_i + b|/\|\boldsymbol{w}\|$ です。$\boldsymbol{w}, b$ を定数倍しても同じ超平面なので、最も近い点で $|\boldsymbol{w}^\top\boldsymbol{x}_i + b| = 1$ となるように尺度をそろえます。すると距離は $1/\|\boldsymbol{w}\|$ で、マージン最大化は $\|\boldsymbol{w}\|$ の最小化、つまり $\frac{1}{2}\|\boldsymbol{w}\|^2$ の最小化になります。
- **サポートベクトル**：ラグランジュ乗数 $\alpha_i$ は、制約がぎりぎり（$y_i f(\boldsymbol{x}_i) = 1$、マージン上）の点でだけ正になります（相補性条件）。$\boldsymbol{w}$ は $\alpha_i > 0$ の点だけで決まるので、境界から遠い点を動かしても境界は変わりません。
- **$C$ の役割**：$C$ が大きいとはみ出しを強く嫌い、マージンが狭く複雑な境界になります。$C$ が小さいと多少の誤りを許してマージンを広くとります。$C$ は交差検証で選びます。
- **カーネル法**：双対問題とその解は、データを内積 $\boldsymbol{x}_i^\top\boldsymbol{x}_j$ の形でしか使いません。これをカーネル関数 $K(\boldsymbol{x}_i, \boldsymbol{x}_j)$ に置き換えると、高次元の特徴空間での線形判別、つまり元の空間での非線形な判別ができます（カーネルトリック）。

**線形判別分析との違い**：線形判別分析は全データの平均と共分散から境界を決め、確率モデル（正規分布）に基づきます。SVM は境界付近の点だけで決まり、確率モデルを仮定しません。そのため、SVM の出力 $f(\boldsymbol{x})$ はそのままでは事後確率になりません。

## 具体例

**例：6点のハードマージンSVM**

群 $-1$：(0, 0), (1, 0), (0, 1)、群 $+1$：(2, 2), (3, 2), (2, 3) とします。図を描くと、$x_1 + x_2$ が群 $-1$ では最大 1、群 $+1$ では最小 4 なので、対称性から $\boldsymbol{w} = (a, a)$ の形が最適です。

サポートベクトルの候補 (1, 0)（または (0, 1)）と (2, 2) で制約が等号になるとすると

$$
a \cdot 1 + b = -1, \qquad a \cdot 4 + b = 1
$$

より $a = 2/3,\ b = -5/3$ です。判別関数は

$$
f(\boldsymbol{x}) = \frac{2}{3}x_1 + \frac{2}{3}x_2 - \frac{5}{3} = \frac{2}{3}\left(x_1 + x_2 - 2.5\right)
$$

で、境界は $x_1 + x_2 = 2.5$ です。$\|\boldsymbol{w}\| = \frac{2}{3}\sqrt{2} \approx 0.9428$ なので、マージンの幅は $2/\|\boldsymbol{w}\| = 3/\sqrt{2} \approx 2.121$ です。これは直線 $x_1 + x_2 = 1$ と $x_1 + x_2 = 4$ の間の距離 $3/\sqrt{2}$ と一致します。

他の点では $y_i f(\boldsymbol{x}_i)$ が (0, 0) で $5/3$、(3, 2) と (2, 3) で $5/3$ と、すべて1以上で制約を満たします。

双対変数は、対称性から $\alpha_{(1,0)} = \alpha_{(0,1)} = \alpha$、$\alpha_{(2,2)} = 2\alpha$（$\sum \alpha_i y_i = 0$ より）とおくと

$$
\boldsymbol{w} = 2\alpha\begin{pmatrix} 2 \\ 2 \end{pmatrix} - \alpha\begin{pmatrix} 1 \\ 0 \end{pmatrix} - \alpha\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 3\alpha \\ 3\alpha \end{pmatrix}
$$

で、$3\alpha = 2/3$ から $\alpha = 2/9$ です。サポートベクトルは (1, 0), (0, 1), (2, 2) の3点だけです。

新しい点 (1.5, 1.5) は $f = \frac{2}{3}(3 - 2.5) = 1/3 > 0$ で群 $+1$、(2, 0) は $f = \frac{2}{3}(2 - 2.5) = -1/3 < 0$ で群 $-1$ と判別します。

## 現場での使われ方

- **IT・Webサービス**：文章の自動分類（迷惑メール、話題の分類など）で、単語の出現を表す高次元の特徴量に対して線形SVMを使います。変数が多くても安定して動きます。
- **医療**：遺伝子発現データのように、変数の数が標本数よりはるかに多いデータでの病型の判別に使われます。
- **製造・品質管理**：正常データだけから「正常な領域」を囲む境界を学習する1クラスSVMで、異常検知を行います。
- **金融・保険**：審査や不正検知で、カーネルを使った非線形な判別の候補の1つとして、交差検証で他の方法と比較します。

## グラフ・イメージ

実線が分離境界、破線がマージンの境界で、破線の上にある点がサポートベクトルです。
間隔のスライダーを小さくしていくとマージンが狭くなり、負の値にすると2群が重なってハードマージンでは分離できなくなります（ソフトマージンが必要になります）。

## 試験のポイント

- マージンが $1/\|\boldsymbol{w}\|$（幅 $2/\|\boldsymbol{w}\|$）になる理由と、最適化問題の形を書けるようにします。
- サポートベクトルだけで境界が決まること、$\boldsymbol{w} = \sum \alpha_i y_i \boldsymbol{x}_i$ の形を押さえます。
- ソフトマージンの $C$ の意味（大きいほど誤りを許さない）と、ヒンジ損失＋正則化という見方を説明できるようにします。
- 内積だけで書けるのでカーネル法が使える、という点は重要です。

## よくある間違い

- × マージンを $1/\|\boldsymbol{w}\|^2$ や $\|\boldsymbol{w}\|$ とする → ○ 片側のマージンは $1/\|\boldsymbol{w}\|$、両側の幅は $2/\|\boldsymbol{w}\|$ です。
- × $C$ を大きくするとマージンが広がる → ○ $C$ が大きいほどはみ出しへの罰が強く、マージンは狭くなります。
- × すべてのデータ点が境界の位置に影響する → ○ 境界を決めるのはサポートベクトル（$\alpha_i > 0$ の点）だけです。
- × SVM の出力 $f(\boldsymbol{x})$ をそのまま事後確率として使う → ○ $f(\boldsymbol{x})$ は境界からの符号付きの距離に比例する量で、確率ではありません。確率が必要なら別途変換（較正）します。
