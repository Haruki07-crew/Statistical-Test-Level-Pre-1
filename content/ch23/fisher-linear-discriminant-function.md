---
id: fisher-linear-discriminant-function
title: フィッシャーの線形判別関数
reading: ふぃっしゃーのせんけいはんべつかんすう
en: Fisher's linear discriminant function
aliases: [フィッシャーの判別関数, フィッシャー判別, 正準判別関数]
chapter: 23
importance: A
tags: [判別分析, 群間変動, 群内変動, 一般化固有値問題]
related: [linear-discriminant-analysis, mahalanobis-distance, variance-covariance-matrix, pooled-variance, lagrange-multiplier-pca, one-way-anova]
charts:
  - type: custom
    id: fisher-projection
    params: { angle: 0, rho: 0.6, delta: 2, n: 60 }
    sliders:
      - { param: angle, label: "射影の向き（度）", min: 0, max: 180, step: 5 }
      - { param: rho, label: "群内の相関 ρ", min: -0.9, max: 0.9, step: 0.1 }
    caption: 2群の散布図を、選んだ向きの直線に射影したときの1次元の分布（下段）と、群間変動／群内変動の比。比が最大になる向きがフィッシャーの判別方向で、平均を結ぶ向きとは一般に一致しない。
---

## 一言でいうと

多変量のデータを1本の軸に射影（重み付き和を計算）したとき、2群が**最もよく分かれて見える**向きを選ぶ方法です。
「群の平均どうしは遠く、群の中のばらつきは小さく」なる向きを探します。

## 定義・数式

2群 $G_1, G_2$ の $p$ 変量データがあり、標本平均を $\bar{\boldsymbol{x}}_1, \bar{\boldsymbol{x}}_2$、プールした分散共分散行列を $S$（正則）とします。
係数ベクトル $\boldsymbol{a}$ で1次元に射影した値 $y = \boldsymbol{a}^\top \boldsymbol{x}$ について、

$$
\lambda(\boldsymbol{a}) = \frac{\{\boldsymbol{a}^\top(\bar{\boldsymbol{x}}_1 - \bar{\boldsymbol{x}}_2)\}^2}{\boldsymbol{a}^\top S \boldsymbol{a}}
\qquad \left( = \frac{\text{射影後の平均の差の2乗}}{\text{射影後の群内分散}} \right)
$$

を最大にする $\boldsymbol{a}$ は、定数倍を除いて

$$
\boldsymbol{a} \propto S^{-1}(\bar{\boldsymbol{x}}_1 - \bar{\boldsymbol{x}}_2)
$$

で、最大値は

$$
\max_{\boldsymbol{a}} \lambda(\boldsymbol{a}) = (\bar{\boldsymbol{x}}_1 - \bar{\boldsymbol{x}}_2)^\top S^{-1} (\bar{\boldsymbol{x}}_1 - \bar{\boldsymbol{x}}_2) = D^2
$$

（2群間の**マハラノビス距離の2乗**）です。
**フィッシャーの線形判別関数**は、この $\boldsymbol{a}$ を使った

$$
y(\boldsymbol{x}) = \boldsymbol{a}^\top \left( \boldsymbol{x} - \frac{\bar{\boldsymbol{x}}_1 + \bar{\boldsymbol{x}}_2}{2} \right)
$$

で、$y$ の符号で群を判別します。

分散分析の言葉で書くと、群間平方和 $B$ と群内平方和 $W$ の比 $\boldsymbol{a}^\top B \boldsymbol{a} / \boldsymbol{a}^\top W \boldsymbol{a}$ の最大化です。2群では $B = \frac{n_1 n_2}{n_1+n_2}(\bar{\boldsymbol{x}}_1 - \bar{\boldsymbol{x}}_2)(\bar{\boldsymbol{x}}_1 - \bar{\boldsymbol{x}}_2)^\top$、$W = (n_1+n_2-2)S$ なので、$\lambda(\boldsymbol{a})$ とは定数倍しか違わず、最大にする向きは同じです。

## 数式の読み解き

- 分子：射影した値の2群の平均の差の2乗です。大きいほど群が離れています。
- 分母：射影した値の群内分散です。小さいほど各群がまとまっています。
- 比なので $\boldsymbol{a}$ を何倍しても値は変わりません。向きだけが意味をもちます。

**導出（微分による方法）**　$\boldsymbol{d} = \bar{\boldsymbol{x}}_1 - \bar{\boldsymbol{x}}_2$ とおきます。$\lambda(\boldsymbol{a}) = (\boldsymbol{a}^\top \boldsymbol{d})^2 / (\boldsymbol{a}^\top S \boldsymbol{a})$ を $\boldsymbol{a}$ で微分して $\boldsymbol{0}$ とおくと、

$$
\frac{2(\boldsymbol{a}^\top \boldsymbol{d})\,\boldsymbol{d}\,(\boldsymbol{a}^\top S \boldsymbol{a}) - (\boldsymbol{a}^\top \boldsymbol{d})^2 \, 2S\boldsymbol{a}}{(\boldsymbol{a}^\top S \boldsymbol{a})^2} = \boldsymbol{0}
\;\Longrightarrow\;
S\boldsymbol{a} = \frac{\boldsymbol{a}^\top S \boldsymbol{a}}{\boldsymbol{a}^\top \boldsymbol{d}}\,\boldsymbol{d}
$$

右辺の分数はスカラーなので、$\boldsymbol{a} \propto S^{-1}\boldsymbol{d}$ が得られます。

**導出（コーシー・シュワルツの不等式による方法）**　$S$ は正定値なので $S^{1/2}$ がとれます。

$$
(\boldsymbol{a}^\top \boldsymbol{d})^2 = \{(S^{1/2}\boldsymbol{a})^\top (S^{-1/2}\boldsymbol{d})\}^2 \le (\boldsymbol{a}^\top S \boldsymbol{a})(\boldsymbol{d}^\top S^{-1} \boldsymbol{d})
$$

両辺を $\boldsymbol{a}^\top S \boldsymbol{a}$ で割れば $\lambda(\boldsymbol{a}) \le D^2$ です。等号は $S^{1/2}\boldsymbol{a} \propto S^{-1/2}\boldsymbol{d}$、つまり $\boldsymbol{a} \propto S^{-1}\boldsymbol{d}$ のときに成り立ちます。

この方法は**正規分布を仮定していない**点が重要です。それでも結果は、等分散共分散の正規分布のもとでのベイズ判別（線形判別分析）の係数と一致します。

3群以上では、$W^{-1}B$ の固有値問題 $B\boldsymbol{a} = \lambda W \boldsymbol{a}$ を解き、大きい固有値に対応する固有ベクトルを順に判別軸とします（正準判別分析）。

## 具体例

**例1：射影の向きを比べる**

線形判別分析の例と同じ、良品 $G_1$ と不良品 $G_2$ の4個ずつのデータを使います。

| 群 | 個体 $(x_1, x_2)$ |
|---|---|
| $G_1$ | (1, 2), (4, 5), (1, 5), (2, 4) |
| $G_2$ | (3, 5), (6, 8), (6, 5), (5, 6) |

比は $\boldsymbol{d}$ の符号に影響されないので、ここでは $\boldsymbol{d} = \bar{\boldsymbol{x}}_2 - \bar{\boldsymbol{x}}_1 = (3, 2)^\top$ とします。また $S = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ です。いくつかの向きで $\lambda(\boldsymbol{a})$ を計算します。

| 向き $\boldsymbol{a}$ | $(\boldsymbol{a}^\top \boldsymbol{d})^2$ | $\boldsymbol{a}^\top S \boldsymbol{a}$ | $\lambda(\boldsymbol{a})$ |
|---|---|---|---|
| $(1, 0)$：$x_1$ だけ | 9 | 2 | 4.500 |
| $(0, 1)$：$x_2$ だけ | 4 | 2 | 2.000 |
| $(1, 1)$：単純な合計 | 25 | 6 | 4.167 |
| $(4, 1)$：$S^{-1}\boldsymbol{d}$ の3倍 | 196 | 42 | 4.667 |

$S^{-1}\boldsymbol{d} = \frac{1}{3}(4, 1)^\top$ の向きで最大になり、その値 $14/3 \approx 4.667$ は $D^2 = \boldsymbol{d}^\top S^{-1}\boldsymbol{d} = (3 \cdot 4 + 2 \cdot 1)/3 = 14/3$ と一致します。

**例2：射影した値で確かめる**

$y = 4x_1 + x_2$ を計算すると、$G_1$ は 6, 21, 9, 12（平均 12）、$G_2$ は 17, 32, 29, 26（平均 26）です。
各群の偏差平方和はどちらも $36 + 81 + 9 + 0 = 126$ なので、プールした分散は $(126 + 126)/6 = 42$ で、表の $\boldsymbol{a}^\top S \boldsymbol{a}$ と一致します。
平均の差は 14 なので $\lambda = 14^2 / 42 = 4.667$ です。判別の境界は平均の中点 $y = 19$ で、$4x_1 + x_2 > 19$ なら $G_2$ と判別します。

## 現場での使われ方

- **医療**：多数の検査項目を1つの「判別得点」にまとめ、得点の大きさで疾患の可能性を評価します。得点は1次元なので、しきい値を決めやすくなります。
- **行政・調査**：アンケートの多くの設問から、回答者のグループ（例えば利用者と非利用者）を最もよく分ける設問の組み合わせを探します。
- **IT・Webサービス**：画像や音声の特徴量を、クラスがよく分かれる少数の軸に圧縮する「教師ありの次元削減」に使われます。主成分分析と違い、群の情報を使って軸を選びます。

## グラフ・イメージ

射影の向きのスライダーを回すと、下段の1次元の分布の重なり具合と、群間／群内の比が変わります。
比が最大になる向きでは、2つの山が最もはっきり分かれます。
群内の相関 $\rho$ を0から離すと、最適な向きが「平均を結ぶ向き」からずれていく様子が見えます。

## 試験のポイント

- 「分子＝射影後の平均差の2乗、分母＝射影後の群内分散」という比を書き、微分またはコーシー・シュワルツの不等式で $\boldsymbol{a} \propto S^{-1}\boldsymbol{d}$ を導けるようにします。
- 最大値が $D^2$（マハラノビス距離の2乗）になることは計算問題でよく使います。
- 正規性を仮定せずに導かれるが、正規・等分散共分散のもとのベイズ判別と同じ係数になる、という関係を押さえます。
- 3群以上では一般化固有値問題 $B\boldsymbol{a} = \lambda W\boldsymbol{a}$ になり、判別軸は最大 $\min(K-1, p)$ 本です。
- 主成分分析（全体の分散を最大化、群の情報を使わない）との違いを説明できるようにします。

## よくある間違い

- × 比の分母に全体の分散（群を区別しない分散）を使う → ○ 分母は群内分散です。全体の分散を使っても最適な向きは同じになりますが、比の値と意味が変わります。
- × 係数ベクトルの大きさにも意味があると考える → ○ 比は $\boldsymbol{a}$ の定数倍で変わらないので、意味があるのは向き（係数の比）だけです。
- × 最適な向きは平均を結ぶ向き $\boldsymbol{d}$ と同じ → ○ $S^{-1}$ が掛かるので、変数に相関や分散の違いがあればずれます。例1では $\boldsymbol{d} = (3, 2)$ の向き（$\lambda = 169/38 \approx 4.447$ にとどまる）ではなく $(4, 1)$ の向きが最適です。
- × 主成分の第1軸で判別すればよい → ○ 第1主成分は全体のばらつきが最大の向きであり、群を分ける向きとは限りません。
