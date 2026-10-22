---
id: principal-component-analysis
title: 主成分分析
reading: しゅせいぶんぶんせき
en: Principal component analysis
aliases: [PCA, 主成分解析]
chapter: 22
importance: A
tags: [多変量解析, 次元削減, 固有値問題, 教師なし学習]
related: [principal-component, eigenvalue-and-eigenvector, contribution-ratio, pca-on-correlation-matrix, variance-covariance-matrix, factor-analysis-model]
charts:
  - type: custom
    id: pca-axes
    params: { rho: 0.8, sx: 1, sy: 1, n: 200 }
    sliders:
      - { param: rho, label: "相関係数 ρ", min: -0.95, max: 0.95, step: 0.05 }
      - { param: sx, label: "x1 の標準偏差", min: 0.5, max: 3, step: 0.1 }
    caption: 2変数の散布図に、第1主成分（ばらつきが最大の方向）と第2主成分（それに直交する方向）の軸を重ねた図。線の長さは各主成分の標準偏差の2倍。
  - type: custom
    id: bar-data
    params:
      labels: [第1主成分, 第2主成分, 第3主成分, 第4主成分]
      values: [3.2501, 0.6287, 0.1012, 0.0200]
      xlabel: 主成分
      ylabel: 固有値（主成分の分散）
      cumulative: true
    caption: 具体例の4教科データ（相関行列）の固有値と累積寄与率。第1主成分だけで約81%、第2主成分までで約97%を説明する。
---

## 一言でいうと

たくさんの変数を、情報（ばらつき）をなるべく失わずに、少数の「合成変数」にまとめる手法です。
合成変数は元の変数の重み付き和で、互いに無相関になるように作ります。

## 定義・数式

$p$ 個の変数をまとめた確率ベクトル $\boldsymbol{X}=(X_1,\dots,X_p)^\top$ の分散共分散行列を $\Sigma$ とします。
係数ベクトル $\boldsymbol{a}$ による線形結合 $Y=\boldsymbol{a}^\top\boldsymbol{X}$ の分散は $V[Y]=\boldsymbol{a}^\top\Sigma\boldsymbol{a}$ です。

**第1主成分**は、制約 $\boldsymbol{a}^\top\boldsymbol{a}=1$ のもとで $V[Y]$ を最大にする線形結合です。

$$
\boldsymbol{a}_1=\mathop{\arg\max}_{\boldsymbol{a}^\top\boldsymbol{a}=1}\ \boldsymbol{a}^\top\Sigma\boldsymbol{a}
$$

**第 $k$ 主成分**は、それまでの係数ベクトルすべてと直交する（$\boldsymbol{a}^\top\boldsymbol{a}_j=0,\ j<k$）という条件を加えて、分散を最大にするものです。

答えは $\Sigma$ の固有値問題で与えられます。$\Sigma$ の固有値を $\lambda_1\ge\lambda_2\ge\dots\ge\lambda_p\ge0$、対応する長さ1の固有ベクトルを $\boldsymbol{h}_1,\dots,\boldsymbol{h}_p$ とすると、

$$
\Sigma\boldsymbol{h}_k=\lambda_k\boldsymbol{h}_k,\qquad Y_k=\boldsymbol{h}_k^\top\boldsymbol{X},\qquad V[Y_k]=\lambda_k,\qquad \mathrm{Cov}[Y_j,Y_k]=0\ (j\ne k)
$$

です。データから行うときは $\Sigma$ の代わりに標本分散共分散行列 $S$（または相関行列 $R$）を使います。

## 数式の読み解き

- $\boldsymbol{a}^\top\boldsymbol{a}=1$：係数の長さをそろえる制約です。これがないと、係数を何倍にもすれば分散はいくらでも大きくなってしまいます。
- $\boldsymbol{a}^\top\Sigma\boldsymbol{a}$：データを方向 $\boldsymbol{a}$ に射影したときの分散です。「その方向から見たときの広がり」と考えます。
- 固有値 $\lambda_k$：第 $k$ 主成分の分散そのものです。大きいほど多くの情報を運んでいます。

導出の筋道は、ラグランジュ乗数法です。

$$
L(\boldsymbol{a},\lambda)=\boldsymbol{a}^\top\Sigma\boldsymbol{a}-\lambda(\boldsymbol{a}^\top\boldsymbol{a}-1)
$$

を $\boldsymbol{a}$ で微分して0とおくと $2\Sigma\boldsymbol{a}-2\lambda\boldsymbol{a}=\boldsymbol{0}$、つまり $\Sigma\boldsymbol{a}=\lambda\boldsymbol{a}$ です。
このとき分散は $\boldsymbol{a}^\top\Sigma\boldsymbol{a}=\lambda\boldsymbol{a}^\top\boldsymbol{a}=\lambda$ なので、最大の固有値を選べば分散最大になります。

さらに、固有値の和は対角成分の和（トレース）に等しいので、

$$
\sum_{k=1}^{p}\lambda_k=\mathrm{tr}\,\Sigma=\sum_{j=1}^{p}V[X_j]
$$

が成り立ちます。主成分分析は「全体の分散（総分散）を、無相関な成分に配り直す」操作だと言えます。
幾何学的には、データの雲に合わせて座標軸を回転させ、いちばん長い方向を第1軸にとり直しています。

## 具体例

**例1：2変数を手計算で**

5人のデータ $x_1=(7,9,10,11,13)$、$x_2=(3,5,5,7,5)$ の標本分散共分散行列（$n-1$ で割る）は

$$
S=\begin{pmatrix}5&2\\2&2\end{pmatrix}
$$

です。固有方程式 $(5-\lambda)(2-\lambda)-4=\lambda^2-7\lambda+6=0$ より $\lambda_1=6,\ \lambda_2=1$。
第1主成分は $\boldsymbol{h}_1=(2,1)^\top/\sqrt5\approx(0.894,0.447)^\top$ で、総分散 $7$ のうち $6/7\approx0.857$ を説明します。

**例2：4教科のテスト（numpy で計算）**

8人の生徒の得点です。

| 生徒 | 国語 | 英語 | 数学 | 理科 |
|---|---|---|---|---|
| 1 | 72 | 68 | 60 | 58 |
| 2 | 85 | 80 | 70 | 72 |
| 3 | 60 | 66 | 78 | 80 |
| 4 | 90 | 88 | 92 | 90 |
| 5 | 55 | 58 | 50 | 52 |
| 6 | 78 | 70 | 85 | 82 |
| 7 | 66 | 72 | 64 | 60 |
| 8 | 82 | 86 | 74 | 78 |

教科ごとに満点が同じでも散らばりが違うので、相関行列 $R$ から主成分を求めます（`numpy.linalg.eigh(R)`）。

$$
R=\begin{pmatrix}1&0.902&0.663&0.656\\0.902&1&0.638&0.664\\0.663&0.638&1&0.976\\0.656&0.664&0.976&1\end{pmatrix}
$$

固有値は $3.250,\ 0.629,\ 0.101,\ 0.020$（合計4）です。

- 第1主成分 $\boldsymbol{h}_1\approx(0.494,\ 0.492,\ 0.506,\ 0.509)$：4教科をほぼ均等に足した「総合力」。寄与率 $3.250/4\approx0.8125$。
- 第2主成分 $\boldsymbol{h}_2\approx(0.499,\ 0.515,\ -0.501,\ -0.484)$：国語・英語が高く数学・理科が低いほど大きい「文系－理系」の軸。寄与率 $0.629/4\approx0.1572$。

第2主成分までの累積寄与率は $0.9697$ で、4つの得点を2つの数でほぼ表せることがわかります。

## 現場での使われ方

- **マーケティング**：顧客アンケートの数十項目を少数の主成分にまとめ、「価格重視」「品質重視」のような軸で顧客を地図上に並べます。
- **製造・品質管理**：多数のセンサー値を主成分に圧縮し、第1・第2主成分の得点の推移を監視して異常を早く見つけます。
- **医療**：多数の血液検査値や遺伝子発現量を少数の成分にまとめ、患者群の違いを2次元の図で確認します。
- **金融・保険**：多数の年限の金利の変動を主成分に分けると、「水準」「傾き」「曲がり」の3成分でほとんどを説明でき、リスク管理に使われます。

## グラフ・イメージ

相関 $\rho$ を0に近づけると雲が丸くなり、2本の軸の長さ（固有値）がそろって、次元を減らす意味が薄れます。
$|\rho|$ を大きくすると雲が細長くなり、第1主成分の寄与率が1に近づきます。
$x_1$ の標準偏差だけを大きくすると、第1主成分の軸が $x_1$ 方向へ倒れていきます。これが単位に依存する性質で、相関行列から行う理由になります。

## 試験のポイント

- 「分散最大化 → ラグランジュ乗数法 → 固有値問題」の流れを説明できるようにします。
- 第 $k$ 主成分の分散は第 $k$ 固有値、固有値の和は総分散（相関行列なら変数の個数 $p$）です。
- 2×2 の行列の固有値・固有ベクトルを手で求めさせる問題は重要です。$\lambda^2-(\mathrm{tr})\lambda+\det=0$ を使います。
- 寄与率・累積寄与率・主成分負荷量・主成分得点の計算はセットで出題されます。
- 共分散行列からか相関行列からかで結果が変わる点を、理由とともに押さえます。

## よくある間違い

- × 主成分分析と因子分析は同じもの → ○ 主成分分析は観測変数の線形結合で合成変数を作る「記述的な要約」、因子分析は観測変数を潜在因子で説明する「モデル」です。
- × 固有ベクトルの符号は一意に決まる → ○ $\boldsymbol{h}$ と $-\boldsymbol{h}$ はどちらも固有ベクトルです。ソフトによって符号が逆になっても、解釈が反転するだけで誤りではありません。
- × 単位の違う変数をそのまま共分散行列で分析する → ○ 分散の大きい変数が第1主成分を独占します。単位が異なるときは相関行列（標準化したデータ）を使います。
- × 主成分は目的変数をうまく予測する方向 → ○ 主成分は説明変数のばらつきだけで決まり、目的変数は一切使いません。
