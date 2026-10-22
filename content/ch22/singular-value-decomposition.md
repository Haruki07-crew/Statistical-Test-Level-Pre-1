---
id: singular-value-decomposition
title: 特異値分解
reading: とくいちぶんかい
en: Singular value decomposition
aliases: [SVD, 特異値, 特異ベクトル]
chapter: 22
importance: B
tags: [線形代数, 行列分解, 主成分分析, 低ランク近似]
related: [eigenvalue-and-eigenvector, principal-component-score, low-rank-approximation, principal-component-analysis, classical-mds, ridge-regression]
charts:
  - type: custom
    id: bar-data
    params:
      labels: [第1, 第2, 第3, 第4]
      values: [3614.95, 664.33, 91.51, 25.08]
      xlabel: 特異値の番号
      ylabel: 特異値の2乗 d²
      cumulative: true
    caption: 4教科データの中心化行列の特異値の2乗（棒）と、その累積割合（折れ線）。特異値の2乗を n−1=7 で割ると共分散行列の固有値になり、累積割合は累積寄与率と一致する。
---

## 一言でいうと

どんな長方形の行列も「回転 × 伸び縮み × 回転」の3つの行列の積に分解する方法です。
中心化したデータ行列を特異値分解すると、分散共分散行列を作らずに主成分分析の結果がすべて得られます。

## 定義・数式

$n\times p$ の実行列 $X$（階数 $r$）は、次の形に分解できます。

$$
X=UDV^\top=\sum_{k=1}^{r}d_k\,\boldsymbol{u}_k\boldsymbol{v}_k^\top
$$

- $U=(\boldsymbol{u}_1,\dots,\boldsymbol{u}_r)$：$n\times r$、$U^\top U=I$（**左特異ベクトル**）
- $V=(\boldsymbol{v}_1,\dots,\boldsymbol{v}_r)$：$p\times r$、$V^\top V=I$（**右特異ベクトル**）
- $D=\mathrm{diag}(d_1,\dots,d_r)$、$d_1\ge\dots\ge d_r>0$（**特異値**）

固有値分解との関係は次のとおりです。

$$
X^\top X=VD^2V^\top,\qquad XX^\top=UD^2U^\top
$$

つまり、$\boldsymbol{v}_k$ は $X^\top X$ の固有ベクトル、$d_k^2$ はその固有値です。

## 数式の読み解き

$X$ を中心化したデータ行列 $X_c$ とすると、標本分散共分散行列は $S=X_c^\top X_c/(n-1)$ です。したがって

$$
S=V\,\frac{D^2}{n-1}\,V^\top
$$

となり、主成分分析との対応は次のとおりです。

- 主成分の係数（$S$ の固有ベクトル）：$\boldsymbol{h}_k=\boldsymbol{v}_k$
- 主成分の分散（$S$ の固有値）：$\lambda_k=d_k^2/(n-1)$
- 主成分得点：$Z=X_cV=UDV^\top V=UD$、つまり $z_{ik}=d_k u_{ik}$

$X_c^\top X_c$ をわざわざ計算しないので、桁落ちなどの数値誤差に強く、多くのソフトは内部で特異値分解を使って主成分を計算しています。

さらに、上位 $m$ 項だけを残した $X_m=\sum_{k=1}^{m}d_k\boldsymbol{u}_k\boldsymbol{v}_k^\top$ は、階数 $m$ 以下の行列のなかで $X$ との差の2乗和を最小にします（エッカート・ヤングの定理）。そのときの誤差は $\sum_{k>m}d_k^2$ です。

## 具体例

**例1：5×2 のデータ行列を分解する**

$x_1=(7,9,10,11,13)$、$x_2=(3,5,5,7,5)$ を中心化した行列は

$$
X_c=\begin{pmatrix}-3&-2\\-1&0\\0&0\\1&2\\3&0\end{pmatrix},\qquad X_c^\top X_c=\begin{pmatrix}20&8\\8&8\end{pmatrix}
$$

です。$X_c^\top X_c$ の固有値は $\lambda^2-28\lambda+96=0$ より $24,\ 4$ なので、特異値は $d_1=\sqrt{24}\approx4.899$、$d_2=2$ です。
$n-1=4$ で割ると $6,\ 1$ となり、$S$ の固有値と一致します。右特異ベクトルは $\boldsymbol{v}_1=(2,1)^\top/\sqrt5$、$\boldsymbol{v}_2=(1,-2)^\top/\sqrt5$ です。
左特異ベクトルは $\boldsymbol{u}_1=X_c\boldsymbol{v}_1/d_1=(-8,-2,0,4,6)^\top/\sqrt{120}\approx(-0.730,\ -0.183,\ 0,\ 0.365,\ 0.548)^\top$ で、$d_1\boldsymbol{u}_1$ は主成分得点 $(-3.578,\ -0.894,\ 0,\ 1.789,\ 2.683)$ に一致します。

第1項だけで近似した $d_1\boldsymbol{u}_1\boldsymbol{v}_1^\top$ と $X_c$ の差の2乗和は、numpy で計算すると $4=d_2^2$ です。

**例2：4教科のテスト（numpy）**

主成分分析の項目の8×4 データを中心化して `numpy.linalg.svd` にかけると、特異値は $60.12,\ 25.77,\ 9.57,\ 5.01$ でした。2乗して $n-1=7$ で割ると $516.42,\ 94.90,\ 13.07,\ 3.58$ で、共分散行列の固有値と一致します。

## 現場での使われ方

- **IT・Webサービス**：利用者×商品の評価行列を特異値分解して低ランクで近似し、まだ評価していない商品の評価を予測する推薦システムに使われます。
- **IT・Webサービス**：文書×単語の行列を特異値分解し、少数の「話題」の次元で文書の類似度を測る手法（潜在意味解析）があります。
- **製造・品質管理**：画像やスペクトルなどの高次元データを、上位の特異値の成分だけで圧縮・ノイズ除去します。

## グラフ・イメージ

棒は特異値の2乗、折れ線はその累積割合です。
特異値の2乗の割合が、そのまま主成分の寄与率になっていることを確認してください。第2項までで全体の97%以上を占めます。

## 試験のポイント

- 「$X_c^\top X_c$ の固有値 $=d_k^2$」「$\lambda_k=d_k^2/(n-1)$」「主成分得点 $=UD$」の3つの対応を押さえます。
- 特異値は非負で、固有値（$S$ の）とは2乗と $n-1$ の分だけ違います。
- 低ランク近似の誤差が捨てた特異値の2乗和になる点は、累積寄与率の解釈とつながります。

## よくある間違い

- × 特異値をそのまま主成分の分散とする → ○ 分散は $d_k^2/(n-1)$ です（$n$ で割る流儀なら $d_k^2/n$）。
- × 中心化せずに特異値分解して主成分とする → ○ 平均を引かないと、第1特異ベクトルは平均の方向を向いてしまい、主成分とは一致しません。
