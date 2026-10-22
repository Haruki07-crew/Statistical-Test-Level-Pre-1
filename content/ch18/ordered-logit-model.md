---
id: ordered-logit-model
title: 順序ロジットモデル
reading: じゅんじょろじっともでる
en: Ordered logit model
aliases: [比例オッズモデル, 累積ロジットモデル, 順序ロジスティック回帰, "proportional odds model", 順序プロビットモデル]
chapter: 18
importance: B
tags: [質的回帰, 順序カテゴリ, ロジスティック回帰, 潜在変数]
related: [multinomial-logit-model, logistic-regression, odds-and-odds-ratio, probit-model, ordinal-contingency-table, logit-logistic-transformation]
charts:
  - type: custom
    id: line-data
    params: { x: [-3, -2, -1, 0, 1, 2, 3, 4, 5], series: [{ label: "不満（Y=1）", y: [0.8022, 0.6457, 0.4502, 0.2689, 0.1419, 0.0691, 0.0323, 0.0148, 0.0067] }, { label: "普通（Y=2）", y: [0.1655, 0.2852, 0.408, 0.4621, 0.408, 0.2852, 0.1655, 0.085, 0.0407] }, { label: "満足（Y=3）", y: [0.0323, 0.0691, 0.1419, 0.2689, 0.4502, 0.6457, 0.8022, 0.9002, 0.9526] }], xlabel: "サポート対応の評価点 x", ylabel: "各カテゴリの確率" }
    caption: 例1のモデル（α1=−1, α2=1, β=0.8）の各カテゴリの確率。x が増えると確率の山が「不満 → 普通 → 満足」と順に右へ移る。
---

## 一言でいうと

順序ロジットモデルは、「不満・普通・満足」や「軽症・中等症・重症」のように、順序のあるカテゴリを応答とする回帰モデルです。
「ある段階以下になる確率」の対数オッズを、説明変数の一次式で表します。

## 定義・数式

応答 $Y$ が順序のあるカテゴリ $1<2<\cdots<K$ をとるとします。累積確率 $\gamma_j(\boldsymbol x)=P(Y\le j\mid\boldsymbol x)$ について、

$$
\operatorname{logit}\gamma_j(\boldsymbol x)=\log\frac{P(Y\le j\mid\boldsymbol x)}{P(Y>j\mid\boldsymbol x)}=\alpha_j-\boldsymbol x^\top\boldsymbol\beta\qquad(j=1,\dots,K-1)
$$

とするモデルを**順序ロジットモデル**（累積ロジットモデル）といいます。切片（閾値）は $\alpha_1<\alpha_2<\cdots<\alpha_{K-1}$ で、傾き $\boldsymbol\beta$ はすべての $j$ で共通です。

各カテゴリの確率は累積確率の差です。

$$
P(Y=j\mid\boldsymbol x)=\gamma_j(\boldsymbol x)-\gamma_{j-1}(\boldsymbol x),\qquad \gamma_0=0,\ \gamma_K=1
$$

パラメータの数は $(K-1)+p$ 個で、多項ロジットモデルの $(K-1)(p+1)$ 個より少なくなります。

なお、$\operatorname{logit}\gamma_j=\alpha_j+\boldsymbol x^\top\boldsymbol\beta$ と符号をプラスで書く流儀もあります。その場合は $\beta>0$ が「低いカテゴリになりやすい」ことを意味するので、ソフトの出力を読むときは定義を確認します。ここでは「$\beta>0$ なら高いカテゴリになりやすい」マイナスの書き方を採用します。

## 数式の読み解き

**潜在変数による導出**

観測できない連続的な量（満足度の本音など）を

$$
Y^*=\boldsymbol x^\top\boldsymbol\beta+\varepsilon,\qquad \varepsilon\sim\text{標準ロジスティック分布}
$$

とし、閾値 $\alpha_1<\cdots<\alpha_{K-1}$ で区切って、$\alpha_{j-1}<Y^*\le\alpha_j$ なら $Y=j$ とします。すると

$$
P(Y\le j)=P(Y^*\le\alpha_j)=P(\varepsilon\le\alpha_j-\boldsymbol x^\top\boldsymbol\beta)=\frac{1}{1+e^{-(\alpha_j-\boldsymbol x^\top\boldsymbol\beta)}}
$$

となり、順序ロジットモデルが得られます。$\varepsilon$ を標準正規分布にすれば**順序プロビットモデル**です。

**比例オッズ性**

2つの説明変数の値 $\boldsymbol x_1,\boldsymbol x_2$ で累積オッズの比をとると

$$
\frac{\text{odds}(Y\le j\mid\boldsymbol x_1)}{\text{odds}(Y\le j\mid\boldsymbol x_2)}=\exp\{-(\boldsymbol x_1-\boldsymbol x_2)^\top\boldsymbol\beta\}
$$

で、$j$ によりません。どこで「低い／高い」を区切っても、オッズ比が同じになります。これを**比例オッズ**の仮定といい、このモデルを比例オッズモデルとも呼びます。

- $\alpha_j$：区切り位置。$K-1$ 個あり、増加列でなければなりません。
- $\beta_k$：$x_k$ が1増えると、すべての $j$ で「$j$ より上になる」オッズが $e^{\beta_k}$ 倍になります。

## 具体例

**例1：満足度の3段階**

顧客満足度（1：不満、2：普通、3：満足）を、サポート対応の評価点 $x$ で説明するモデルが

$$
\operatorname{logit}P(Y\le1)=-1-0.8x,\qquad \operatorname{logit}P(Y\le2)=1-0.8x
$$

と推定されたとします（$\alpha_1=-1$、$\alpha_2=1$、$\beta=0.8$）。

$x=0$ のとき：$P(Y\le1)=1/(1+e^{1})=0.2689$、$P(Y\le2)=1/(1+e^{-1})=0.7311$ なので、

$$
P(Y=1)=0.2689,\quad P(Y=2)=0.7311-0.2689=0.4621,\quad P(Y=3)=1-0.7311=0.2689
$$

$x=2$ のとき：$\alpha_j-0.8\times2$ は $-2.6$ と $-0.6$ なので $P(Y\le1)=0.0691$、$P(Y\le2)=0.3543$。

$$
P(Y=1)=0.0691,\quad P(Y=2)=0.2852,\quad P(Y=3)=0.6457
$$

比例オッズを確かめます。$P(Y\le1)$ のオッズは $x=0$ で $e^{-1}=0.3679$、$x=2$ で $0.0743$。$P(Y\le2)$ のオッズは $e^{1}=2.7183$ と $0.5488$。どちらも比は $e^{-1.6}\approx0.2019$ で同じです。
言い換えると、$x$ が2増えると「普通以上」「満足」のどちらのオッズも $e^{1.6}\approx4.953$ 倍になります。

## 現場での使われ方

- **マーケティング**：5段階評価のアンケート（とても不満〜とても満足）を、サービスの属性で説明します。平均点を回帰するより、カテゴリの性質を正しく扱えます。
- **医療**：重症度（軽症・中等症・重症）や、痛みの段階を応答として、治療や危険因子の効果を1つの共通オッズ比で表します。
- **金融・保険**：格付け（AAA, AA, A, …）の推定で、財務指標から格付けの段階を説明する順序ロジット・順序プロビットが使われます。

## グラフ・イメージ

例1のモデルで、$x$ に対する3つのカテゴリの確率を描いた図です。$x$ が大きくなるにつれて、確率の山が「不満 → 普通 → 満足」と順に移ります。
中間の「普通」は山形で、ピークは2つの閾値の中央 $x=(\alpha_1+\alpha_2)/(2\beta)=0$ 付近にあります。

## 試験のポイント

- 累積確率 $P(Y\le j)$ から差をとってカテゴリ確率を求める計算が典型です。
- 傾きがすべての $j$ で共通（比例オッズ）であること、そのため累積オッズ比が $j$ によらないことを説明できるようにします。
- パラメータ数は $(K-1)+p$ で、多項ロジットより少なくなります。
- 潜在変数＋閾値の考え方で導けること、誤差を正規にすれば順序プロビットになることを押さえます。
- 符号の流儀（$\alpha_j-\boldsymbol x^\top\boldsymbol\beta$ か $\alpha_j+\boldsymbol x^\top\boldsymbol\beta$ か）で解釈が逆になるので、問題文の定義を確認します。

## よくある間違い

- × カテゴリに 1, 2, 3 と点数をつけて線形回帰すれば十分 → ○ 間隔が等しいという根拠のない仮定が入り、予測値がカテゴリの範囲を外れることもあります。
- × $P(Y=j)$ を直接 $1/(1+e^{-(\alpha_j-\boldsymbol x^\top\boldsymbol\beta)})$ とする → ○ その式は累積確率 $P(Y\le j)$ です。カテゴリ確率は隣どうしの差をとります。
- × 閾値ごとに別の傾きを推定してもよい → ○ それは比例オッズの仮定を外した別のモデル（部分比例オッズモデルなど）です。傾きが閾値ごとに違うと、累積確率が交差して負の確率が出ることもあります。
