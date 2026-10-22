---
id: canonical-variate
title: 正準変量と正準相関係数
reading: せいじゅんへんりょうとせいじゅんそうかんけいすう
en: Canonical variates and canonical correlation coefficients
aliases: [正準変数, 正準相関係数, 正準負荷量, 正準係数]
chapter: 26
importance: A
tags: [正準相関分析, 正準負荷量, バートレット検定, ウィルクスのラムダ]
related: [canonical-correlation-analysis, coefficient-of-determination, principal-component-loading, likelihood-ratio-test, correlation-matrix, chi-squared-distribution]
charts:
  - type: custom
    id: points-2d
    params:
      points:
        - { label: 運動時間, x: 0.9244, y: 0.3815 }
        - { label: 睡眠時間, x: 0.7194, y: -0.6946 }
        - { label: 体力, x: 0.9623, y: -0.2718 }
        - { label: 集中力, x: 0.5480, y: 0.8365 }
      arrows: 1
      circle: 1
      xlabel: 第1正準変量との相関
      ylabel: 第2正準変量との相関
    caption: 例1の正準負荷量。運動時間・睡眠時間は U1, U2 との相関、体力・集中力は V1, V2 との相関。4変数とも第1正準変量と強く相関し、とくに運動時間と体力が第1の軸を作っている。
---

## 一言でいうと

正準変量は、正準相関分析で作る「各変数群の合成変数」、正準相関係数はその合成変数どうしの相関です。
合成の重み（正準係数）と、元の変数との相関（正準負荷量）を見て、2群のつながりの中身を解釈します。

## 定義・数式

$\boldsymbol{x}$（$p$ 変数）と $\boldsymbol{y}$（$q$ 変数）の正準相関分析で、第 $k$ の固有値問題

$$
\Sigma_{11}^{-1}\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\boldsymbol{a}_k=\rho_k^2\boldsymbol{a}_k
$$

の解から作る

$$
U_k=\boldsymbol{a}_k^\top\boldsymbol{x},\qquad V_k=\boldsymbol{b}_k^\top\boldsymbol{y},\qquad \boldsymbol{b}_k=\frac{1}{\rho_k}\Sigma_{22}^{-1}\Sigma_{21}\boldsymbol{a}_k
$$

を**第 $k$ 正準変量**、$\rho_k\ (\ge0)$ を**第 $k$ 正準相関係数**といいます（$k=1,\dots,\min(p,q)$）。係数は $\mathrm{Var}(U_k)=\mathrm{Var}(V_k)=1$ となるように基準化します。

正準変量は次の性質を持ちます（$k\ne l$）。

$$
\mathrm{Corr}(U_k,V_k)=\rho_k,\quad \mathrm{Corr}(U_k,U_l)=\mathrm{Corr}(V_k,V_l)=\mathrm{Corr}(U_k,V_l)=0
$$

**正準負荷量**（構造係数）は元の変数と正準変量の相関で、変数を標準化しておけば

$$
\mathrm{Corr}(\boldsymbol{x},U_k)=R_{11}\boldsymbol{a}_k,\qquad \mathrm{Corr}(\boldsymbol{y},V_k)=R_{22}\boldsymbol{b}_k,\qquad \mathrm{Corr}(\boldsymbol{x},V_k)=\rho_k R_{11}\boldsymbol{a}_k
$$

です。最後のものは**交差負荷量**と呼ばれます。

**正準相関の検定**（多変量正規分布を仮定、$n$ は標本サイズ）：$H_0:\rho_{k+1}=\dots=\rho_{\min(p,q)}=0$ に対し、バートレットの近似

$$
\chi^2=-\Bigl(n-1-\frac{p+q+1}{2}\Bigr)\sum_{j=k+1}^{\min(p,q)}\log(1-\hat\rho_j^2)\ \ \overset{\text{近似}}{\sim}\ \ \chi^2\bigl((p-k)(q-k)\bigr)
$$

を使います。$k=0$ の場合は「2群は無相関（$\Sigma_{12}=O$）」の検定で、$\prod_j(1-\hat\rho_j^2)=|R|/(|R_{11}||R_{22}|)$ は**ウィルクスのラムダ**です。

## 数式の読み解き

- **正準係数** $\boldsymbol{a}_k,\boldsymbol{b}_k$：合成の重みです。重回帰の偏回帰係数と同じく、群の中の変数どうしの相関（多重共線性）の影響を受けて符号や大きさが不安定になります。
- **正準負荷量**：元の変数と正準変量の相関です。解釈には係数よりこちらがよく使われます。$\mathrm{Var}(U_k)=1$ なので $\mathrm{Cov}(\boldsymbol{x},U_k)=R_{11}\boldsymbol{a}_k$ がそのまま相関になります。
- **交差負荷量**：$\mathrm{Cov}(\boldsymbol{x},V_k)=R_{12}\boldsymbol{b}_k$ で、固有値問題の途中式 $R_{12}\boldsymbol{b}_k=\rho_kR_{11}\boldsymbol{a}_k$ から $\rho_k$ 倍の関係が出ます。
- **$\boldsymbol{b}_k$ の式**：ラグランジュ条件 $\Sigma_{21}\boldsymbol{a}=\rho\Sigma_{22}\boldsymbol{b}$ を $\boldsymbol{b}$ について解いたものです。$\boldsymbol{a}_k$ がわかれば、もう一方の固有値問題を解かずに済みます。
- **検定統計量**：$-n\log\Lambda$ 型の尤度比統計量です。$(n-1-\frac{p+q+1}{2})$ は $\chi^2$ 近似を良くするための補正です。

## 具体例

**例1：正準係数と正準負荷量を求める**

正準相関分析の例（運動時間・睡眠時間 と 体力・集中力、$n=100$）では

$$
R_{11}=\begin{pmatrix}1&0.4\\0.4&1\end{pmatrix},\ R_{22}=\begin{pmatrix}1&0.3\\0.3&1\end{pmatrix},\ R_{12}=\begin{pmatrix}0.5&0.3\\0.4&0.2\end{pmatrix}
$$

から $\rho_1=0.5668$, $\boldsymbol{a}_1=(0.7579,\ 0.4162)$ を得ました。確認すると $\boldsymbol{a}_1^\top R_{11}\boldsymbol{a}_1=0.7579^2+0.4162^2+2\times0.4\times0.7579\times0.4162=1.000$ です。

$\boldsymbol{b}_1$ は次のように求めます。

$$
R_{21}\boldsymbol{a}_1=\begin{pmatrix}0.5454\\0.3106\end{pmatrix},\quad
R_{22}^{-1}R_{21}\boldsymbol{a}_1=\begin{pmatrix}0.4970\\0.1615\end{pmatrix},\quad
\boldsymbol{b}_1=\frac{1}{0.5668}\begin{pmatrix}0.4970\\0.1615\end{pmatrix}=\begin{pmatrix}0.877\\0.285\end{pmatrix}
$$

正準負荷量は

$$
R_{11}\boldsymbol{a}_1=\begin{pmatrix}0.924\\0.719\end{pmatrix},\qquad R_{22}\boldsymbol{b}_1=\begin{pmatrix}0.962\\0.548\end{pmatrix}
$$

です。$U_1$ は運動時間と特に強く相関し、$V_1$ は体力と特に強く相関します。交差負荷量は $0.5668\times(0.924,\ 0.719)=(0.524,\ 0.408)$ で、生活習慣の各変数が健康側の合成変数 $V_1$ とどれだけ相関するかを表します。

**例2：何個の正準相関が有意か**

同じ例で $\hat\rho_1^2=0.3212$, $\hat\rho_2^2=0.0016$、補正係数は $n-1-\frac{2+2+1}{2}=96.5$ です。

- $k=0$（すべて0か）：$\chi^2=-96.5\{\log(0.6788)+\log(0.9984)\}=96.5\times0.3891=37.55$。自由度 $2\times2=4$ の上側5%点 $9.488$ を超えるので棄却します。
- $k=1$（$\rho_2=0$ か）：$\chi^2=-96.5\log(0.9984)=0.157$。自由度 $1\times1=1$ の上側5%点 $3.841$ より小さいので棄却しません。

したがって、意味のある正準相関は第1組の1個だけと判断します。ウィルクスのラムダは $0.6788\times0.9984=0.678$ です。

## 現場での使われ方

- **医療・健康**：生活習慣と検査値の正準相関分析で、第1正準変量の負荷量を見て「運動中心の習慣は体力系の検査値と結びつく」のように軸に名前を付けます。
- **教育**：入試科目群と入学後成績群の正準変量を作り、検定で有意な組の数を確認してから、各組の負荷量で「理系の力」「語学の力」などの対応を読み取ります。
- **マーケティング**：顧客属性群と購買行動群の正準変量の得点を顧客ごとに計算し、セグメント分けの軸に使います。

## グラフ・イメージ

例1の正準負荷量を、第1・第2正準変量との相関を座標にして矢印で描いた図です。
矢印が単位円に近いほど、その変数は正準変量でよく表されています。4変数とも横軸方向に大きく伸びており、第1正準変量が2群に共通する主な軸であることがわかります。

## 試験のポイント

- 正準変量は分散1に基準化し、同じ群の中でも群をまたいでも、組が違えば無相関になる性質を押さえます。
- $\boldsymbol{b}_k=\rho_k^{-1}\Sigma_{22}^{-1}\Sigma_{21}\boldsymbol{a}_k$ で、片側の係数からもう片側を求める計算がよく使われます。
- 正準負荷量は $R_{11}\boldsymbol{a}_k$、交差負荷量はその $\rho_k$ 倍です。
- バートレットの検定の自由度 $(p-k)(q-k)$ と補正係数 $n-1-\frac{p+q+1}{2}$ を覚えます。
- ウィルクスのラムダ $\prod(1-\rho_j^2)$ は、2群の無相関の尤度比検定に使われます。

## よくある間違い

- × 正準係数の大きさで変数の重要度を判断する → ○ 群内の変数どうしが相関していると係数は不安定です。解釈には正準負荷量を使います。
- × 第2正準変量 $U_2$ と第1正準変量 $V_1$ は相関してよい → ○ $k\ne l$ なら $\mathrm{Corr}(U_k,V_l)=0$ です。
- × バートレットの検定の自由度を $pq$ のまま使い続ける → ○ $k$ 個を除いた残りを調べるときの自由度は $(p-k)(q-k)$ に減ります。
- × 正準相関係数を負の値で答える → ○ $\boldsymbol{a}$ か $\boldsymbol{b}$ の符号を反転すれば正にできるので、正準相関係数は $0$ 以上で定義します。
