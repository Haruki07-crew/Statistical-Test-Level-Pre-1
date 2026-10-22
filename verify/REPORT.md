# 数値検算レポート

## 第1章（事象と確率）

- スクリプト: `verify/ch01.py`（実行: `.venv/bin/python verify/ch01.py`）
- 対象: bayes-theorem, conditional-independence, conditional-probability, independence-of-events, law-of-total-probability
- 結果: **OK 54 / NG 0**
- 方法: Fraction による厳密計算、サイコロ・コイン・玉の全列挙、sympy（逐次更新式＝一括ベイズ、補事象の独立）、乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.0005〜0.01）
- 本文の修正: なし
- 検算対象外: 定性的な記述（現場での使われ方、多変量正規で偏相関0⇔条件付き独立 など他章で扱う主張）


## 第3章（期待値・分散・相関などの基本量）

- スクリプト: `verify/ch03.py`（実行: `.venv/bin/python verify/ch03.py`）
- 対象: coefficient-of-variation, conditional-expectation, conditional-variance, correlation-matrix, covariance-and-correlation, expectation-variance-properties, mean-vector, partial-correlation-coefficient, quantile-function, skewness-kurtosis, spurious-correlation, variance-covariance-matrix, weighted-geometric-harmonic-mean（13ファイル）
- 結果: **OK 105 / NG 0**
- 方法: Fraction による厳密計算（同時確率表・サイコロ）、sympy（条件付き期待値・全分散の積分、指数分布の分位点、最小分散の微分、AM-GM の恒等式）、numpy/scipy.stats（分布表の歪度・尖度・CV、分位点、逆行列・固有値）、乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.003〜3、標準誤差の数倍に設定）
- 本文の修正: なし
- 検算スクリプト側の修正: Cov[AX,BY] のシミュレーションの許容誤差を 0.05→0.15（標準誤差 約0.03 の約4.5倍）に変更し、標本共分散での厳密一致チェックを追加
- 検算対象外: 定性的な記述（現場での使われ方、アイスクリームと水難事故の例、最頻値<中央値<平均の傾向など）

## 第2章（確率分布と母関数）

- スクリプト: `verify/ch02.py`（実行: `.venv/bin/python verify/ch02.py`）
- 対象: characteristic-function, conditional-probability-density-function, conditional-probability-function, cumulative-distribution-function, hazard-function, joint-probability-density-function, joint-probability-function, marginal-distribution, marginal-probability-density-function, marginal-probability-function, moment-generating-function, probability-generating-function, survival-function（13ファイル）
- 結果: **OK 139 / NG 0**
- 方法: sympy（二項・ポアソン・正規の MGF の導出と1・2階微分、キュムラント母関数 K',K'',K'''、PGF の導出と G'(1),G''(1)、幾何・負の二項、確率の取り出し G^(k)(0)/k!、特性関数 φ=M(it)・一様分布・指数分布の φ'(0)、ワイブルのハザード／累積ハザード、2変量正規の条件付き分布の恒等式 f=f_X·f(y|x)、各種重積分）、scipy（ガンマ・χ²・コーシーの母関数を数値積分、正規・二項・多項の確率）、Fraction（同時確率表・条件付き分散の分解）、乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.001〜0.15）
- 本文の修正: なし
- 検算スクリプト側の修正: 初回 NG 11 件はすべて検算側の問題（sympy が記号のべき・Piecewise を簡約できない、t>0 の MGF 数値積分で inf×0、コーシーの振動積分の精度）。有限 n での記号確認・QAWF（フーリエ積分）・有限上限の数値積分に切り替えて解消
- 検算対象外: 定性的な記述（現場での使われ方、一意性・反転公式・レヴィの連続性定理などの定理そのもの、対数正規分布がモーメントで決まらないこと、MGF が存在しない分布の列挙）

## 第4章（変数変換とデータ変換）

- スクリプト: `verify/ch04.py`（実行: `.venv/bin/python verify/ch04.py`）
- 対象: change-of-variables, box-cox-transformation, log-transformation, logit-logistic-transformation, power-transformation, probit-transformation（6ファイル）
- 結果: **OK 85 / NG 0**
- 方法: sympy（1変数の逆変換と |dx/dy|、Z² の2枝の和、2変数のヤコビ行列と J=-u、Box-Cox の λ→0 極限と導関数、ロジット/ロジスティックの逆関数・微分、デルタ法、λ=1-β の導出）、scipy（χ²・ガンマ・対数正規・ロジスティック・正規の分布関数/分位点、ポアソンの V[√X] を確率関数から厳密計算）、Box-Cox の例2は表の σ̂²(λ)・ℓp(λ) を再計算し、プロファイル対数尤度を数値最大化して λ̂=0.03868 を得て scipy.stats.boxcox とも一致を確認。乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.002〜0.02）
- 本文の修正: なし
- 検算スクリプト側の修正: 対数変換の例3（CV 0.2 の2群で log 後 SD≈0.2）を許容 0.01 で照合して初回 NG（実測 0.212）。本文はデルタ法による1次近似としての主張なので、デルタ法の値 0.2 の厳密確認と、シミュレーションでの「2群の SD が一致・0.2 に近い（許容 0.02）」の2チェックに分けて解消
- 検算対象外: 定性的な記述（現場での使われ方、ロジット係数がプロビット係数の「およそ 1.6〜1.8 倍」という経験則は代表値の範囲確認のみ）

## 第5章（離散型分布）

- スクリプト: `verify/ch05.py`（実行: `.venv/bin/python verify/ch05.py`）
- 対象: bernoulli-distribution, binomial-distribution, finite-population-correction, fisher-exact-test, geometric-distribution, hypergeometric-distribution, multinomial-distribution, negative-binomial-distribution, poisson-distribution（9ファイル）
- 結果: **OK 128 / NG 0**
- 方法: sympy（各分布の MGF・PGF の導出と1・2階微分、E・V・階乗モーメント、ポアソンのキュムラント、再生性の母関数の積、幾何・負の二項の試行回数版、超幾何の共分散・分散の代数、多項分布の MGF の偏微分による E[XiXj]・Cov、ガンマ・ポアソン混合の積分、少数の法則の極限、MLE）、scipy.stats（全具体例の確率・分位点・正規近似、ポアソンとχ²の関係、`fisher_exact` で片側 p=0.0203・両側 p=0.0406・オッズ比21）、有限母集団修正は小さな母集団で全組合せを列挙して E[ȳ]・V[ȳ] の2つの表現と Cov=-σ²/(N-1) を確認、乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.005〜0.05）
- 本文の修正: なし
- 検算スクリプト側の修正: 初回 NG 5 件はすべて検算側の問題（sympy の無限級数・対数・平方根の簡約、多項分布例1の真値 0.08505 が丸め境界上で浮動小数誤差により許容 5e-5 をわずかに超過）。変数置換・数値代入・許容に 1e-12 を加えて解消
- 検算対象外: 定性的な記述（現場での使われ方、「無記憶性を持つ離散分布は幾何分布だけ」という特徴づけの定理そのもの、正規近似・FPC 省略の目安（np≥5、抽出率5%未満）、フィッシャー検定が保守的になりやすいこと）
