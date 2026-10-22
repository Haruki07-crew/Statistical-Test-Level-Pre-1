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
