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

## 第7章（極限定理・収束・確率不等式）

- スクリプト: `verify/ch07.py`（実行: `.venv/bin/python verify/ch07.py`）
- 対象: almost-sure-convergence, central-limit-theorem, chebyshev-inequality, continuity-correction, continuous-mapping-theorem, convergence-in-distribution, convergence-in-probability, delta-method, law-of-small-numbers, markov-inequality, slutsky-theorem, strong-law-of-large-numbers, weak-law-of-large-numbers（13ファイル）
- 結果: **OK 94 / NG 0**
- 方法: sympy（CLT の n log(1+t²/2n)→t²/2 と標準化一様分布の MGF 展開、(1-y/n)^n→e^{-y}、二項 pmf・PGF のポアソン極限、デルタ法の g' と漸近分散、E[(ΣX)^4]=nE[X⁴]+3n(n-1)σ⁴ の展開、積の telescoping）、scipy.stats（連続修正・符号検定・サイコロ和・ガンマ・t 分位点・二項の正確値、チェビシェフ図の4系列）、サイコロ100回の和は畳み込みで正確値 0.0420 を確認、乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.0003〜0.01。スラツキー例1の被覆率は各20万回で n=30: 0.9178、n=200: 0.9446、MC標準誤差≈0.0006 に対し許容 0.003。デルタ法例1の SD(λ̂) は 0.2046（逆ガンマの厳密値 0.2041）、例3の SD(√X̄) は 0.0707、許容 0.002／0.001）
- 本文の修正:
  - `content/ch07/law-of-small-numbers.md` 例1の表 P(X≥4)：二項 0.0525→0.1427、ポアソン 0.0527→0.1429（1-P(X≤3) と不整合だった）
  - `content/ch07/continuous-mapping-theorem.md` 例2：「1.96² ≈ 3.8415」→「1.96² = 3.8416 … 3.8415 と（1.96 の丸めの差を除いて）一致」
- 検算スクリプト側の修正: 大数の弱法則例2の上限 0.1079 は丸め前の p(1-p) で計算した値として照合（0.1685 で計算すると 0.1078）
- 検算対象外: 定性的な記述（現場での使われ方、収束の強弱関係・ボレル・カンテリ・レヴィの連続性定理・コルモゴロフの強法則の必要十分性など定理そのもの、近似の目安 n≥50・np≤5）

## 第6章（連続型分布・標本分布・非心分布）

- スクリプト: `verify/ch06.py`（実行: `.venv/bin/python verify/ch06.py`）
- 対象: uniform-distribution, normal-distribution, standard-normal-distribution, exponential-distribution, gamma-distribution, beta-distribution, lognormal-distribution, cauchy-distribution, chi-squared-distribution, t-distribution, f-distribution, t-statistic, normal-mixture-distribution, bivariate-normal-distribution, multivariate-normal-distribution, noncentral-chi-squared-distribution, noncentral-t-distribution, noncentral-f-distribution（18ファイル）
- 結果: **OK 222 / NG 0**（初回 OK 210 / NG 12）
- 方法: sympy（各分布の MGF の導出と1・2階微分による E・V、キュムラント、標準正規の偶数次モーメントと漸化式・級数係数、Z² の密度、ベータの E・V・最頻値、ガンマ・対数正規の最頻値、2変量正規の平方完成と MGF の偏微分、混合正規の MGF から E・V、非心χ²の MGF 微分と再生性）、scipy（MGF・特性関数の数値積分／QAWF、全具体例の確率・分位点、t/F/χ² の表の値、ttest_ind_from_stats、ncx2・nct・ncf による検出力の表の再計算と「80%に必要な最小サイズ」の確認）、混合正規の例は山 159.44・168.86、谷 166.27 を格子探索で確認、乱数シミュレーション（seed=20261023、KS 統計量 0.005〜0.015 や確率±0.003〜0.015 など許容誤差は各チェックに明記）で再生性・分布の構成（t, F, χ², コーシー, 非心分布, 順序統計量, 一元配置の F）を確認
- 本文の修正:
  - `content/ch06/noncentral-f-distribution.md` 例：「各群15区画程度が必要」→「各群14区画以上が必要」（r=13 で 0.7922、r=14 で 0.8255）
  - `content/ch06/noncentral-chi-squared-distribution.md` 例1：「棄却限界は 1.96²=3.841」→「χ²_0.05(1)=3.841（≈1.96²）」（1.96²=3.8416）
  - `content/ch06/standard-normal-distribution.md` 試験のポイント：「1.96²≈3.841」→「1.96²≈3.84」
- 検算スクリプト側の修正: 初回 NG のうち本文由来は2件（上記の1・3番目。2番目は同じ理由で併せて修正）、残り 10 件は検算側の問題（実行前に、scipy の kstest に分布名文字列と args を渡すとエラーになる件は凍結分布の cdf を渡して回避。t>0 の MGF 数値積分のオーバーフロー → 対数密度で計算・有限上限、sympy の Piecewise、丸め境界 0.02835、χ²(9) の区間は表示値 2.700 ではなく正確な分位点で計算された値、χ² 上側点の「おおよそ1.8〜2倍」は最小 1.778 のため 1.75 以上を許容、t(1) の平均は積分の発散で確認、比較の許容 1e-9 が厳しすぎ、混合正規のシミュ平均の許容を 0.05→0.07（標準誤差 0.0137 の約5倍））
- 検算対象外: 定性的な記述（現場での使われ方、無記憶性をもつ連続分布が指数分布だけという特徴づけ、多変量正規の同値な定義、グラフの見た目の説明）

## 第8章（点推定・最尤法・十分統計量・情報量）

- スクリプト: `verify/ch08.py`（実行: `.venv/bin/python verify/ch08.py`）
- 対象: asymptotic-normality-of-mle, consistent-estimator, cramer-rao-inequality, efficient-estimator, exponential-family, factorization-theorem, fisher-information, invariance-of-mle, likelihood-function, maximum-likelihood-estimator, mean-squared-error, method-of-moments, point-estimation, rao-blackwell-theorem, score-function, sufficient-statistic, unbiased-estimator（17ファイル）
- 結果: **OK 190 / NG 0**
- 方法: sympy（ベルヌーイ・二項・ポアソン・指数・正規(μ)・正規(σ²) のスコア、E[s]=0、E[s²] と -E[∂²log f] がともに表の I(θ) に一致、正規の情報行列の交差項 0、母数の取り換え I(θ)=1/θ²、尤度方程式を solve して各 MLE、CR 下限 g'²/(nI)、指数レートの E[1/S]・E[1/S²]（ガンマ積分）と V[(n-1)/S]=λ²/(n-2)、一様最大値の E・V、指数型分布族の A'(η)・A''(η)・正規化・キュムラント、ガンマ・正規の分解、ポアソン/ベルヌーイの条件付き分布、ラオ・ブラックウェル例2の E[φ]・E[φ²]・V[φ]、MSE(Q/d) の最小点 d=n+1、c*、モーメント法の連立方程式）、scipy.stats（正規近似・CI・対数尤度・norm.fit・二項による MSE の厳密計算・ガンマ分布の平均分散）、全グラフのデータ列（17系列以上）を式から再計算して表示桁で照合、乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.001〜0.06、または相対1%。S²・σ̂² の不偏性/偏り（非正規の指数母集団でも）、(n-1)/S の平均・分散、V[S²]=32/9、中央値の漸近相対効率 2/π、RB 改良前後の分散 0.2325/0.01423、MSE の d=9,10,11、一様分布の2つの不偏推定量の分散、スコアの平均・分散、Cov(T,U)=g'(θ)、MLE の漸近分散）
- 本文の修正:
  - `content/ch08/invariance-of-mle.md` 図のデータ列：格子点上の最大値（θ=0.425）で割った値になっており、真の最大（θ=3/7）で割った相対尤度と小数第4位で9点ずれていたので、L(θ)/L(3/7) の値に置き換え（例：0.425 の点 1→0.9999、0.275 の点 0.8239→0.8238 など）
- 検算スクリプト側の修正: スコア検定 U²/(nI) の棄却率をシミュレーションで 0.05 と照合して初回 NG（0.0456）。ポアソン n=50 の離散性による正しい値（正確計算 0.0453）なので、正確計算との一致と、n=20000 で 0.0498→0.05 に近づくことの2チェックに分けて解消
- 検算対象外: 定性的な記述（現場での使われ方、正則条件のもとでの MLE の一致性・漸近正規性の定理そのもの（シミュレーションで傍証のみ）、レーマン・シェフェの定理と完備性、「下限に達するのは指数型分布族の平均パラメータに限る」という特徴づけ、コーシー分布の最小十分統計量）

## 第9章（区間推定）

- スクリプト: `verify/ch09.py`（実行: `.venv/bin/python verify/ch09.py`）
- 対象: ci-mean-known-variance, ci-mean-unknown-variance, ci-paired-difference, ci-proportion, ci-two-sample-mean-difference, ci-two-sample-proportion, ci-variance-ratio, ci-variance, confidence-coefficient, confidence-interval-test-duality, confidence-interval, pivotal-quantity, required-sample-size, wald-confidence-interval, wilson-interval（15ファイル）
- 結果: **OK 330 / NG 0**（初回 OK 318 / NG 12）
- 方法: scipy.stats の分位点（z・t・χ²・F、自由度が小数のウェルチの t も含む）で全数値例を再計算し、表示桁で照合。ttest_1samp・ttest_rel・ttest_ind・ttest_ind_from_stats の CI/統計量とも照合。二項分布で被覆確率を正確計算（n=20, p=0.1：ワルド 0.876、ウィルソン 0.957）。sympy（ウィルソン区間の2次式・判別式・根と公式の一致、ベルヌーイ・ポアソン・指数のフィッシャー情報量と MLE、有限母集団修正の式の導出、p(1-p)=1/4-(p-1/2)²、指数ピボット）。図のデータ列：p 値関数17点（2·P(t₉>|x̄-μ₀|/SE)、最大誤差 <5e-5）、必要な n の2系列20点（⌈z²/(4E²)⌉、未丸めの z）を式から再計算して一致。乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：被覆率は 20万回で SE≈0.0005 に対し ±0.002、10万回・不等分散ウェルチ等は ±0.003〜0.006、KS 統計量 <0.004〜0.006）で z/t/χ²/F/指数/一様/対応あり/2標本/比率の差の各区間の被覆率、ピボット量の分布、n=5 で z 分位点を使った区間の被覆率 0.8784、指数 n=40 のワルド型の被覆率（本文「約94%」→ 0.9391）、t(5) 母集団 n=30 の分散区間の被覆率（本文「8割強」→ 0.8258）、検定の受容 ⇔ 区間が μ₀ を含む（20万回で完全一致）を確認
- 本文の修正:
  - `content/ch09/required-sample-size.md`：本文は 1.9600 と表示しながら途中の値を丸め前の z で計算していたので、表示どおり 1.9600 で計算した値に統一（例1 9.7998²→9.8000²、384.15→384.16、例2 1067.07→1067.11、682.93→682.95、例3 n₀ 1067.07→1067.11、1066.07→1066.11、1.21321→1.21322、879.54→879.57）。結論（97・385・1068・683・880 人）は不変
  - `content/ch09/wilson-interval.md` 例1：k=1.96²/20 は 0.19208（0.19207→0.19208、2次式の係数 1.19207・0.39207→1.19208・0.39208、大きい根 0.30103→0.30104）。区間 [0.0279, 0.3010] は不変
- 検算スクリプト側の修正: 信頼係数の例1の 90% の長さ 6.580 は本文の 1.6449 で計算した値（丸め前の z では 6.579）なので表示値で照合。ウィルソン区間の中心 ≈ (X+2)/(n+4)（「ほぼ」）の許容を絶対差 0.002 → 相対差 2% に（実際 0.1645 vs 0.1667、相対 1.3%）
- 検算対象外: 定性的な記述（現場での使われ方、信頼区間の解釈、予備検定の問題、クロッパー・ピアソン区間が保守的という位置づけ、custom 図 ci-coverage の描画そのもの（被覆率はシミュレーションで確認））

## 第10章（仮説検定の基礎・検定法の導出・多重検定）

- スクリプト: `verify/ch10.py`（実行: `.venv/bin/python verify/ch10.py`）
- 対象: bonferroni-correction, false-discovery-rate, holm-method, likelihood-ratio-test, monotone-likelihood-ratio, multiple-testing, neyman-pearson-lemma, null-and-alternative-hypothesis, one-sided-two-sided-test, p-value, power-function, rejection-region, sample-size-determination-test, score-test, significance-level, type-i-and-ii-errors, uniformly-most-powerful-test, wald-test, wilks-theorem（19ファイル）
- 結果: **OK 166 / NG 0**（初回 OK 162 / NG 4）
- 方法: sympy（NP 補題の対数尤度比（正規・ベルヌーイ・指数）、平方和の分解と -2logΛ=Z²、σ̂0²/σ̂²=1+t²/(n-1)、指数の -2logΛ=2n{λ0x̄-1-log(λ0x̄)}、ワルドの表の 1/I(θ)（ベルヌーイ・ポアソン・指数・正規）、デルタ法 V[logλ̂]=1/(nλ)、スコア U・nI・S の形（ベルヌーイ・ポアソン・指数）と Z_S=母比率の z 検定、スコア統計量の ψ=logλ への変換不変性、コーシーの尤度比の最大点 (1+√5)/2、標本サイズ公式を n について解く、検出力関数の単調性）、scipy.stats（全具体例の p 値・棄却限界・検出力、無作為化検定の γ=0.8933 と検出力 0.6456、ポアソンの UMP 検出力表、非心 t による t 検定の検出力 0.4840 と n=32/33/34 の 0.782/0.795/0.808、ボンフェローニ・ホルム・BH の判定と調整済み p 値、ウィルクス例3 の棄却率の正確値 0.057）、line-data の全データ列（対数尤度の落差 21点、MLR の正規・コーシー 21点×2、ワルド・尤度比・スコア 26点×3、スコア関数 17点、FWER 15点×2、ボンフェローニの検出力 15点、BH・ホルムの基準）を式から再計算、乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記：0.001〜0.004。p 値の一様性、両側 z 検定のサイズ、指数 n=50 での -2logΛ の χ²(1) 近似、等相関 0.8 でのボンフェローニの FWER（0.05 未満で独立より小さい）、相関 0.5・m0=6 でのホルム法の FWER、独立 m=20・m0=15 での BH の FDR≈m0q/m=0.0375 と FDR≤FWER）
- 本文の修正:
  - `content/ch10/one-sided-two-sided-test.md` 読み解き：「1.96²≈3.841」→「1.96²≈3.84」（1.96²=3.8416。第6章と同じ扱い）
  - `content/ch10/sample-size-determination-test.md`：式に表示した z（1.645, 1.960, 0.842, 1.282）で計算すると合わない途中値を、表示値での計算結果に統一（例1 4.973²≈24.73→4.974²≈24.74、試験のポイントの 24.73→24.74、90% の 42.03→42.04、例2 62.79→62.81、例3 152.46→152.52）。結論（25・32・43・63・153 人）は正確な z でも同じで不変
- 検算スクリプト側の修正: sympy の平方根の簡約（Z_S の一致は2乗と符号で確認）、一様分布の MLR 判定で ∞-∞ が nan になる件（大きな有限値で代用）。棄却域の例2 の検出力 0.3595 は正確な分位点 2×3.9403=7.8806 での値（丸めた 7.880 では 0.35944）なので、正確な値で照合し、丸め値とは 1e-4 以内であることを確認
- 検算対象外: 定性的な記述（現場での使われ方、NP 補題・カーリン・ルビンの定理・ウィルクスの定理・BH 法の FDR 保証の証明そのもの（シミュレーションと記号計算で傍証のみ）、最強力検定の一意性、UMPU の主張、境界上の帰無仮説の漸近分布（混合分布での保守性のみ数値確認）、グラフの見た目の説明）

## 第11章（正規母集団の検定：平均・分散・相関・多変量）

- スクリプト: `verify/ch11.py`（実行: `.venv/bin/python verify/ch11.py`）
- 対象: one-sample-t-test, z-test-known-variance, two-sample-t-test, pooled-variance, welch-test, paired-t-test, power-of-t-test, test-ci-relationship-mean-difference, f-test-equal-variances, chi-square-test-variance, fisher-z-transformation, test-correlation-coefficient, hotelling-t2（13ファイル）
- 結果: **OK 216 / NG 0**（初回 OK 213 / NG 3）
- 方法: scipy.stats（ttest_1samp・ttest_ind_from_stats・ttest_rel・ttest_ind・linregress で全例の統計量と p 値、t/z/χ²/F の分位点、信頼区間、nct による検出力と最小標本サイズ n=34・各群26、F・χ² 検定の検出力）、sympy（ウェルチ・サタースウェイトの ν がモーメント法 2E[V̂]²/V[V̂] と一致、プールの最適重み w*=(m−1)/(m+n−2) と最小分散 2σ⁴/(m+n−2)、フィッシャー z のデルタ法 g'=1/(1−r²) と ρ の消去）、ν の範囲 min(m−1,n−1)≤ν≤m+n−2 をランダム2万組で確認、line-data 3図（t 検定の検出力・z 近似の各10点、p 値関数21点）を式から再計算し4桁丸めで完全一致、フィッシャー z の図（n=20・4万回の SD）は再シミュレーションで r の SD 相対3%以内・z の SD 0.005 以内、乱数シミュレーション（seed=20261023、KS 統計量 0.005〜0.007、確率は MC 標準誤差の3〜5倍の許容 0.002〜0.006 を各チェックに明記）で T〜t、S_p² の不偏性・分散・χ²、F/r〜F、非心 t、ホテリング T²（1標本・2標本・Σ既知）、相関の t、ウェルチの水準を確認
- 本文の修正:
  - `content/ch11/paired-t-test.md` 例2：「標準誤差が約2.7倍」→「約2.8倍」（独立 SE 3.0050 ÷ 対応 SE 1.0856 = 2.768）
  - `content/ch11/welch-test.md` 例2：20万回シミュレーションの棄却率「約17.4%・約5.2%」→「約17.6%・約5.3%」（200万回で 0.1755・0.0531。旧値はいずれも MC 標準誤差の約2倍低かった）
- 検算スクリプト側の修正: プールの例3「≈10.13」は 10.125 の四捨五入で正しく、許容 5e-3 の浮動小数点境界の問題だったので許容を 5.01e-3 に。artanh=½log((1+r)/(1−r)) は sympy の simplify が 0 に簡約できなかったため数値格子（199点）で確認
- 検算対象外: 定性的な記述（現場での使われ方、予備検定の問題点、ベーレンス・フィッシャー問題に正確な解がないこと、F 検定・χ² 検定が正規性に敏感という一般論（t(5) 母集団の1例で水準 0.147 を確認したのみ）、図のスライダー挙動）

## 第12章（適合度検定・分割表・比率とポアソンの検定）

- スクリプト: `verify/ch12.py`（実行: `.venv/bin/python verify/ch12.py`）
- 対象: binomial-test, chi-square-approximation-condition, degrees-of-freedom-gof, expected-frequency, g-statistic, goodness-of-fit-test, homogeneity-test, independence-test, kolmogorov-smirnov-test, pearson-chi-square-statistic, poisson-parameter-test, poisson-two-sample-test, proportion-test, two-sample-proportion-test（14ファイル）
- 結果: **OK 211 / NG 0**（初回 OK 209 / NG 2）
- 方法: scipy.stats（chisquare・chi2_contingency（連続修正あり/なし、lambda_="log-likelihood"）・power_divergence・binomtest・fisher_exact・kstest(exact)・kstwo・kstwobign）で全具体例の期待度数・寄与・統計量・自由度・棄却限界・p値を再計算。二項検定の両側 p 値は2倍法と「確率の小さい順に足す方法」（binomtest と一致）の両方、p0=0.5 で両者が一致すること（n=5..30 全x）、実サイズ≤α。ポアソンの P(T≥t)=P(χ²_{2t}≤2nλ0)、調整済み標準化残差 ±2.31、2×2 公式とイェーツ、|O−E|=|ad−bc|/n、(O−E) の行和・列和=0（ランダム表）、コルモゴロフ分布の級数と kstwobign の一致・漸近5%/1%点。sympy（k=2 での χ²=Z²、G の2次近似 2(E+δ)log(1+δ/E)=2δ+δ²/E+O(δ³)、自由度 (rc−1)−(r−1)−(c−1) と r(c−1)−(c−1)、条件付き二項分布の導出、H0 で p0=t1/(t1+t2)）。charts の params（obs-exp-bar の observed/expected、mosaic の表から計算される統計量 7.11・8.43・4.566、ks-ecdf のデータ、Bin(25,0.5455) の F(17)・P(X≥18)、各 pdf-cdf のキャプションの F(x)）も本文と照合。乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記）：G と X² の5%棄却率（20万回、±0.004/±0.003）、行和固定・総数固定での χ²(2) の棄却率（10万回、±0.004）、ポアソン適合度で生データ平均を使うと統計量が χ²(3) と χ²(4) の間（平均 3.03）・グループ化 MLE なら水準どおり（5000回、±0.012）、KS の D10 が一様・指数で同じ kstwo(10) に従う（5万回、KS<0.0073）、母数推定時の KS が極端に保守的（棄却率 0.00005）、ポアソンの再生性、合計で条件付けた二項分布、比率の z 検定のサイズ
- 本文の修正:
  - `content/ch12/two-sample-proportion-test.md` 図1キャプション：「x を 2.14 に合わせると…右片側 p 値は約 0.016、両側 p 値はその2倍の約 0.033」→「約 0.032」（x=2.14 での両側は 2×0.01618=0.0324。本文の p 値 0.0326 は Z=2.137 での値で不変）
- 検算スクリプト側の修正: 200×0.55==110 の浮動小数点比較を許容つきに。KS の分布非依存性の確認を2標本 KS（初回 0.0102、p≈0.01 の偶然の外れ）から、各標本を kstwo(10) と比べる1標本 KS に変更
- 検算対象外: 定性的な記述（現場での使われ方、「5」は経験的な目安、ピアソンの統計量のほうが近似がよいという一般論、チャーノフ・レーマンの結果の証明（シミュレーションで傍証のみ）、KS が裾のずれに感度が低いこと、UMP 性）。なお custom 図 `obs-exp-bar` と `ks-ecdf` は `src/` に描画関数が見当たらなかった（params の値の整合のみ確認）

## 第13章（ノンパラメトリック法）

- スクリプト: `verify/ch13.py`（実行: `.venv/bin/python verify/ch13.py`）
- 対象: sign-test, wilcoxon-signed-rank-test, wilcoxon-rank-sum-test, mann-whitney-u-test, kruskal-wallis-test, friedman-test, spearman-rank-correlation, kendall-rank-correlation, concordant-discordant-pairs, rank-and-ties, permutation-test, normal-approximation-rank-tests, nonparametric-methods-overview（13ファイル）
- 結果: **OK 200 / NG 0**（初回 OK 198 / NG 2）
- 方法: 全通りの数え上げ（符号 2⁸・2⁹、群分け C(9,4)・C(13,6)・C(8,4)、C(20,10) は DP、KW の 34650 通り、フリードマンの (3!)⁵=7776 通り、並べ替え 6!・10!、ケンドールの n=10 は転倒数分布）で全例の正確な p 値（7/256、26/512、7/126、0.0446、47/1716、48/34650、66/7776、63/720、0.0008、98/720、0.0022、2/70、8/256）を再計算。scipy.stats（wilcoxon exact/approx、mannwhitneyu exact/asymptotic、ranksums、kruskal、friedmanchisquare、spearmanr、kendalltau exact、permutation_test、ttest_ind/1samp、f_oneway、binomtest）とも一致。正規近似・χ² 近似の z と p 値（連続修正あり/なし、タイ補正あり/なし）。bar-data 6図（T+ n=8 の37値、W・U の21値、フリードマン Q の12値、r_s n=6 の36値、ケンドール S の16値、並べ替え分布32値、一致対の内訳、平均順位、外れ値での p 値）を再計算し完全一致。sympy（E[T+]・V[T+]、タイ群の2乗和の減少 (t³−t)/12、Cov(Ri,Rj)=−(N+1)/12 と V[W]、U のタイ補正式＝C 倍、E[H]=k−1（一般の n_j）、H・Q の2つの形の一致、k=2 で H=Z²・Q=符号検定の Z²、E[Q]=k−1、V[r_s]=1/(n−1)、V[S]・V[τ]、順位の E・V、ARE：正規 3/π・2/π、ラプラス 1.5・2、放物線密度で下限 108/125）。分散公式は小さい n の全数え上げでも確認。ケンドールの共分散 ±1/3 は 3! 通りで確認。乱数シミュレーション（seed=20261023）：U/(n1n2) の不偏性（10万回、±0.003）、2変量正規で τ=2/π·arcsin ρ（40万組、±0.005）、順位ベクトルの 4! 通り等確率（20万回、±0.003）、|γ|≥|τb|≥|τa|（ランダム表300件）、タイありの V[R] 公式（300件）
- 本文の修正:
  - `content/ch13/wilcoxon-signed-rank-test.md` 図キャプション：「右端の T+≥32 の4本の合計」→「5本の合計」（T+=32,33,34,35,36 の5本で 2+2+1+1+1=7 通り）
- 検算スクリプト側の修正: 8/256=0.03125 を「0.0313」とする表示は四捨五入として正しいので、許容を 5.01e-5 に
- 検算対象外: 定性的な記述（現場での使われ方、正規近似が使える標本サイズの目安、0.864 が「任意の分布での下限」であることの証明（下限を与える密度での値のみ確認）、ケンドールの正規近似がスピアマンより良いという一般論、交換可能性が崩れる場合の議論、図のスライダー挙動）

## 第14章（マルコフ連鎖）

- スクリプト: `verify/ch14.py`（実行: `.venv/bin/python verify/ch14.py`）
- 対象: markov-chain, markov-property, transition-probability-matrix, chapman-kolmogorov-equation, n-step-transition-probability, stationary-distribution, detailed-balance, limiting-distribution, ergodicity, periodicity, irreducibility, communicating-class, recurrence-and-transience, first-passage-time, absorbing-state, absorption-probability（16ファイル）
- 結果: **OK 179 / NG 0**（初回 OK 178 / NG 1。NG は検算スクリプト側の記号簡約の問題）
- 方法: sympy の有理数行列で厳密計算（全例の π^(n)、P^2・P^3・P^5・Q^10 などのべき、P^(m+n)=P^mP^n（m,n=0..4）、定常分布を πP=π・Σπ=1 で解く、平均初到達時間・平均再帰時間の連立方程式と m_j=1/π_j（全状態）、吸収確率・平均吸収時間の一次方程式、基本行列 N=(I−Q)^{-1}・N1・B=NR）、numpy（P^T の固有ベクトル、N=I+Q+Q²+… の級数、P^400 の極限、ランダム確率行列200個で固有値1と |λ|≤1、破産問題の公式と対称時の i(K−i) を NR・N1 と12組で照合）。記号計算: 2状態の P^n=Π+λ^nD（Π²=Π、D²=D、ΠD=DΠ=O）と固有値 {1,1−a−b}、2状態の定常分布 (b/(a+b),a/(a+b))、可約例の π_w が任意の w で定常、MH 受容確率の詳細つり合い、破産問題の一般解が漸化式と境界条件を満たすこと。詳細つり合い: 例1・例3（メトロポリス）で全組成立、例2（ブランド）で不成立と流量 0.04375/0.0625、コルモゴロフ判定 0.001/0.004、1→3→2→1 の向きの正味の流れが正。周期: 戻れる n の gcd（周期2、長さ2と3の閉路で周期1）、類の分割と閉じた類を到達可能性行列で判定。ランダムウォーク: C(2n,n)/4^n≈1/√(πn)、p=0.6 で Σp00^(2n)=5 から f00=0.8
- charts: 全16ファイルの front matter を yaml で読み、markov-chain 図の P（計19枚）が本文の行列と一致すること（詳細つり合い図1の 0.3333/0.1667 は 1/3・1/6 の4桁丸めで行和1）、`src/charts_custom.js` と同じ遅延べき乗法で計算した「図の下の定常分布」がキャプションの値と一致すること、キャプションの収束先・往復・「いったん上がって下がる」・「0.5 倍ずつ減る」・「差が毎回 0.3 倍」を再計算で確認。random-walk 図の params（壁・出発点・p）が本文の例と一致、bar-data（初到達時間）は 0.4×0.6^(n−1) の4桁丸めと一致。ergodicity の line-data（乱数の一例）は厳密値ではなく妥当性のみ確認：y·n が整数、累積の晴日数が単調で増分≤日数の増分、n≥50 で漸近分散 π1π2(1+λ)/(1−λ)=0.4548 による |z|≤4（最大 1.32）、例2の表（0.300/0.560/0.570/0.575）と一致
- 乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記）: エルゴード定理（20万日で晴の割合 0.5692・平均売上 8.277、許容 0.01・0.04）、p=0.6 の原点回帰確率（2万本、0.8034、許容 0.01）、A の平均再帰時間 16/7（5万回、2.281、許容 0.05）、試用からの平均吸収時間 8（5万人、7.996、許容 0.15）、破産問題の上の壁到達率 4/19 と平均終了回数 35/19（10万本、許容 0.006・0.02）、訪問回数の幾何分布の平均 2
- 本文の修正: なし
- 検算スクリプト側の修正: 破産問題の一般解の記号確認で sympy が (1−p)^i の約分を簡約できなかったため、r=q/p で表し直して確認。級数 Σ C(2n,n)x^n の計算を項の漸化式に変更（大きな整数の float 変換のオーバーフロー回避）
- 検算対象外: 定性的な記述（現場での使われ方、エルゴード定理・基本定理・ポリアの定理・コルモゴロフの判定法の証明そのもの（具体例と数値で傍証のみ）、CK 方程式を満たす非マルコフ過程の存在、p=1/2 の零再帰性（部分和の √N 的な発散のみ確認）、図の見た目とスライダー挙動。random-walk 図は乱数描画なので params のみ確認）

## 第15章（確率過程・ポアソン過程・待ち行列）

- スクリプト: `verify/ch15.py`（実行: `.venv/bin/python verify/ch15.py`）
- 対象: stochastic-process, independent-stationary-increments, poisson-process, interarrival-time, superposition-thinning-poisson, random-walk, gambler-ruin-problem, martingale, brownian-motion, birth-death-process, queueing-theory, traffic-intensity, littles-law, mm1-queue（14ファイル）
- 結果: **OK 157 / NG 0**（初回 OK 153 / NG 4。NG はすべて検算スクリプト側の問題）
- 方法: sympy（P(N(t)≥n) の微分＝アーラン密度と E=n/λ・V=n/λ²（n=1..6）、各項の積の微分、Exp・Gamma(3) の MGF、尺度表記との一致、P0'=−λP0 の解と Q_n=(λt)^n/n!、N(s)|N(t)=n ~ Bin(n,s/t)（n≤5）、間引きの同時確率の Po(λpt)×Po(λqt) への分解、P(T1<T2)=λ1/(λ1+λ2)、PGF の積、破産問題の P_i・D_i が差分方程式と境界条件を満たすこと（r=q/p で表示）・対称時の2階差分 −2・r→1 で i/N、マルチンゲール S_n²−n・r^{S_n}・S_n−n(p−q)、出生死滅過程の積の公式が平衡方程式を満たすこと（K=5 記号）と前向き方程式、M/M/1 の L・V[N]・Lq・P(N≥n)・W・Wq・W=Wq+1/μ、系内時間 MGF の和＝(μ−λ)/(μ−λ−s)、P-K 公式が M/M/1・M/D/1 に帰着、dL/dρ）。fractions（機械修理モデル π=(9/17,6/17,2/17)・平均 10/17、M/M/1 例1〜3、M/D/1 比較表、リトル例1〜3、80/243、8/27）。第14章型の吸収連鎖の基本行列で破産問題の例1・例2・マルチンゲール例2（P=0.0419）を再計算。scipy.stats でブラウン運動の確率（0.1587・0.1103・0.3173・0.3085）、ランダムウォークの二項・正規近似（0.2508・0.2461・0.1841）、ガンマ cdf（0.5768）など全数値例
- charts: birth-death-process の markov-chain 図の P が一様化 I+Q/5 と一致し、πP=π、P^20 の全行が (0.529,0.353,0.118)。Po(8)=λ/μ。littles-law の line-data が例1の N(t) の階段と完全一致し面積13。mm1-queue・traffic-intensity・queueing-theory の line-data（L=ρ/(1−ρ)、Lq=ρ²/(1−ρ)、M/D/1 の L=ρ+ρ²/{2(1−ρ)}）が2桁丸めで一致。geom の p=1−ρ（0.4、0.2）、expon rate=3、random-walk 図の start・lower・upper・p が各例と一致
- 乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記）: 機械修理モデルの時間割合（40万遷移、±0.005）、ブラウン運動の Cov(W1,W3)=1、W(1)|W(3)=3 の N(1,2/3)、反射原理（2000刻み格子では最大値を見落とすため Broadie-Glasserman 補正値 0.3110 と比較、±0.013）、破産問題 p=0.47 の P=0.2312・D=89.59（10万本）、p=0.5 の 0.5・100、マルチンゲール例1の 0.3・21 と例2の 0.0419、ポアソン過程の同時確率 0.0529（積 0.0362 とは異なること）と条件付き二項 0.3292、Exp 間隔で単位時間の個数が Po(3)、検査のパラドックス（間隔の平均 2/λ、残り 1/λ）、N(1)=2 のもとでの S1 の平均 1/3、間引きの独立性（P(N1=0|N2=k) が k によらない）、合成の先着確率 0.4・最小値の平均6分、M/M/1（リンドレー漸化式100万人で W=0.25・Wq=0.15・P(W>0.5)=e^−2・P(Wq=0)=0.4、出生死滅シミュで π0・L・P(N≥3)）、M/D/1 の Wq=0.4、一様サービス M/G/1 で P-K 公式、ランダムウォーク p=0.6 の E・V・Cov(S5,S10) と P(S10=3)=0、表の枚数過程の相関 0.6325
- 本文の修正: なし
- 検算スクリプト側の修正: M/M/1 の無限和を sympy.summation で求めると ρ<1 の条件付き Piecewise が返り比較が NG になったため、本文と同じ「等比級数を微分する」方法で Σnρ^n・Σn²ρ^n を作り、打ち切り和（ρ=0.3,0.6,0.8）とも照合する形に変更
- 備考: gambler-ruin-problem 例2 の D10≈89.59 は厳密値 89.594 の丸め（表示の P10=0.2312 を代入すると 89.60 になる）。誤りではないので修正していない
- 検算対象外: 定性的な記述（現場での使われ方、ドンスカーの定理・見本路の微分不可能性・2次元の再帰性/3次元以上の一過性・任意停止定理の条件・PASTA の証明（M/M/1 のシミュで傍証のみ）、リトルの公式の一般性（例1とM/M/1でのみ確認）、図の見た目とスライダー挙動。random-walk・poisson-process 図は乱数描画なので params のみ確認）

## 第17章（回帰診断）

- スクリプト: `verify/ch17.py`（実行: `.venv/bin/python verify/ch17.py`）
- 対象: residual, leverage, standardized-residual, studentized-residual, cooks-distance, dffits, outlier-and-influential-point, residual-plot, normal-qq-plot, durbin-watson-statistic, serial-correlation, heteroscedasticity（12ファイル）
- 結果: **OK 174 / NG 0**（初回 OK 169 / NG 5。本文の丸め誤り2件、検算スクリプト側の許容誤差3件）
- 方法: 共通データ x=(1,…,7,12), y=(3,5,6,9,10,13,14,20) を fractions で厳密計算（x̄・Sxx=84・Sxy=133・係数・残差・Σe=Σxe=ŷᵀe=0・SSE=65/12・Σh=2）し、numpy で H・てこ比・標準化残差・スチューデント化残差・クックの距離・DFFITS を「1点除いて実際に当てはめ直す」定義どおりに全8店計算して本文・表・bar-data と照合。簡便式（D_i の2形・DFFITS=t_i√(h/(1−h))・t_i と r_i の関係・(n−p−2)s(i)²=(n−p−1)s²−e²/(1−h)・D=s(i)²DFFITS²/((p+1)s²)）、シャーマン・モリソンによる β̂−β̂(i)、削除残差 e/(1−h)、1+x_iᵀ(X(i)ᵀX(i))⁻¹x_i=1/(1−h)、PRESS、平均シフトモデルの t 値＝t_8 を全店で確認。sympy（(I−H)X=O・対称・べき等・tr(I−H)=n−p−1、h_ii=Σh_ij²、二次形式の簡約、x̄ の点をずらすと傾き不変・切片 Δ/n、DW 分子の展開、V[ȳ] の二重和、AR(1) の γ0、WLS(w=1/x²)=平均(y/x)）。ランダム X 300個で 1/n≤h≤1・tr H=p+1・中心化不変・h=1/n+MD²/(n−1)。scipy で Beta(1/2,5/2) の 0.0250・0.0126（t(5) の p 値と一致）、t_0.025(5)・t_{0.05/16}(5)=4.526、F_0.5(2,6)=0.780、F(2,6) で 95.5%点、BP 検定 p=0.0251（statsmodels het_breuschpagan robust 版とも一致）、Q-Q の分位点と相関 0.9926・0.9200・0.9893。外れ値例2（店5=14）・例3（店8=23.7143）を当てはめ直して確認。DW 例2 の P(d≤1.006|H0) は Imhof の公式で 0.0105（本文「≈0.011」と整合）
- charts: 全 bar-data（てこ比・e・r・t・D・DFFITS）と line-data（DW 例1の残差、残差プロット例1、不均一分散例1の残差）を再計算値と照合。regression-residual の「右端 x=16」は charts_custom.js の固定値と一致し、n=6,20,60 で右端のてこ比の平均が減少。t 分布 df=5、ar-sim・dw-sim・acf-plot・residual-pattern・qq-plot の params が本文と整合
- 乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記）: V[e]=σ²(I−H)・Cov(e,ŷ)=O・Cov(e,y)=σ²(I−H)・E[s²]=σ²・V[ŷ]=σ²h（20万回）、r_i²/6~Beta(1/2,5/2) と t_8~t(5)（KS<0.004）、|r|≤√6、DW 例2 の P(d≤1.006)（20万回 0.0101、±0.001）、AR(1) で d≈2(1−ρ)（ρ=±0.5、n=200）、AR(1) の分散・自己相関 ρ^k・V[ȳ] の膨張率、正の系列相関で OLS の SE 過小、n=8 の Q-Q 相関の5%点 0.906（本文「約0.91」）、不均一分散で OLS 0.2108σ²・WLS 0.125σ²・E[s²/Σx²]=23/204・サンドイッチ分散、t(3)・指数分布の平均 Q-Q の形
- 本文の修正:
  - `content/ch17/durbin-watson-statistic.md` 例2：端の補正「0.1032」→「0.1033」（(e1²+e10²)/Σe²=0.10325）
  - `content/ch17/studentized-residual.md` 例1：t_8 の分母「0.2851」→「0.2850」（s(8)√(1−h88)=0.28504）
- 検算スクリプト側の修正: 丸めた途中値どうしの積（4.4572/2×2.4286、−2.6310/(0.5278×√0.7083)、0.2786/(2×0.9028)×5.9228²）は最終桁が 2〜5e-4 ずれるのが丸めの伝播として正しいので許容を広げ、代わりに丸めない値で厳密に 5.4122・−5.9228 になることを併せて確認
- 検算対象外: 定性的な記述（現場での使われ方、DW 数表の d_L=0.879・d_U=1.320 は数表の値として妥当性のみ（X 固定の正確な p 値が判定保留域にあることは確認）、ダービンの h 統計量・ブロイシュ・ゴッドフリー検定・ニューイ・ウエスト型 SE、ガウス・マルコフの定理の一般証明、図の見た目とスライダー挙動。regression-residual・qq-plot・ar-sim・dw-sim・acf-plot・residual-pattern は乱数描画なので params のみ確認）

## 第16章（重回帰分析）

- スクリプト: `verify/ch16.py`（実行: `.venv/bin/python verify/ch16.py`）
- 対象: least-squares-method, normal-equations, multiple-regression-model, hat-matrix, gauss-markov-theorem, coefficient-of-determination, adjusted-r-squared, overall-f-test-regression, regression-coefficient-t-test, confidence-and-prediction-interval-regression, multicollinearity, variance-inflation-factor, standardized-partial-regression-coefficient, dummy-variable, interaction-term（15ファイル）
- 結果: **OK 228 / NG 0**（初回 OK 224 / NG 4。NG はすべて検算スクリプト側の問題）
- 方法: 共通データ（x1, x2, y の8店舗）を sympy の有理数行列で厳密計算（X'X・X'y・det=756・(X'X)⁻¹=adj/756・β̂=(5/21,22/21,115/84)・残差・X'e=0・ŷ'e=0・平方和分解 ST=SR+Se・H の対称/べき等/HX=X/tr H=3/h11=339/756/H の第1行・HJ=J と H−J のべき等・ST/SR/Se の二次形式表示・Cov(β̂,e)=O・q0=1/7）し、numpy/scipy で全数値例（単回帰 1.8742/2.9245、偏差形のクラメル、Se=4.7738、別係数で 5.04、R²=0.9428・R=0.9710、x1/x2 単回帰の Se・R²・adjR²、x3 追加の R²=0.9442・adjR²=0.9024・t=0.3145、F=41.23・p=0.00078・F_0.05/0.01(2,5)、R²=0.35 の F=3.365・p=0.025、係数表の SE・t・p・95%CI、β2=1 の t=0.824、信頼区間 [7.908,9.807]・予測区間 [6.172,11.542]・例2 の q0=1.877 と半幅、s²(X'X)⁻¹、VIF=2.524・(R⁻¹)_jj=1.565/3.261/2.783・R1²=0.3611、標準化係数 0.5111/0.5190 と R²=Σβ*r、多重共線性例の係数・t・p・F=1021.5・VIF=454.5・係数の符号反転、直交計画 (13.5,2.5,1.5)、ダミー例 1.75/1.5/3.5・SE 0.3・t=11.67、交互作用例・Se 6.525/0.9・部分F=25=t²）を再計算。sympy で ∂S/∂β=−2X'y+2X'Xβ とその解・ヘッセ行列の正定値、一般の x（n=4）で H²=H・tr H=2・h_ii=1/n+(x−x̄)²/Sxx、(X'X)⁻¹X'X=I と AA'=(X'X)⁻¹（E[β̂]=β・V[β̂]=σ²(X'X)⁻¹）、F の R² 表示、adjR² の2式の一致、2変数での (S⁻¹)_11=1/{S11(1−r²)} と corr(β̂1,β̂2)=−r12、相関行列のブロック逆行列で (R⁻¹)_11=1/(1−r'R⁻¹r)。ランダム例：2000例で t²=部分F かつ「adjR² 上昇 ⇔ |t|>1」、200例で 1/(1−Rj²)=(R⁻¹)_jj=Sjj·c_jj、300個の線形不偏推定量で V 差が非負定値。FWL 定理、ダミー回帰＝2標本 t・一元配置 F、ダミーの罠の rank 落ち、中心化で β3・t 不変
- charts: 全15ファイルの front matter を yaml で読み照合。least-squares の line-data（β1 を動かした S の10点）、normal-equations の残差、hat-matrix の h_ii、R² の平方和3本、標準化係数・多重共線性の bar-data、信頼区間・予測区間の帯5本（単回帰・自由度6、3桁丸め）、VIF と √VIF の曲線、ダミー・交互作用の直線が再計算と一致。gauss-markov の正規曲線の σ（√0.1, 1/3, √0.125）、F(2,5)・t(5) の params とキャプションの 5.786・0.985・0.015。adjusted-r-squared の line-data（シミュレーション平均）は、平均 adjR² が平均 R² の一次式 1−(1−R²)·19/(19−p) と ±0.002 で整合し、R² の増分が (1−adjR²)/19 と整合することを確認。scatter・regression-residual・bvnorm の params が本文と整合
- 乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記、概ね4SE）: 共通 X で 20万回、E[β̂]=β・V[β̂]=σ²(X'X)⁻¹（全成分4SE）・E[Se]=5σ²・V[e]=σ²(I−H)・Se/σ²~χ²(5)（KS）、H0 下の F 検定の棄却率 0.05・SR と Se の無相関、E[V_R] の式・非心 F の検出力、t による95% CI の被覆率、信頼区間・予測区間の被覆率、ガウス・マルコフ例1の3推定量の平均と分散（0.125/0.1111/0.1）、共線性下での不偏性、n=20 で無関係変数を足したときの R² の単調増加と adjR² の横ばい（4000回）
- 本文の修正: なし
- 検算スクリプト側の修正: ヘッセ行列の正定値判定で sympy の固有値が複素根号表示になり比較できなかったため主小行列式で判定。V[β̂] のシミュ比較を相対誤差（0 に近い非対角成分で過敏）から標本共分散の標準誤差の4倍に変更。(1−R²)/5=0.01143 は丸める前の R² で照合。corr(β̂1,β̂2)=−r12 の記号確認は平方根を避けて2乗と符号で確認。配列の形の不一致（allclose の比較先）を修正
- 備考: overall-f-test-regression 例1の「(0.9428/2)/(0.0572/5)=41.23」は丸めた R² を代入すると 41.21 になるが、丸める前の値では 41.23 なので誤りとはしていない
- 検算対象外: 定性的な記述（現場での使われ方、ガウス・マルコフの定理・正規性下の UMVU・コクランの定理の一般証明（具体例・乱数で傍証のみ）、adjusted-r-squared 図のシミュレーション設定（真の係数・誤差分散は本文に無いため完全再現はせず、整合性と定性的挙動を確認）、図の見た目とスライダー挙動。scatter・regression-residual・bvnorm は乱数描画なので params のみ確認）

## 第18章（質的回帰・一般化線形モデル）

- スクリプト: `verify/ch18.py`（実行: `.venv/bin/python verify/ch18.py`）
- 対象: logistic-regression, odds-and-odds-ratio, logistic-likelihood-estimation, iteratively-reweighted-least-squares, probit-model, poisson-regression, overdispersion, deviance, link-function, exponential-dispersion-family, generalized-linear-model, multinomial-logit-model, ordered-logit-model（13ファイル）
- 結果: **OK 199 / NG 0**（初回 OK 193 / NG 5。NG はすべて検算スクリプト側の問題）
- 方法: numpy で IRLS（フィッシャーのスコア法、二項ロジット・二項プロビット・ポアソン、オフセット対応）を独立に実装し、推定値・反復列・ŵ・X'ŴX とその逆行列・標準誤差を再計算。scipy.optimize（BFGS）の最尤推定でもロジット・プロビット・ポアソンの係数が一致。IRLS の1回目（スコア (−3,32)、X'WX=[[25,50],[50,150]]、β(1)=(−1.64,0.76)、作業応答 z と単回帰での手計算、ポアソン z と β(1)=(0.9918,0.4000)）と反復表（対数尤度 Σ{yη−μ} を含む）。スコア法の更新＝WLS を3モデルで数値確認、正準リンクで観測情報量（数値ヘッセ）＝X'ŴX。オッズ比・SE・CI（例2 (1.188,5.985)、ロジット例1 (1.746,3.746)）、率比とオフセットの例（IRLS の SE が √(1/18+1/30) と一致）、ワルド z と p 値、逸脱度（寄与 d_i、D=1.744、D0=33.695、ポアソン 0.280/10.240、対数尤度 −8.5168/−7.6450、ピアソン X²=1.726）、過分散（X²=40、φ̂=4.444、上側確率 7.6e−6、SE 0.1581→0.3333、k=1.161、P(0)=0.177）、多項ロジット・順序ロジットの確率と比例オッズ。sympy（ロジット対数尤度の書き換え・η での1・2階微分、∂p/∂x=βp(1−p)、オッズ比 e^β、OR=RR(1−p2)/(1−p1)、指数分散型分布族4分布の b'・b''・分散関数、キュムラント母関数 K'(0)=b'・K''(0)=ab''、正準リンク (b')⁻¹（正規・ポアソン・二項・ガンマ・逆ガウス）、ガンマ密度の書き換え、正規の σ²D*=RSS、ポアソン逸脱度式、補対数対数の逆関数、比例オッズ性）。scipy.stats で表の対数密度の再現、χ² 分位点・上側確率、ベータ二項分散、負の二項 pmf
- charts: 全13ファイルの front matter を yaml で読み、logistic-curve の β0=−2.048・β1=0.939（推定値の3桁丸め）と probit=1（実装が Φ((β0+β1x)/1.702)）、line-data（観測割合・ロジット/プロビット当てはめ値、ポアソン当てはめ値、IRLS の係数推移 t=0..5、プロファイル対数尤度12点を各 β1 で β0 を最大化して再計算、オッズ曲線、3つの逆リンク17点、分散関数、多項ロジット・順序ロジットの確率）、bar-data（d_i）、chi2 df=3、mosaic の表、pmf の λ=5.0164（x=2 の当てはめ値）・λ=4、nbinom r=1.161・p=0.225（=k/(k+4)、平均4・分散17.78）が本文と一致。1.702 で尺度をそろえたときの最大差 <0.01
- 乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記）: プロビット/ロジットの潜在変数（40万回、±0.004）、対数オッズ比のデルタ法 SE（セル20倍で ±0.002）、ポアソン−ガンマ混合の平均・分散・P(0)（100万回）、過分散データでポアソン SE が過小（比 1.75）、H0 下の D0−D1 の χ²₁ 近似（40万回、超過率 0.0551・平均 1.019）、補対数対数の P(N≥1)、多項ロジットの MLE が真値を再現、順序ロジットの潜在変数＋閾値
- 本文の修正: なし
- 検算スクリプト側の修正: デルタ法 SE のシミュレーションは元のセル（最小18）では近似誤差が 0.016 あるため許容を 0.03 にし、セル20倍での収束を追加。ロジット/プロビット比（1.680、1.668）は「約1.67」として許容 0.015。ポアソン図の当てはめ値は厳密な MLE と照合（ラベルの丸め係数 exp(0.7501+0.4313x) とは 6e−4 以内）。ウィルクスのシミュレーションを反復回数不足のないベクトル化ニュートン法・40万回に変更
- 備考: probit-model の Φ(−0.0935) は丸めた η を代入すると 0.46275（境目）、厳密な MLE では 0.46274 なので「0.4627」は正しい。generalized-linear-model の x=5 予測 18.30 は厳密な MLE で 18.2956（丸めた係数では 18.2945）。logistic-regression の η̂=−0.1698 も厳密値（丸め係数では −0.1697）。いずれも誤りではないので修正していない
- 検算対象外: 定性的な記述（現場での使われ方、準尤度の一般理論、完全分離での発散（小例で尤度の単調増加のみ確認）、IIA の不自然さの議論、AIC が準尤度で定義されないこと、ホスマー・レメショウ検定、図の見た目とスライダー挙動）

## 第20章（分散分析と実験計画法）

- スクリプト: `verify/ch20.py`（実行: `.venv/bin/python verify/ch20.py`）
- 対象: one-way-anova, sum-of-squares-decomposition, expected-mean-square, fixed-and-random-effects, tukey-multiple-comparison, two-way-anova-with-replication, two-way-anova-without-replication, main-effect-and-interaction, randomized-block-design, latin-square-design, orthogonal-array, factorial-design, confounding, split-plot-design, fisher-three-principles（15ファイル）
- 結果: **OK 198 / NG 0**（初回 OK 196 / NG 2。NG はすべて検算スクリプト側の問題）
- 方法: 全分散分析表（一元配置の肥料例・3群小例・ロット例、繰り返しあり/なしの二元配置、乱塊法、4×4 ラテン方格、L8 直交表、2² 要因計画、分割法）を numpy で定義式から計算し、修正項 CT 方式の中間値（5253、3267、37917、16291、9308.5 など）とも照合。F・p・F 分位点（4.256、8.022、5.143、5.987、4.757、18.513、7.709）は scipy.stats。ラテン方格と分割法の誤差平方和は最小二乗の残差でも確認。テューキー法は scipy.stats.studentized_range で q_0.05(3,9)=3.948・q_0.05(3,6)=4.339・HSD・q 値・調整済み p・同時信頼区間を再計算し、scipy.stats.tukey_hsd とも一致。テューキー・クレーマー例、ボンフェローニ境目 3.168・補正なし 2.443、q_α(2,ν)=√2 t_α/2(ν)、k に対する q の単調増加。sympy で一元配置 E[S_A]=Σn_iα_i²+(k−1)σ²（二次形式の期待値、不釣り合い n=(2,3,4)）・E[S_E]=(N−k)σ²、交差項の恒等的消滅、自由度の分解（二元配置・ラテン方格・分割法）。ラテン方格の行・列条件と行・列・処理の直交性（中心化ダミーの内積0）、L8 の全21列対の直交性・S'S=8I・成分記号と列の積の一致・交互作用列（1×2→3、2×4→6 など）、列平方和の合計＝S_T。2² の対比・効果・平方和と二元配置としての一致。交絡例の y をモデル＋ブロック差から再生成し7効果を再計算、ABC 以外がブロック内で釣り合うこと、I=ABC の別名関係。主効果・交互作用の分解（2×2 と 3×2、行和・列和0）。分散成分（σ̂_A²=0.3387、ρ̂=0.873、例2 の 7.667/12/6.5）、相対効率 11.18、非心 F の検出力 0.441、t 検定の検出力（非心 t）0.095/0.223/0.461/0.781、k=2 での F=t²（独立2標本・対応あり）
- charts: 全15ファイルの front matter を yaml で読み照合。bar-data（群平均、平方和3本と偏差2乗9本、分散成分、テューキーの差と HSD、V_E の比較3図、ラテン方格の処理平均、L8 列平方和、交絡例の効果、分割法の V_E1・V_E2）、line-data（交互作用プロットのセル平均、繰り返しなしと乱塊法のデータ、2×2 の平行例で交互作用0、2² のセル平均、σ/√r の10点）、F・ncf・studentized-range の params とキャプションの数値（9.571・0.0059、8.667・5.143、p=0.067 など）がすべて再計算と一致
- 乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記、標準誤差の4倍）: H0 下の一元配置 F の棄却率・F(2,9) への KS・S_A と S_E の無相関（20万回）、EMS 例1（V_A 平均 15.999、V_E 4.005。本文の 15.98・3.99 も許容内）と非心 F・検出力、二元配置の期待平均平方表を母数・変量・混合（制約付き／制約なし）の4通りで確認（10万回）、変量模型の E[V_A]=σ²+nσ_A²・σ̂_A² の不偏性・級内相関、テューキー法の FWER=0.05（0.0507）とボンフェローニの保守性、繰り返しなしで交互作用があるときの E[V_E]、分割法の期待平均平方表（20万回）、t 検定の検出力（r=4）、V[ȳ]=σ²/r
- 本文の修正: なし
- 検算スクリプト側の修正: expected-mean-square 例2・fixed-and-random-effects 例2 で、丸めた平均平方（65.333 など）を入力にしたため 7.666・32.666 と最終桁がずれた。丸め前の厳密値（V_A=196/3、V_A×B=52/3）で計算すると本文の 7.667・32.667・3.769・p=0.210 と一致
- 備考: split-plot-design の S_A×B=4.083・S_E2=3.333 は、本文の丸め値どうしを引くと 4.084・3.334 になるが、厳密値は 4.0833・3.3333 なので誤りではない
- 検算対象外: 定性的な記述（現場での使われ方、無作為化の役割、フィッシャーの3原則の説明、テューキー・クレーマー法の保守性の一般証明、グレコ・ラテン方格、分解能の概念、図の見た目とスライダー挙動）

## 第19章（回帰分析の発展：WLS・GLS・正則化・内生性・非線形・分位点・変数選択）

- スクリプト: `verify/ch19.py`（実行: `.venv/bin/python verify/ch19.py`）
- 対象: weighted-least-squares, generalized-least-squares, ridge-regression, lasso, bias-in-penalized-estimation, regularization, endogeneity, instrumental-variable-method, two-stage-least-squares, omitted-variable-bias, nonlinear-regression, quantile-regression, polynomial-regression, variable-selection（14ファイル）
- 結果: **OK 281 / NG 0**（初回 OK 277 / NG 4。本文側1件、検算スクリプト側3件）
- 方法: numpy/scipy で全数値例を再計算。WLS（X'WX・行列式 15.45・(0.9922, 2.0498)・OLS (1.0267, 2.04)・サンドイッチ分散 0.2000 と 0.1586（約21%減）・W^{1/2} 変換・重みの定数倍不変・例2 の個票35人 OLS との一致・重みなし 1.05）、GLS（AR(1) ρ=0.6 の (1−ρ²)Ω⁻¹ が三重対角、1'Ω⁻¹・重み・μ̂_G=10.125・V=0.5/0.5233、プレイス・ウィンステン変換 y*・x* と原点回帰 10.125、P'P∝Ω⁻¹、コレスキー白色化＝GLS、V_OLS−V_GLS の非負定値、等相関で 0.4545・GLS=OLS、固有空間条件）、リッジ（X'X・OLS・λ=1,5 の解とノルム、数値最適化との一致、固有値 19/1、λ=4 のバイアス²・分散・MSE、最適 λ=3.81・MSE 0.1375、d/(d+λ) の縮小、事後平均＝リッジ）、Lasso（直交計画の b=(3.1,2.1,0.3)、ソフト閾値 λ=1,2、座標降下法との一致、1変数問題の格子探索、例2 の λ/19・β2 が 12.6 で0・18.6 で全0・λ=10 の (0.7368,0.1368)、ラプラス MAP）、罰則付き推定のバイアス（例1 の表、Lasso の偏りと MSE を数値積分、β=0 の MSE 0.1507）、正則化（6次多項式リッジの係数ノルム・訓練 MSE・一個抜き CV を7つの λ で再現、標準化は各回の訓練点の母標準偏差（ddof=0）で本文の値と一致、最小は λ=0.3）、内生性（希薄化、需給モデルの plim、供給シフトの操作変数で −1 を回収）、IV（偏差積和・OLS 0.8133・IV 0.6000・切片・Z'X・相関 0.8783・s²=0.4733・SE 0.2558、ワルド＝s_zy/s_zx）、2SLS（第1段階係数・R²=0.9665・F=72.12・x̂・(1.3783, 1.1343)・OLS (1.5506,1.0699)・正しい σ̂²=0.1856/SE 0.1676 と第2段階そのままの 0.0683/0.1017、P_Z の対称べき等、丁度識別で IV と一致）、欠落変数バイアス（係数・δ̂1=2.9538・恒等式）、非線形回帰（ガウス・ニュートン1回目の J'J・J'r・更新量、収束値 (10.2452, 0.5062)・S=0.1080・SE・半減期 1.369、curve_fit と一致、対数線形化 (10.1968,0.5016)・S=0.1116）、分位点回帰（例1 の損失値と最小点、右微分、例2 を線形計画（highs）で解き切片・傾き・直線の上下の点の数、OLS 直線）、多項式回帰（1〜3次の S_e・R²・adjR²・AIC、中心化の正規方程式と γ、展開形、頂点 4.2085・9.7755、部分 F=1554.0・F=0.0843・p=0.790）、変数選択（8モデルの S_e・AIC・BIC・adjR²、回帰式、変数増加法/減少法の経路、e^{2/n} 条件と F≈2）。sympy で WLS の正規方程式の解、リッジの勾配・期待値・バイアス・直交計画の b/(1+λ)、Lasso の直交分解、直交計画リッジの MSE・導関数・λ*=σ²/β²・最小値・λ=0 の傾き、縮小平均 nȳ/(n+λ)、正規・ラプラス事前の MAP と罰則の対応、希薄化と需給モデルの plim・Cov(p,u)、ロジスティックの変曲点 K/2・ゴンペルツの K/e
- charts: 全14ファイルの front matter を yaml で読み照合。WLS・IV・非線形・多項式・分位点回帰の直線/曲線、GLS の重み、リッジの MSE 分解（9点）と λ≈3.8、罰則付き推定のバイアス²・分散・MSE、Lasso のソフト閾値・リッジ比較、正則化の log10 λ と訓練/CV 誤差、希薄化 2/(1+r)、欠落変数の 1+0.5ρ、2SLS の x と x̂、チェック関数、変数選択の AIC 棒と3本の経路がすべて再計算と一致。ridge-lasso-path の method、iv-scatter・dw-sim・residual-pattern・bias-variance の params がスライダー範囲内で本文と整合
- 乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記）: WLS の E・V・E[s_W²]（20万回）、GLS/単純平均の分散 0.5・0.5233（20万回、±1.5%）、リッジ λ=4 の E・MSE と OLS の MSE（20万回）、Lasso β=2 のバイアス（100万回）、測定誤差 n=10⁶（本文の 1.6009 は 4SE 以内）、同時性 −0.25（本文の −0.2499 も範囲内）、IV の一致性と分散 σ²/(n s_x² r²)（4000回、±6%）、外生性の破れの plim、2SLS（操作変数2個）の一致性、X 固定での E[β̃1]=β1+β2δ̂1
- 本文の修正:
  - `content/ch19/weighted-least-squares.md` charts の「WLS の直線（重み 1/x）」: [3.042, 5.0918, 7.1417, 9.1915, 11.2413, 13.2912] → [3.0421, 5.0919, 7.1417, 9.1916, 11.2414, 13.2913]（厳密な係数 15.33/15.45・31.67/15.45 による値。元の値は丸めた係数での計算で最終桁が 1e-4 ずれていた）
- 検算スクリプト側の修正: 丸め境界ちょうど（0.5/16=0.03125→0.0312）が浮動小数点で落ちるため比較に 1e-12 の余裕を追加。分位点回帰の図の直線は丸めた係数ではなく LP の厳密解（例：4/3+(23/30)x）と照合
- 備考: polynomial-regression の部分 F=1554.0 は丸め前の S_e で正しい（本文の丸め値 0.0610 を代入すると 1552.8）。omitted-variable-bias の 0.0646+0.0602×2.9538 は丸め値では 0.2424 だが厳密値の恒等式で 0.2423。いずれも誤りではないので修正していない。regularization 例2 の標準化（母標準偏差）は本文に明記がないが、標本標準偏差では表の値にならない
- 検算対象外: 定性的な記述（現場での使われ方、エイトケンの定理・ホーエル・ケナードの定理の一般証明（数値例・乱数で傍証のみ）、FGLS の漸近性、弱い操作変数の議論、分位点回帰の漸近理論、選択後推論の過大な有意性、図の見た目とスライダー挙動。dw-sim・residual-pattern・iv-scatter・bias-variance・ridge-lasso-path は乱数/固定データ描画なので params のみ確認）

## 第21章（標本調査法）

- スクリプト: `verify/ch21.py`（実行: `.venv/bin/python verify/ch21.py`）
- 対象: simple-random-sampling, stratified-sampling, proportional-allocation, neyman-allocation, cluster-sampling, two-stage-sampling, systematic-sampling, ratio-estimation, regression-estimation, horvitz-thompson-estimator, inclusion-probability, probability-proportional-to-size-sampling, design-effect, nonresponse-bias, sampling-and-nonsampling-error（15ファイル）
- 結果: **OK 201 / NG 0**（初回 OK 193 / NG 8。NG はすべて検算スクリプト側の問題）
- 方法: 本文の小さな母集団はすべて全標本を列挙して期待値・分散・MSE を独立に再計算し、公式値・近似値と照合。SRS（{3,5,6,8,9,11} の20通り：E=7・V=7/5・E[s²]=42/5 を有理数で厳密に、n=1..6 の分散も列挙）、層化（225通り・SRS 495通り・(1,3) 配分の 1.0）、集落（6通り、ρ=0.9161/−0.4825 を定義式から計算し (11/9){1+2ρ} が分散比と一致、ρ を使った分散式が 3.75 を再現）、二段（216通りで V=3.0885 と V̂ の期待値、(4,1)・(1,4)・SRS 1820通り、(3,2),(2,3),(3,3) でも公式と V̂ の不偏性、m=M で集落・a=A で層化に帰着）、系統（k 通り、π_i・π_ij）、比推定・回帰推定（20通り・924通りで偏り 0.0314/−0.0152、MSE 0.0785/0.1131/3.067/12.893/1.773/0.1474、b0 固定推定量の不偏性と分散）、HT（6通りで E=40・V=2.9167、HT 分散式・SYG 形・V̂_YG の期待値の一致、4ȳ の期待値 43.6）、PPS（順序付き16通りで E=200・V=136.25・V̂ 不偏、等確率復元 3228・非復元 2152）、包含確率（例1の表、Σπ=n・Σ_{j≠i}π_ij=(n−1)π_i、表の SRS・層化・集落・二段・系統を列挙で確認、Cov(Z_i,Z_j)）。数値例（例2の比率の SE・必要標本サイズ 695.8、層化例2の 41.667/2.2010/1.484/46.111、比例配分 4.41・S²=617.52・5.558・近似 1.161、比例配分の丸め、ネイマン 26.32/31.58/42.11・3.12・3.1203・費用つき (50,60,40)・2.21、deff 1.95/512.8/0.0216/0.042/1.25/0.793/0.561、無回答バイアス 4.8・5・Cov=3、MSE・被覆確率 0.235/0.908・カバレッジ偏り 0.04）。sympy で Cov(Z_i,Z_j)=−f(1−f)/(N−1)、V(ΣZ_iY_i)=f(1−f)NS²（N=5 の一般の Y）、0/1 での S²、σ² 流儀との一致、自己加重、ネイマン配分のラグランジュ乗数法（∂L/∂n_h=0 の解・√λ・n_h・最小値 (ΣWS)²/n・ヘッセ行列の正定値・費用つき n_h∝W_hS_h/√c_h・∂V/∂n_h）、ΣWS²−S̄²=ΣW(S−S̄)²、比推定の恒等式と条件 ρ>C_x/(2C_y)、比推定=原点を通る WLS（V∝x）の解、回帰推定の b0 の2次式の最小化、Kish の恒等式、交換可能相関の平均の分散、偏りの分解式。SLSQP で費用制約下の最適配分、ランダム配分2000通りがネイマン以上（コーシー・シュワルツ）、ランダム母集団200個で集落の分散比の恒等式、ランダムな固定サイズデザイン200個で HT の不偏性・分散式=SYG 形・V̂_YG の不偏性、0/1 母集団で超幾何分布と (1−f)p̂(1−p̂)/(n−1) の不偏性
- charts: 全15ファイルの front matter を yaml で読み照合。SRS の bar-data（20標本の度数）と line-data（非復元/復元の分散）、層化・比例配分・二段・比推定・回帰推定・HT・PPS・計画効果の bar-data（いずれも列挙または公式の再計算値）、ネイマン配分の line-data（17点）、集落の 1+(M−1)ρ、系統の2系列、包含確率の p(s)・π_i、無回答バイアスの (1−回答率)×差、標本誤差の SE と √MSE（7点）
- 乱数シミュレーション（seed=20261023、許容誤差は各チェックに明記）: SRS の V(ȳ)（10万回、相対2%）、比推定・回帰推定の近似分散（N=2000,n=100、4万回、相対3%/4%）、復元 PPS の HH 推定量の不偏性と分散（20万回）、Kish の加重平均の分散比 1.25（20万回）、確率的無回答モデルの偏り≈Cov(φ,y)/φ̄ と層の重み付け直しで偏り≈0（2000回）、MSE=V+Bias² と偏りのある CI の被覆率（40万回）
- 本文の修正: なし
- 検算スクリプト側の修正: allclose に比較先がスカラーのとき形が合わずに NG になる不具合を修正（包含確率の表・系統・PPS の5件）。SLSQP の収束誤差（約0.01）のため許容を 0.05 に。層化例1のネイマン配分 n1 の手計算値をスクリプト側で誤記（0.98→正しくは 1.097）していたのを修正。浮動小数点の == 比較を許容つきに変更
- 備考: horvitz-thompson-estimator 例2の「約 1/15」は 45.333/2.9167=15.5 で、近似表現として妥当とした
- 検算対象外: 定性的な記述（現場での使われ方、非標本誤差の種類、系統抽出で分散の不偏推定量が作れないことの一般論、MCAR/MAR/MNAR の議論、比推定の偏りの2次展開の一般導出（近似式の値のみ照合）、図の見た目）

## 第22章（主成分分析）

- スクリプト: `verify/ch22.py`（実行: `.venv/bin/python verify/ch22.py`）
- 対象: eigenvalue-and-eigenvector, principal-component-analysis, principal-component, lagrange-multiplier-pca, contribution-ratio, cumulative-contribution-ratio, kaiser-criterion, scree-plot, pca-on-correlation-matrix, principal-component-loading, principal-component-score, singular-value-decomposition（12ファイル）
- 結果: **OK 158 / NG 0**（初回 OK 153 / NG 5。すべて本文側の丸め誤り）
- 方法: S=[[5,2],[2,2]] の固有値 6・1、固有ベクトル (2,1)/√5・(1,−2)/√5、スペクトル分解、相関行列 1±r=1.6325/0.3675 と45度方向、26.57度。5人×2変数のデータから S を再現し、主成分得点・分散6/1・共分散0、負荷量 0.9798/0.7746/0.2/−0.6325、SVD（X_c'X_c=[[20,8],[8,8]]、d=√24・2、u1=(−8,−2,0,4,6)/√120、第1項近似の誤差4）。8人×4教科を numpy で再計算（平均・SD・分散、相関行列、S と R の固有値・寄与率・累積寄与率、det R=0.0041、固有ベクトル、負荷量表と2乗和、共分散行列の負荷量 0.856/0.842/0.944/0.947 と得点との直接相関、u4・z41=2.800・最大最小の生徒、共分散の z11=−16.95、再構成誤差 0.1776/0.0265=1−C_m、得点の2乗和=(n−1)λ_k、国語10倍の結果と相関行列の不変性、特異値 60.12/25.77/9.57/5.01 と d²/7=固有値、h=v・Z=UD、エッカート・ヤング）。固有ベクトルの符号は本文の向き（第1は全正、第2は国語が正など）にそろえてから比較。sympy で 2×2 の固有方程式・解の公式・(b, λ−a)、∂L/∂a=2Σa−2λa、2次形式の微分（非対称なら (A+A')a）、例1 の連立方程式、5cos²θ+4sinθcosθ+2sin²θ=3.5+1.5cos2θ+2sin2θ。第2主成分の μ=0 は 3×3 の数値例
- charts: 全12ファイルの front matter を yaml で読み照合。pca-axes の params はすべてスライダー範囲内、lagrange の line-data（25点）は 3.5+1.5cos2θ+2sin2θ と小数第4位で一致、bar-data（R の固有値、S の寄与率、例の固有値 2.4〜0.3・3.1〜0.22、特異値の2乗）、kaiser の line-data、points-2d（負荷量4点、主成分得点8点）が再計算と一致
- 乱数シミュレーション（seed=20261023）: 単位ベクトル20万本でレイリー商 ≤ λ1、h1 に直交な方向の最大が λ2（許容 1e-3）、pca-axes の ρ=0.5・n=200 で標本固有値の平均 ≈1.5/0.5（許容 0.03、500回）と主軸≈45度（許容1度）、ρ=0.8 で第1寄与率≈0.9（許容 0.01）、無相関 p=10・n=50 で固有値1以上の個数の平均が3以上（カイザー基準の過大推定）、1次元射影近似で主成分が誤差最小（乱数方向2000本）、乱数の階数2行列よりエッカート・ヤング近似の誤差が小さい
- 本文の修正:
  - `content/ch22/principal-component-analysis.md`・`principal-component.md`・`pca-on-correlation-matrix.md`: 相関行列の第1主成分係数の英語 0.492 → 0.491（厳密値 0.49149）
  - `content/ch22/principal-component-loading.md` 負荷量表の数学・第2主成分 −0.398 → −0.397（厳密値 −0.39750 の小数第3位。図の −0.3975 はそのまま）
  - `content/ch22/singular-value-decomposition.md` charts の bar-data: 3614.96 → 3614.95、664.34 → 664.33（厳密値 3614.9498、664.3331。元の値は丸めた固有値×7 由来）
- 検算スクリプト側の修正: スカラーの許容比較のブロードキャスト、1×1 配列の float 変換（いずれも実行エラー）
- 検算対象外: 定性的な記述（現場での使われ方、因子分析との違い、多変量正規なら独立になること、平行分析・0.7 基準、肘の位置の主観的判断、図の見た目とスライダー挙動）

## 第23章（判別分析）

- スクリプト: `verify/ch23.py`（実行: `.venv/bin/python verify/ch23.py`）
- 対象: linear-discriminant-analysis, mahalanobis-distance, fisher-linear-discriminant-function, bayes-discriminant-rule, misclassification-rate, quadratic-discriminant-analysis, confusion-matrix, sensitivity-and-specificity, precision-recall-f-measure, roc-curve-and-auc, k-nearest-neighbors, support-vector-machine（12ファイル）
- 結果: **OK 164 / NG 0**（初回 OK 161 / NG 3。本文側1件、検算スクリプト側2件）
- 方法: 共通データ G1=(1,2),(4,5),(1,5),(2,4)、G2=(3,5),(6,8),(6,5),(5,6) を numpy で再計算（平均、偏差平方和積和 [[6,3],[3,6]]、併合 S=[[2,1],[1,2]]・S⁻¹・|S|=3、a=(4,1)/3、z=(4x1+x2−19)/3、z(4,4)=1/3 と log9 のしきい値、見かけの誤判別 2/8（(4,5)・(3,5)）、一つ抜き法（各点を除いて併合 S から作り直し、同じ2点だけ誤判別・(4,5) の得点 2.222）、D²=14/3・Φ(−D/2)=0.140、(4,4) の D1²=8/3・D2²=2・ユークリッド 2/2.236、混同行列 [[3,1],[1,3]]、得点 4x1+x2 の経験 ROC・AUC=15/16（面積・組の数え上げ・scipy の Mann–Whitney U=15）、Φ(D/√2)=0.9367、kNN の距離順位表と k=1,3,5 の判別、k=3 の一つ抜き（第3近傍の同順位はすべての選び方を試し結果が一意）。フィッシャー比の表（4.5/2/4.167/4.667、169/38）、有界最適化で最大方向 ∝(4,1)・最大値 14/3、B/W 比との定数倍関係。ベイズ判別の例（境界 124.815、事後確率 0.1909・0.5826・0.1343（密度から直接計算と一致）、誤判別 0.0065/0.3020/0.0213/0.0668、格子探索で 124.815 が最小）、誤判別率 Φ(−1±c/Δ)=0.0179/0.5393・全体 0.0701、二次判別の根 −2.571/1.238・g(−3)=0.682・誤判別 0.113/0.340/0.227 と中点の 0.234、感度・特異度表（J・LR⁺）と LR⁻・検査後確率、適合率・再現率・F値の表と F₂=0.7018、正規得点のしきい値ごとの R/P/F、マハラノビス例2（2.5/22.5、χ²₂ 上側確率 e^{−c/2}、9.210）。sympy で一般の 2×2 Σ について log f2/f1=z（ヘッセ行列=0）、D1²−D2²=2z、フィッシャー比の勾配が a=Σ⁻¹d で 0・最大値 d'Σ⁻¹d、事後確率のロジスティック形、二次判別の 8 倍した式。p=3 や異なる Σ の対数事後比の式は乱数の数値例で確認。AUC=∫P(S1>t)f0(t)dt=Φ(Δμ/√(σ0²+σ1²)) を数値積分（3組）と ROC 曲線の台形積分で確認。SVM の6点の例は主問題・双対問題・スラック付きソフトマージン（C=100）を scipy.optimize の SLSQP で解き、w=(2/3,2/3)、b=−5/3、α=(0,2/9,2/9,4/9,0,0)、サポートベクトル3点、y_i f(x_i)、幅 3/√2、Σα=‖w‖²=8/9、(1.5,1.5)・(2,0) の f=±1/3 を確認（許容 1e-5〜1e-6）
- charts: 全12ファイルの front matter を yaml で読み照合。誤判別率の line-data（Φ(−Δ/2)、9点）、適合率・再現率・F値の line-data（17点×3系列）、ROC の line-data（経験 ROC の点列と対角線）、mosaic の表 [[80,20],[90,810]]、bayes-area（混同行列・感度特異度・適合率の (0.1,0.8,0.9)、ベイズ判別の感度 Φ(0.5185)=0.698・特異度 Φ(2.4815)=0.993）、roc-curve（(0,2,1,1) と caption の 0.159・0.841）、その他 custom 図の params がスライダー範囲内
- 乱数シミュレーション（seed=20261023）: Δ=2 の2変量正規20万点で P(2|1)≈Φ(−1)、c=log9 の P(2|1)・P(1|2)（許容 0.003）、群1の z の平均−2・分散4（許容 0.03）、N₃ で D²~χ²₃（10万点、KS p>0.01）、π2f2>π1f1 ⇔ z>log9（2000点）、経験 AUC≈0.9214（各3000点、許容 0.01）と単調変換での不変性
- 本文の修正:
  - `content/ch23/precision-recall-f-measure.md` charts の caption:「F値は t=1.75 付近で最大（約0.611）」→「t=1.7 付近で最大（約0.612）」（連続的に最大化すると t=1.689 で F=0.6121。0.611 は格子点 t=1.75 の値 0.6112 だった。line-data の値と本文例2の t=1.75 の値はそのまま）
- 検算スクリプト側の修正: numpy 2 の np.cross が2次元ベクトルに使えない実行エラー（自前の外積に変更）、kNN 一つ抜きで第3近傍の同順位を NG としていたのを全選び方の列挙に変更（結果は一意）、ソフトマージンの Nelder-Mead が非平滑で収束しきらないためスラック変数つき SLSQP に変更（SLSQP の success フラグは最適点での直線探索警告のため判定から除外し、解と目的関数値で判定）
- 備考: misclassification-rate の「Φ(0.099)≈0.5393」は引数を丸めた表記で、厳密な Φ(1.0986−1)=0.5393 と一致（Φ(0.099) そのものは 0.5394）。誤りではないので修正していない
- 検算対象外: 定性的な記述（現場での使われ方、カーネル法・次元の呪い・C の役割の一般論、ボックスの M 検定、二次判別の過学習、ベイズ判別の最適性の一般証明（1変数の格子探索で傍証のみ）、図の見た目とスライダー挙動）

## 第24章（クラスター分析）

- スクリプト: `verify/ch24.py`（実行: `.venv/bin/python verify/ch24.py`）
- 対象: hierarchical-clustering, single-linkage, complete-linkage, average-linkage, ward-method, centroid-method, lance-williams-formula, dendrogram, k-means-method, non-hierarchical-clustering, number-of-clusters, distance-measures, similarity-measures（13ファイル）
- 結果: **OK 177 / NG 0**（初回 OK 173 / NG 3。本文側の記述不足1件）
- 方法: 5店舗の距離行列で、ランス・ウィリアムズの更新式を自前実装して各段の距離表（最短 11/8/5→4/8→5、最長 12/9/5→7/12→12、群平均 11.5/8.5/5→5.5/10→8.5）・併合順・高さを再現し、scipy の linkage（single/complete/average/weighted）の高さ・2クラスターの切り方と照合。群平均の全6組平均 51/6、WPGMA の 7.75、最小全域木の辺 2,3,4,5（合計14）=最短距離法の高さ、直径 5/3/7。ウォード法は5点の d²・Δ 表、段2〜4 の Δ（6, 27.333, 48, 22.833, 55.25, 55.5）を ESS の差と公式の両方で、W の推移 0→2→4.5→10.5→66=全体平方和 38.8+27.2、更新式の値=2Δ（4,5,12,111）、scipy の √(2Δ)=2, 2.236, 3.464, 10.536。重心法は C(1,1.8) の5点で 3.24/41/72/29.84/53.64/5、36.56/65.16、49.61、scipy centroid の 2→1.8→2.236→7.043 の逆転、同じ5点のウォード法 2.16 で逆転なし。ランス・ウィリアムズの係数表7手法は、乱数 9点×40回で全段の更新値を直接計算（min/max/全ペア平均/重心間2乗距離/中点の2乗距離/2Δ）と照合し、WPGMA・メディアン法は scipy の高さとも照合。単調性の十分条件の成否と、乱数 8点×200回で単調手法は逆転なし・重心/メディアン法は逆転ありを確認。デンドログラムの切断（高さ 1/2.5/4/8）、コーフェン距離 c_AC=12 等、コーフェン相関 0.7214/0.7638/0.7712（scipy と相関の直接計算）、超距離不等式。k-means 例1 の各反復の2乗距離表・割り当て・重心・W（36→22.667→16）と収束、全31通りの2分割で W=16 が一意の最小、例2 の局所解 W=28。k-medoids の15通りで J=8.819（BC・BF）、S(n,2)=2^{n−1}−1、1次元の外れ値例。9点の全分割（ベル数、k≤6 で S(9,k) と件数一致）を総当たりして W(k)、減少量、平均シルエット係数、点 (1,1) の a=1・b=6.154・s=0.8375、B+W=全平方和。距離（ミンコフスキー各次数、単位変更の例、q<1 と2乗距離の三角不等式の破れ）、類似度（a,b,c,d、単純一致・ジャッカード・ダイスと scipy、D=2J/(1+J)、コサイン 0.7638・相関 0.3162）。sympy で min/max の恒等式、Δ の導出、重心法の更新式（スチュワートの定理）、平均が2乗和を最小にすること
- charts: 全13ファイルの front matter を yaml で読み照合。ward の W 推移、centroid の高さ（小数第4位）、lance-williams の3系列、number-of-clusters の W(k)・シルエット、distance の line-data（7点）、similarity の bar-data が再計算と一致。dendrogram（method と n）、kmeans-steps、minkowski-ball の params がスライダー範囲内。さらに `src/charts_custom.js` の dendrogram 描画アルゴリズムを Python に移植し、乱数 50組（4〜16点）で4手法とも scipy の高さと一致（ward は √(2Δ) を描くので caption と整合）
- 乱数（seed=20261023、決定的な等式なので許容 1e-9）: 上記の更新式と直接計算の一致、MST=最短距離法、最長距離法の直径 ≤ 高さ、群平均法の高さ=全ペア平均、k-means の W が割り当て・更新の両段で増えないこと（30点×100回）、q≥1 の三角不等式（2000組）、相関=中心化ベクトルのコサイン（500組）
- 本文の修正:
  - `content/ch24/number-of-clusters.md` 例1: k=4, 5 では W(k) を最小にする分割が複数（6個・12個）あり、平均シルエット係数がそれぞれ 0.5470〜0.5571、0.3146〜0.3301 と分かれる。本文の 0.5571・0.3301 はその最大値だったので、「W(k) を最小にする分割が複数ある k=4,5 では、その中で平均シルエット係数が最大の分割の値」と注記を追加（数値・図は変更なし。どの分割でも k=3 が最大という結論は不変）
- 検算スクリプト側の修正: シルエットの比較を「W 最適分割の中の最大値」との比較に変更
- 検算対象外: 定性的な記述（現場での使われ方、鎖状効果・外れ値への強さの一般論、計算量、ギャップ統計量・カリンスキー・ハラバズ指数の選択基準としての性質、混合正規分布の EM の極限が k-means になること、図の見た目と乱数で描かれる点の配置）

## 第25章（因子分析・グラフィカルモデル）

- スクリプト: `verify/ch25.py`（実行: `.venv/bin/python verify/ch25.py`）
- 対象: factor-analysis-model, common-factor-unique-factor, factor-loading, communality, factor-rotation, varimax-rotation, promax-rotation, factor-score, number-of-factors, maximum-likelihood-factor-analysis, graphical-model, undirected-graphical-model, precision-matrix, covariance-selection, directed-acyclic-graph（15ファイル）
- 結果: **OK 148 / NG 0**（初回 OK 141 / NG 7。本文側1件、検算スクリプト側6件）
- 方法: 4教科2因子モデル Λ=(0.8,0.2),(0.7,0.3),(0.2,0.8),(0.3,0.7)、Ψ=diag(0.32,0.42,0.32,0.42) で ΛΛ'+Ψ（R 全成分・対角1）、共通性 0.68/0.58、列の2乗和 1.26・63%、自由度 {(p−m)²−(p+m)}/2 とパラメータ数え上げの一致（p≤14）、1因子例のテトラッド 0.3024 と λ1²=0.81。回転は初期解 A=ΛT（T=(1/√2)[[1,1],[1,−1]]）、AT=Λ、30° 回転の国語 (0.7928,−0.2268)、バリマックス基準 V(0)=0.0025・0.01° 刻みで ±45° のみ最大 0.13125、u・v・Σ(u²−v²)=−0.4824・分母 −4.12 → θ=45°。tan4θ 公式は sympy で dV/dθ=(分子·cos4θ−分母·sin4θ)/p² を確認し、乱数 200 組で atan2 の θ が格子探索（0.05°）の最大と一致。プロマックス（k=4）は目標 P、U=[[1.1774,−0.3843],…]、Φ の対角1・φ=0.590、パターン AU、構造 ΛΦ=(0.823,0.438)、ΛΦΛ'=AA'、カイザー正規化つきバリマックスでも ±45°。SMC（R⁻¹ 対角 2.6432…、SMC 0.6217…）を回帰の R² と照合、乱数 300 組で SMC≤共通性。因子得点は λ/ψ、s=7.5642、重み2系列、f̂_B=0.893・f̂_R=0.789（Λ'R⁻¹x でも）、s/(1+s)=0.883・1/s=0.132、乱数でウッドベリー公式・回帰法2形・バートレットの WΛ=I。固有値 2.3712/0.8752/0.3848/0.3688、|S|=0.2004・T=316.40・χ²₆ 12.59。最尤1因子は L-BFGS-B（10初期値）＋BFGS 仕上げで F=0.012343、λ̂・Ψ̂（小数第3位）、対角一致、tr(SΣ̂⁻¹)=p、残差6組、T=2.421・p=0.298・5.991、2(ℓ(S)−ℓ(Σ̂))=nF、尺度不変（DSD で λ̂ が D 倍）。グラフィカルモデルは連鎖 Ω と Σ、相関 0.577/0.333、偏相関 0.5/0.5/0、合流の Ω と ρ12·3=−0.5、4変数連鎖 R⁻¹（小数第4位、(1,3)(1,4)(2,4) が厳密に0）と偏相関 0.545/0.275/0.756、分離集合を与えた偏相関0（ρ14·2・ρ13·2・X1⊥{X3,X4}|X2）、回帰係数 (0.5,0)・残差分散 1/ω11。乱数 200 組で ρ=−ω_ij/√(ω_ii ω_jj)・条件付き共分散=Ω_aa⁻¹・回帰係数・残差分散、sympy で 3変数公式との一致と 2×2 逆行列の符号。共分散選択は偏相関 0.504/0.355/0.1443、σ̂13=0.30・Σ̂⁻¹ の (1,3)=0、|S|=0.47・|Σ̂|=0.48・dev=2.105 と −n log(1−r²) の一致（1e-10）、p=0.147・3.841、ω13=0 制約の最尤を数値最大化しても Σ̂、反復比例スケーリングを自前実装し乱数 30 組（3〜5変数）で 1辺除去の dev=−n log(1−r²)。DAG 例1 の周辺・条件付き確率（0.5, 0.11, 0.073, 0.146, 0.074, 0.16）、パラメータ数 5/7・1023/19
- charts: 全15ファイルの front matter を yaml で読み照合。points-2d（Λ、回転前 A・回転後 AT、プロマックス6点 0.7089/0.0842 等）、bar-data（共通性・独自分散・固有値・ML 残差・偏相関）、line-data（SMC、バリマックス基準 19点、因子得点の重み）、rotation-angle（loadings=A、`src/charts_custom.js` と同じ回転 L·T(θ) で整数角の最大が ±45°）、graph-network（辺と edgeLabels が偏相関、有向・無向の別）
- 乱数（seed=20261023）: x=Λf+ε 40万点で標本共分散≈ΛΛ'+Ψ・Cov[x,f]≈Λ（許容 0.01）、因子得点 20万人で Var(f̂_R)≈0.883・Var(f̂_B−f)≈0.132・MSE_R<MSE_B（許容 0.01）、1因子が正しいとき T の平均≈2・棄却率≈0.05（400回、許容 0.3・0.03）、合流 40万点で r12≈0・ρ12·3≈−0.5（許容 0.01）、共分散選択 dev の平均≈1・棄却率≈0.05（2万回、許容 0.05・0.008）、DAG 例1 40万日（許容 0.005）
- 本文の修正:
  - `content/ch25/factor-score.md` 例: 回帰法の和の第2項「0.1298」→「0.1297」（0.25947×0.5=0.12974。結果 0.789 は不変）
- 検算スクリプト側の修正: dV/dθ の比例定数（1/32 → 1/p²=1/16）、カイザー正規化バリマックスの列の入れ替わり（±45° の同値解）を許容、edgeLabels の文字列と浮動小数の比較、sympy の 2×2 逆行列で正定値の仮定を明示、スカラーとの allclose の形状不一致（2件）、最尤推定の仕上げに BFGS を追加
- 検算対象外: 定性的な記述（現場での使われ方、ハマーズリー・クリフォードの定理・d分離・モラル化の一般論、カイザー基準の過大過小の傾向、ヘイウッドケースの原因、主因子法の反復、グラフィカル Lasso、図の見た目とスライダー挙動）

## 第26章（その他の多変量解析）

- スクリプト: `verify/ch26.py`（実行: `.venv/bin/python verify/ch26.py`）
- 対象: multidimensional-scaling, classical-mds, canonical-correlation-analysis, canonical-variate, correspondence-analysis, quantification-theory, quantification-type-1, quantification-type-3, structural-equation-modeling, path-analysis, independent-component-analysis（11ファイル）
- 結果: **OK 168 / NG 0**（初回 OK 160 / NG 7。すべて検算スクリプト側）
- 方法: 書き手の計算スクリプトは使わず numpy/scipy で独立に再計算。固有ベクトル・座標の符号は軸ごとに主張値に合わせ、連動する組（CA の行と列、III類の人と項目、正準変量の a_k と b_k）は同じ符号反転だけを許して照合
  - MDS: 長方形4点（行・列・全体平均 −6.25、B、固有値 16/9/0/0、座標と距離の完全再現）、4駅（固有値 38.853/3.955/0/−1.058、当てはまり 0.908、第1次元座標、2次元の復元距離 2.254・8.020）、一直線の4駅（B=xx'、固有値46、トレース 184/4）、三角不等式を破る3点（B、固有値 8.083/0/−1.083、座標、0.882）、成分式=JAJ、J1=0、固有値=座標の2乗和、導出の途中式（ā_i.=−(c_i+c̄)/2 等）。乱数で「座標の距離→古典的MDS=主成分得点（符号を除く）」、ユークリッド距離なら半正定値・正の固有値数=次元（50組）
  - 正準相関: R12R22⁻¹R21、M、tr 0.3229・det 0.000523、ρ²=0.3212/0.0016、a1・b1、もう一方の固有値問題の一致、Nelder-Mead による相関の直接最大化、ラグランジュ条件、q=1 で ρ1²=0.2976=重回帰の R²（乱数20組で恒等式）、ρ1≥群をまたぐ最大|r|（乱数200組）、正準負荷量 (0.924,0.719)/(0.962,0.548)、交差負荷量、U・V の相関構造、バートレット（96.5、37.55・0.157、9.488・3.841）、ウィルクスのラムダ 0.678=|R|/(|R11||R22|)
  - CA: 周辺比率、e11=17.647、χ²=33.481（scipy）、イナーシャ 0.1969、S、γ=0.4393/0.0632、97.97%、主座標、推移公式（主張値と厳密）、Σr F=0・Σr F²=γ²、主座標の距離=χ²距離、自明解（特異値1、√r・√c）
  - 数量化: I類1アイテム（スコア、16.760/22.219=0.754、レンジ 3.333）、II類（η²=0.1534 を数値最大化でも確認、χ²=12.275、φ²=η²）、I類2アイテム（X'X・X'y、b、基準化スコア、b0=ȳ、レンジ、予測 8.591、R²=0.988、全ダミーの階数落ち）、III類（H、固有値 1/0.7566/0.25/0.0490、自明解、スコア、10組の相関=ρ1、CA の特異値²との一致）
  - SEM: 丁度識別の λ=(0.6,0.8,0.7)・ψ・df=0、4指標を BFGS で最尤推定し λ̂・ψ̂・F_ML=0.01430・χ²=2.845・p=0.241・RMSEA=0.046・最大残差 x1-x4 0.062・Σ̂ の対角=1
  - パス解析: c'・b・間接効果・総合効果=r_xy、R²=0.373、誤差パス 0.792/0.866、例2 の 0.30
  - ICA: Σx の固有値、Σx^{-1/2}、Σx^{-1/2}A の直交性と −26.57°、尖度 −0.6/−0.816、κ(θ) の式と −1.2/−0.9/−0.6、一様・ラプラスの超過尖度（scipy）
- charts: 全11ファイルの front matter を yaml で読み照合。MDS の points-2d（4駅、小数第4位）、classical-mds の bar-data（固有値）、正準負荷量の points-2d（第2軸も含む、単位円）、bvnorm の rho=0.57≈ρ1、CA の points-2d（行・列の主座標）と軸ラベルの寄与率・mosaic の表、数量化の bar-data（2件）と caption のレンジ、III類の points-2d（人・項目の第1・第2軸）と軸ラベル・馬蹄形の説明、SEM・パス解析の graph-diagram の weights、ICA の line-data（3系列 13点）
- 乱数（seed=20261023）: CCA 20万点で ρ1≈0.567（許容 0.01）、H0 でのバートレット統計量（4000回、平均≈4・棄却率≈0.05、許容 0.15・0.012）、パス解析 20万点で相関と標準化偏回帰係数（許容 0.01）、ICA 20万点で −1.200/−0.898/−0.603（許容 0.02）・混合の尖度・|κ| 最大方向で源信号を復元（|相関|>0.99）・正規の源信号は尖度0（許容 0.03）・キュムラントの加法性（許容 0.05）
- 本文の修正: なし
- 検算スクリプト側の修正: スカラーとの allclose の形状不一致（3件）、正準負荷量の点は p=q=2 では厳密に単位円上なので表示の丸めで 1e-4 まで許容、丸めた主張値を入力にした再計算（バートレット 37.54、III類 A の −0.9075）は入力の丸めの伝播分を許容（厳密値ではそれぞれ 37.55・−0.908 で一致）
- 検算対象外: 定性的な記述（現場での使われ方、非計量MDSのストレス、数量化IV類、識別の t 規則の十分性、同値モデル、ネゲントロピー、図の見た目とスライダー挙動）

## 第27章（時系列解析）

- スクリプト: `verify/ch27.py`（実行: `.venv/bin/python verify/ch27.py`）
- 対象: time-series, weak-stationarity, white-noise, autocovariance-autocorrelation, correlogram, ar-model, ma-model, arma-model, stationarity-condition, invertibility, yule-walker-equations, partial-autocorrelation, spectrum, forecasting, arima-model, unit-root, seasonal-adjustment（17ファイル）
- 結果: **OK 169 / NG 0**（初回 OK 165 / NG 3。3件とも検算スクリプト側の誤り、本文の誤りなし）
- 方法: ARMA の ψ 重みを自前で計算し、自己共分散 γ(h)=σ²Σψ_jψ_{j+h}（4000項）を共通の基準にして、AR(1)（μ=5・γ(0)=1.5625・ρ=0.6^h）、AR(2) 0.5/0.3（ρ=0.7143/0.6571/0.5429、分母 0.4457・γ(0)=2.2436）、MA(1)・MA(2)（1.3076/0.38/−0.24、ρ=0.2906/−0.1835）、MA(q) 公式（q=1〜3）、ARMA(1,1)（2.08/1.44、ρ=0.6923/0.3462/0.1731、4組で γ(0)・γ(1)・γ(h)=φγ(h−1)）と照合。sympy で平均 c/(1−φ…)、ARMA(1,1) の連立方程式の解、π 重み（(1−φz)/(1+θz) の展開で 0.9/−0.36/0.144）、10θ²−29θ+10=0 とその解 0.4/2.5、θ と 1/θ（σ² も変えて）で同じ自己共分散、|1−φe^{−iλ}|²。特性根（1.1736/−2.8403、λ 版 0.8521/−0.3521、0.4±1.3565i・|z|=√2、0.9221/−5.4221、(1−z)(1−0.2z)、(1+0.8z)(1−0.3z) の −1.25/3.3333）、AR(2) の三角形条件と「根がすべて単位円の外」の同値を約1万格子点で確認、例3 の減衰振動 ρ(h)=A(√0.5)^h cos(ωh+c)（ω=1.2840、周期 4.89）が h=0..12 で厳密に成立。偏自己相関はダービン・レビンソンを自前実装し、k×k 連立の直接解・残差どうしの相関・v_k 更新式と照合（例1 0.5/0.2/0.0556、AR(2) で φ22=0.3・k≥3 で0、MA(1) 0.4412/−0.2417/0.1406、MA(1) の PACF 公式を4値×10ラグで照合）。ユール・ウォーカー（0.515625/0.140625、σ̂²=1.2547、AR(1) 1.28、区間 [0.4432, 0.7568]）。スペクトルは AR(1)・MA(1) の表の式を ARMA 公式と照合し、6モデルで ∫e^{ihλ}f(λ)dλ=γ(h)（h=0..4、quad、許容 1e-7）、定義の和の打ち切り＝公式、f(0)=0.9947・f(π)=0.0622・f(π/2)=0.1170・∫f=1.5625、MA(1) 0.4074/0.0255。予測は AR(1) の漸化式＝μ+φ^h(X_n−μ)、誤差分散 1/1.36/1.4896＝σ²(1−φ^{2h})/(1−φ²)、区間、MA(1)・AR(2) の例、ARIMA(1,1,0) の 103.6/103.72/103.744→103.75 と ψ 重み 1/1.2/1.24・誤差分散 2.44/3.9776、指数平滑 50→50.6→50.72 と ARIMA(0,1,1) の予測＝単純指数平滑（乱数系列で全時点一致）。標本自己相関の手計算例（1.5/0.375/0/−0.75）、リュング・ボックス Q=5.844・p=0.322・χ²₅ 11.070・BP 5.58、バートレットの帯 0.2323、中心化移動平均・季節成分（−3.3594/3.6406/12.0781/−12.3594）・季節差分、DF 例（τ=−1.667）、半減期 6.58/13.51
- charts: 全17ファイルの front matter を yaml で読み照合。line-data は ARMA(1,1)/AR(1) の理論 ACF（11点）、定常域の3境界（17点）、MA(1) の ρ(1)（25点）、AR(1) φ=±0.6 とホワイトノイズのスペクトル（x=kπ/12 の13点）、AR(1) の予測値と95%予測区間の上下限（小数第3位）、時系列の中心化移動平均（両端は最も近い値で埋め）、季節調整済み系列（小数第2位）を式から再計算して一致。arima-model の乱数系列は、原系列の差分＝差分系列（丸め許容 0.011）と、差分系列が AR(1) φ=0.5 として妥当（r1・残差 SD）を確認。acf-plot・ar-sim の params はスライダー範囲内（ar-model の φ1=0.4, φ2=−0.5 は複素根で定常、stationarity-condition の 0.9/0.2 は三角形の外）
- 乱数（seed=20261023）: AR(2) n=3000 で最小二乗≈YW≈真値（許容 0.01/0.05）、MA(1) n=20万の標本 ACF（許容 0.01）、MA(1) の条件付き最小二乗（ε0=0 の逐次計算、許容 0.04）、WN の √n·r1 の SD≈1・帯外率≈0.05（許容 0.04/0.012）、n で割った γ̂ の行列は非負定値（500回）・n−h で割ると崩れる例あり、YW 推定（n で割る）は常に定常（500本）、AR(2) で φ̂33 の √n·SD≈1（許容 0.08）、AR(1) で n·V[φ̂]≈0.64（許容 0.05）、AR(1) 3期先の平均・分散（許容 0.01/0.02）、ピリオドグラムの平均≈SD≈1/(2π) が n=128/1024 で変わらない（許容 0.012/0.015）、LB(10) の平均≈10・棄却率≈0.05（許容 0.3/0.012）、MA(1) のバートレット n·V[r2]≈1.389（許容 0.08）、反転による ε_t の復元、ランダムウォークの分散・相関 0.7071/0.9535・Cov=min(s,t)（許容 1.5/0.006/0.15）、見せかけの回帰で |t|>1.96 が 83.5%
- DF 棄却点: H0（ランダムウォーク）で n=1000 を各2万回シミュレーションし、5%点 −1.943（定数項なし）/−2.854（定数項）/−3.413（定数項＋トレンド）。モンテカルロ誤差（SE≈0.005）と有限標本の差を見込み許容 0.03 で −1.95/−2.86/−3.41 と一致。H0 で −1.645 を使うと棄却率 45.5%（定数項あり）
- 本文の修正: なし
- 検算スクリプト側の修正: スカラーとの allclose の形状不一致（3件、0 との比較）
- 検算対象外: 定性的な記述（現場での使われ方、強定常と弱定常の関係・エルゴード性、ADF・KPSS 検定、X-12-ARIMA、SARIMA の推定、YW 推定の小標本バイアスの一般論、図の見た目とスライダー挙動）

## 第28章（分割表）

- スクリプト: `verify/ch28.py`（実行: `.venv/bin/python verify/ch28.py`）
- 対象: contingency-table, two-by-two-table, relative-risk, risk-difference, odds-ratio-interval, mcnemar-test, yates-correction, loglinear-model, hierarchical-loglinear-model, sampling-schemes, three-way-table, mantel-haenszel-estimator, mantel-haenszel-test, simpsons-paradox（14ファイル）
- 結果: **OK 197 / NG 0**（途中の NG は本文側1件、検算スクリプト側4件）
- 方法: statsmodels は使わず numpy/scipy/sympy で独立に再計算
  - 2×2表: RR 2.000・RD 0.150・OR 2.429、分母 69,750,000・χ²=6.452・p=0.0111（scipy の χ² とも一致）、sympy でセルのずれ ±(ad−bc)/n・Σ1/Ê=n³/(n1n2m1m2)・簡便式・OR=RR(1−p2)/(1−p1)、プール z=2.540 と z²=χ²、イェーツ 5.620/0.0178、フィッシャー 0.0171
  - 相対リスク: SE 0.3391（0.03667+0.07833）、対数区間 (0.0285,1.3578)→(1.029,3.888)、z=2.044・p=0.041、デルタ法の式（sympy）、症例対照 2.176 と OR 2.429 の不変性
  - リスク差: SE 0.02846、区間 (0.0042,0.1158)、NNT、RD=p2(RR−1)（sympy）
  - オッズ比: V=0.10417・SE 0.3227・区間 (0.3483,1.6134)→(1.417,5.020)・z=3.039・p=0.0024、条件付き最尤 2.653 と正確区間 (1.358,5.313) を `scipy.stats.contingency.odds_ratio` と非心超幾何の E[a]=a の根で確認、多項のデルタ法 gᵀΣg=Σ1/(np)（sympy）、0.5 補正例 14.55・SE 1.515・(0.747,283.4)・フィッシャー 0.047
  - 分割表: χ²=33.667・p≈8.7e-7・V=0.265・ピアソン残差 ±3.0
  - マクネマー: 5.143/0.0233、修正 4.321/0.0376、正確 0.0357、SE 0.0515・区間 (0.019,0.221)、対応無視 2.899/0.0887、分散式（sympy）
  - イェーツ: 5.013/0.0252、3.683/0.0550、フィッシャー 0.0536、片側 0.0268/0.0126/0.0275
  - 対数線形: 2×2 飽和モデルの λ 分解（3.7902, ±0.1014, ∓0.4479, 0.2452, 4λ=log OR, 端点制約 λ22=0.9808）、3×3 独立 G²=34.299・p≈6.5e-7、λ̂B、ポアソン回帰（ニュートン法）の μ̂ が制約によらず一致
  - 階層モデル: パラメータを数える関数で例1の自由度 17/15/8/6、sympy で表の自由度の一般式、IPF を numpy で実装し 8 モデルの G²（332.848, 106.744, 3.389, 1.049 ほか）・df・p 値、層ごとの G² 2.619+0.770、入れ子差 2.340・p=0.126、IPF とポアソン回帰の G² 一致
  - 抽出モデル: 2.656e-4、0.07952、0.003340（多項と一致）、0.13674、0.02443（積二項と一致）、0.13598、0.17966（超幾何と一致・成功率に依存しない）、sympy で Π Po/Po(総数)=多項、自由度の3式
  - 3元表: 部分表 OR 1・1、周辺 OR 2.25・χ²=12.0・p=0.0005、C×A・C×B 表、条件付き独立の期待度数=IPF
  - MH 推定量: R_k・S_k・2.500・重み (0.3,0.5,0.2)・RGB 0.02021+0.05300+0.03283=0.10605・SE 0.3256・(1.321,4.733)・ウルフ 2.500、シンプソン例 1.377・併合 0.508・SE 0.2081・(0.916,2.071)
  - MH 検定: E_k・V_k（scipy の超幾何と一致）・ΣV=10.1983・7.943/0.0048・修正 7.085/0.0078、シンプソン例 91.25/176.25・6.0033/18.2213・2.322/0.128、K=1 の式（sympy）
  - シンプソン: 回復率・加重平均・層別 OR 2.111/1.227・併合 0.508・MH 1.377
- charts: 全14ファイルの front matter を yaml で読み照合（mosaic の表 8 枚、RR の line-data の OR=2(1−p2)/(1−2p2)、bar-data（発現率、階層モデル G² 8本、層 OR と MH、a−E、回復率）、obs-exp-bar の observed/expected/ddof、pdf N(0.9808,0.3227)、pmf Bin(28,0.5)・hypergeom(24,13,8)、chi2 df=1 と caption の 7.94/0.005・0.036）
- 乱数（seed=20261023）: log RR̂・log OR̂ の分散（n×10 で漸近近似、相対許容 3%）、RD の分散（相対 2%）、マクネマーの分散式（相対 2%）と H0 での b|(b+c) の二項性（許容 0.006）、ポアソンを総数で条件付けた頻度（400万回、許容 0.0003）、ランダム分布で同時独立⇒条件付き・周辺独立、[AB][C] で OR 不変、MH の一致性と RGB 分散（許容 0.03・相対 8%）、H0 での CMH 棄却率（許容 0.008）、シンプソン 40万人（許容 0.005）
- 本文の修正:
  - `content/ch28/odds-ratio-interval.md` 例1: 対数区間の下限「0.3482」→「0.3483」（0.98083−1.96×0.32275=0.34825。指数で戻した 1.417 は不変）
- 検算スクリプト側の修正: charts の id がない図（pdf 等）での KeyError、log RR̂・log OR̂ の分散を n=200 で比べていた（c=12 などの小度数でデルタ法の近似が粗い）→ n×10 に変更、ポアソン条件付けの乱数が 40万回では標本不足（2.6SE のずれ）→ 400万回に変更
- 検算対象外: 定性的な記述（現場での使われ方、研究デザインとモデルの対応、階層原理の解釈、ブレスロー・デイ検定、因果構造による層別の判断、図の見た目とスライダー挙動）

## 第29章（欠測データ・生存時間解析）

- スクリプト: `verify/ch29.py`（実行: `.venv/bin/python verify/ch29.py`）
- 対象: em-algorithm, censored-data, truncated-data, missing-data-mechanism, mcar, mar, mnar, complete-case-analysis, single-imputation, multiple-imputation, rubins-rules, kaplan-meier-estimator, greenwood-formula, log-rank-test, cox-proportional-hazards-model, partial-likelihood, hazard-ratio（17ファイル）
- 結果: **OK 163 / NG 0**（初回 OK 161 / NG 2。2件とも検算スクリプト側、本文の誤りなし）
- 方法: 書き手の計算は使わず numpy/scipy/sympy で独立に再計算
  - EM: 打ち切り指数の反復表（λ・E[Σt]・観測対数尤度 k=0..5）、単調増加、収束値 4/27、写像の微分 1/3（sympy）、ずれの比≈1/3、観測尤度の直接最大化、E[T|T>c]=c+1/λ（数値積分）。混合正規の負担率6点・合計 2.9595・x=2.5 で 1/(1+e^−2)、M ステップ 0.4932/1.792/4.274、対数尤度 −12.051→−10.187、収束 0.495/1.892/4.186/−10.166、乱数初期値20組で尤度が減らない
  - 打ち切り: λ̂=4/27（尤度 Πf^δS^(1−δ) の数値最大化でも）、2階微分 −d/λ²（sympy）、6.75・4.679・SE 0.0741・区間 [0.0556,0.3947]、誤った扱い 0.235/0.222、区間打ち切り 0.192、f^δS^(1−δ)=h^δS
  - 切断: λ/(1−e^−λ)=1.6 の解 1.027、スコア方程式（sympy）、過大評価56%・0.358、ゼロ切断ポアソンの期待値、左側切断の指数で d/Σ(y−a)
  - 欠測メカニズム: φ(0)/Φ(0)=0.7979、MAR 452.13/547.87/471.28、MNAR 420.21/579.79/452.13、無回答者平均 547.87・δ=95.74、感度分析 452.13/477.13/502.13、MCAR の SE 7.071→7.906（1.118）、ウェルチ t=1.989・p≈0.052（scipy ttest_ind_from_stats と一致）、完全ケース 0.590/0.349/0.599
  - 単一代入: 57.571・117.619・78.413、相関 0.950→0.795、回帰 42.386+3.322x・62.317/68.960/72.282・60.656/108.801/0.964、(r−1)s²/(n−1)
  - ルービンの規則: 例1 Q̄=52.6・Ū=1.24・B=0.545・T=1.894・SE 1.376・r=0.527・ν=33.55・t=2.033・[49.80,55.40]・λ=0.345、例2 t=1.889・p=0.068・2.335、例3 対数OR の統合 0.6572/0.04007/0.00612/0.04822/0.2196・r=0.204・ν=69.9・1.994・1.929・[1.245,2.990]。多重代入 例1 T=0.52・ν=37.56・2.025・[9.040,11.960]、相対効率 0.943/0.985
  - KM・グリーンウッド: 表（n_j, d_j, 1−d/n, Ŝ）、scipy.stats.ecdf（CensoredData）の生存関数と一致、中央値9、誤った扱い 0.600/0.333、グリーンウッドの項・累積和・SE・0〜1 で切った区間（scipy の linear 区間と一致）、t=6 の途中式、log-log 区間 [0.209,0.809]（scipy の log-log と一致）、log-log 尺度の分散（sympy）、打ち切りなしで S(1−S)/n
  - ログランク: 表（n, n_A, n_B, d, d_A, E_A, V）、O_A=5・E_A=2.3528・V=1.3635・χ²=5.140・p=0.023、**scipy.stats.logrank と一致**（統計量²・p値）、E_B=5.6472・簡便法 4.219、コックスのスコア U(0)=O−E・I(0)=V
  - 部分尤度: 閉じた式=定義、log L(0)=−4.0943、U(0)=1.0167・I(0)=0.6497、sympy の微分と一致、ニュートン法 1.5648/1.4359/1.4388、HR 4.215、I(β̂)=0.6547・SE 1.236・z=1.164・p=0.244、スコア検定 1.591/0.207、尤度比 −3.3621・1.465/0.226、時間の単調変換で β̂ 不変
  - コックス・ハザード比: 1.499・z=2.70・[1.117,2.012]、0.640/0.894、1.030/1.350、S0^exp(βx)（sympy）、log(−log S) の平行性、例1 0.05/0.025/0.5・13.86/27.73・SE 0.3651・[0.244,1.023]・z=−1.898、RR 1.80・OR 2.25
- charts: 全17ファイルの front matter を yaml で読み照合。line-data は EM の反復値9点と 4/27、KM 階段（12点、同じ x の2点目を右極限）、グリーンウッドの KM・下限・上限（12点×3）、ログランクの A群・B群 KM（18点×2）、部分尤度の log L(β)（8点）、S0^HR（11点×3）、ワイブルハザード h0・2h0（11点×2）、ポアソン・0切断ポアソン（7点×2）、MNAR の感度分析（7点）、MCAR の 1/√(1−p)（7点）、完全ケースの (1−p)^k（11点×2）、相対効率（6点×3）を再計算して一致。bar-data（欠測メカニズム・MAR・単一代入・ルービン）、normmix の params=EM 収束値、expon rate=4/27、weibull(1.5,10)、missing-scatter の mechanism、各スライダーの初期値が範囲内
- 乱数（seed=20261023）: 打ち切り指数の λ̂ の一致性と SE λ̂/√d（許容 0.003・相対10%）、ゼロ切断ポアソン 40万点（許容 0.01）、2変量正規40万点で MCAR/MAR/MNAR の観測者平均・層別の重み戻し・無回答者平均（許容 0.6）、MAR(X依存) の完全ケース回帰が偏らない（許容 0.6）、完全ケース割合（許容 0.004）、適切な多重代入＋ルービンの規則の被覆率 0.95（2000回、許容 0.02）、KM の一致性（許容 0.02）、グリーンウッド SE≈実際の SD（相対5%）、H0 でのログランク棄却率（許容 0.012）、コックスの β̂≈0.7（n=3000、許容 0.1）、log HR の SD≈√(1/d1+1/d0)（相対3%）
- 本文の修正: なし
- 検算スクリプト側の修正: S0^exp(βx) の sympy 簡約で H0 を一般関数にしていた（複素数を許すため簡約されない）→ 正の記号に変更、log HR の SD を期待件数 12・20 で比べていた（小件数でデルタ法近似が約8%ずれる）→ 件数を10倍にして漸近式を確認
- 検算対象外: 定性的な記述（現場での使われ方、MAR と MNAR の識別不能性、リトルの MCAR 検定、MICE、エフロン・ブレスローの近似、ルイスの方法・SEM アルゴリズム、図の見た目とスライダー挙動・乱数図 km-curve/missing-scatter の描画）

## 第30章（モデル選択）

- スクリプト: `verify/ch30.py`（実行: `.venv/bin/python verify/ch30.py`）
- 対象: model-selection, aic, bic, mallows-cp, cross-validation, leave-one-out-cv, information-criteria-comparison, bias-variance-tradeoff, overfitting, generalization-error, expected-log-likelihood, kullback-leibler-divergence（12ファイル）
- 結果: **OK 118 / NG 0**（途中の NG は本文側2件、検算スクリプト側1件）
- 方法: 共通データ x=1..8, y=(2.2,3.3,3.8,3.7,4.9,4.8,4.3,4.4) の0〜3次多項式を numpy の最小二乗で生データから再計算
  - S_e 5.5150/1.9129/0.5736/0.5734、−2ℓ=n log(2πS_e/n)+n（sympy で σ̂²=S_e/n の代入）、AIC（k=p+1）23.727/17.256/9.621/11.618、定数を省いた形 1.024/−5.447/−13.083/−11.085、BIC 23.886/17.495/9.938/12.016
  - Cp（σ̂²=0.14335、S_e/σ̂² 38.471/13.343/4.001/4.000）32.471/9.343/2.001/4.000、例2 σ̂²=2.0・C3=4.0、E[S_e]=B+(n−p)σ²・E‖ŷ−μ‖²=pσ²+B（乱数）
  - 調整済み R² 0/0.5953/0.8544/0.8180（S_T=5.515）
  - LOOCV: 2次の係数・e_i・h_ii・e_i/(1−h_ii) の表、全次数で公式と実際の一個抜きが一致、シャーマン・モリソンの β(−i)、ŷ(−4)=4.4289、CV 0.1512・PRESS 1.2092、3次の端の h=0.8939・e_1=−0.0803・−0.757、0次で LOOCV=GCV、GCV 0.9004/0.4251/0.1835/0.2867
  - 4分割CV: 4つのフォールドの係数・予測値・誤差・平均二乗誤差、CV(4)=0.7258/0.4417/0.2507/0.4694
  - 赤池ウェイト: Δ 14.107/7.636/0/1.998、exp(−Δ/2)・合計 1.3911・重み 0.0006/0.0158/0.7189/0.2647、ΔBIC と重み 0.0007/0.0166/0.7258/0.2569、寿命データ例の AIC 242.8/239.8・e^1.5=4.48・LR 5.0
  - コイン100回中60回: ℓ0=−69.315・ℓ1=−67.301、AIC 138.629/136.602、BIC 138.629/139.208、LR 4.027、P(χ²₁>2)=0.157、厳密な周辺尤度 log B(61,41)=−69.406（数値積分とも一致）、ラプラス近似 −69.398、BF 0.913・BIC 近似 0.749、ガウス積分（sympy）
  - 真の関数 1+1.2x−0.1x²・σ²=0.16: バイアス² 0.6825/0.21/0、バリアンス σ²p/n、期待予測誤差 0.8625/0.41/0.22/…/0.32、訓練誤差の期待値 0.8225/0.33/0.10/…/0、差 2σ²p/n、汎化誤差の補正推定 0.7252/0.3108/0.1792/0.2150（n/σ̂² 倍して −n で Cp に一致）
  - 縮小推定量 cX̄: c*=μ²/(μ²+σ²/n)（sympy）、MSE(1)=1・MSE(0.5)=0.5
  - 平均対数尤度: 差 1/n（sympy）、n=10 で −1.3689/−1.4689、E_Z[(Z−a)²]=1+a²（sympy）
  - KL: サイコロ 0.4257−0.1831=0.2426・逆向き 0.2939、正規 0.5/0.3181/0.8069（数値積分とも一致）、正規どうしの公式（sympy 積分）、ベルヌーイ 0.0872/0.0823
- charts: 全12ファイルの front matter を yaml で読み照合（line-data: 次数別の AIC・BIC・Cp・Cp=p の線、罰則 log n（9点）、訓練誤差・4分割・LOOCV・GCV・補正推定、バイアス²・バリアンス・期待予測誤差（8点）、訓練・テスト誤差（8点）、平均対数尤度の2本（7点）。bar-data: AIC と赤池ウェイト。pdf の比較分布 N(0,1)）
- 乱数（seed=20261023）: AIC の偏り b≈k（N(μ,σ²)、n=200、許容 0.1）、H0 真で過大モデル選択率 AIC≈0.157・BIC≈0.009（n=10000、許容 0.01）、E[S_e]・E‖ŷ−μ‖²（相対1%）、バイアス・バリアンス分解（相対1%）、cX̄ の MSE（許容 0.01）、3次の訓練 0.08・テスト 0.24（許容 0.002）、ΣCov(ŷ_i,y_i)=pσ²（相対1%）、平均対数尤度 n=10（許容 0.003）、離散 KL の非負性
- 本文の修正:
  - `content/ch30/bic.md` 例1: 「$1.621+4\times2.0794=9.938$」→「$1.6205+4\times2.0794=9.938$」（丸めた 1.621 を使うと 9.9386 で 9.939 になる。真値 9.9383）
  - `content/ch30/information-criteria-comparison.md` 例1: 「$0.7189/0.2647=2.715$」→「$0.7189/0.2647\approx2.72$」（表示した重みの比は 2.7159、真値は 2.7154 で、等号では両立しない）
- 検算スクリプト側の修正: 1要素配列の float 変換で TypeError
- 検算対象外: 定性的な記述（現場での使われ方、一致性・漸近同等性の一般論、TIC、Δ の目安、選択後の推測、図の見た目とスライダー挙動・乱数図 bias-variance の描画）

## 第31章（ベイズ推論）

- スクリプト: `verify/ch31.py`（実行: `.venv/bin/python verify/ch31.py`）
- 対象: prior-distribution, posterior-distribution, conjugate-prior, beta-binomial-model, gamma-poisson-model, normal-normal-model, posterior-mean-map, credible-interval, predictive-distribution, noninformative-prior, jeffreys-prior, bayes-factor, marginal-likelihood, hierarchical-bayes-model（14ファイル）
- 結果: **OK 175 / NG 0**（途中の NG は検算スクリプト側2件のみ）
- 方法:
  - 共役事後分布の導出（sympy）: 「log(尤度×事前) − log(事後の核)」の θ 微分が 0 になることで、ポアソン・指数・二項（m 回×n 個）・正規精度・幾何・観測期間つきポアソン・多項ディリクレ・指数型分布族の一般形（ν1+ΣT, ν0+n）を確認。正規・正規の平方完成、Σ(x−μ)² の分解、ベータ・ガンマの MAP（微分=0 を解く）、E[(θ−d)²]=V+(m−d)²
  - 事後要約（scipy）: Ga(7,40) の平均・最頻値・SD・等裾 [0.0704, 0.3265]、Be(2,8)・Be(2,18) の分散・確率・中央95%、離散事後分布の表、Be(8,4)・Be(16,8)・Be(2,12)・Be(1,21)・Be(3.5,7.5)・Be(4,8)・Be(0.5,10.5)・Ga(23,6)・Ga(20,5)・Ga(20.5,5) の平均・中央値・MAP・等裾区間・裾確率、N(58,20)・N(20.75,0.5) の区間と確率
  - HPD 区間は2通りで独立に算出: (1) 下側確率を動かし幅 F⁻¹(p+0.95)−F⁻¹(p) を minimize_scalar で最小化、(2) 「pdf(L)=pdf(U) かつ被覆 0.95」を brentq で解く。Be(16,8) [0.4811, 0.8446]（幅 0.3635、両端密度 0.681）、Be(2,12) [0.0049, 0.3188]（幅 0.3139、密度 0.729、等裾より 7.98% 短い）、区間内の密度 ≥ 区間外、正規では等裾と一致
  - 損失: 事後平均・MAP・MLE の二乗損失 0.008889/0.009118/0.0100（数値積分とも）、絶対損失の最小点＝中央値（数値最適化）
  - 予測分布: ベータ二項の表（scipy betabinom・ベータ関数の式・二項を Be(16,8) で数値積分の3通り）、平均・分散と公式、全分散の公式、Bin(5,0.7) の表と P(Y≥4)、負の二項の積分（sympy の積分を数値代入で照合）、表・P(Y=0)・P(Y≥6)・分散 4.4722、ポアソンとの比較、N(58,245) の予測区間と P>70、B(a'+1,b')/B(a',b')=a'/(a'+b')（sympy）
  - ジェフリーズ: ベルヌーイ（二階微分とスコア二乗の両方）・ポアソン・正規平均・正規 σ（2/σ²）・指数のフィッシャー情報量（sympy）、logit での変換不変性 I_φ=I_θ(dθ/dφ)²、Be(1/2,1/2) の正則性
  - 無情報: logit(U) の密度＝標準ロジスティック（sympy）、Be(0,0)・x=0 の事後が発散（sympy 極限）、1−0.05^{1/21}
  - ベイズファクター・周辺尤度: m0=0.03696、m1=1/21（ベータ関数・数値積分、全 x で一様）、B10=1.2882・0.5630・p=0.1153、x=17 で 43.80・p=0.0026、正規 B10=1.3418（μ の数値積分とも）・p=0.0455、5つのベータ事前の周辺尤度（閉じた形・数値積分・betabinom）と比 1.6454、恒等式 0.19164/4.02442、ポアソン周辺尤度 4.336/2.860/0.866×10⁻⁵（閉じた形と数値積分）、BIC 6.3000・6.0890、リンドレー（τ² を広げると B10→0）
  - 階層: クラス A・B・C の縮小係数・推定値・事後 SD、ベータ階層の 0.1667/0.1417
- charts: 全14ファイルの front matter を yaml で読み照合。prior-posterior（beta-binomial a,b,n,x／gamma-poisson shape,rate,n,sum／normal-normal mu0,tau0,sigma,n,xbar）が各本文の例と同じモデル・値（Ga(23,6)、Be(16,8)、Be(8,4)、Be(2,12)、Be(1,21)、N(58,20)、クラスA の 66.15）、line-data の log10 BF10（x=10..18、9点）と caption の x=15（p=0.041、BF≈3.2）、bar-data の周辺尤度5つ、pdf/pmf（beta(2,8)・beta(0.5,0.5)・比較 Be(1,1)・logistic(0,1)・nbinom r=23,p=0.857≈6/7）、各スライダーの初期値が範囲内
- 乱数（seed=20261023）: Ga(23,6) 混合ポアソン 40万点の度数≈負の二項（許容 0.003）、予測分布の分散≈245（相対1%）、等裾区間の logit 変換不変性（40万点、許容 0.01）、logit(U) の KS 統計量 < 0.003、階層モデルの周辺分散 τ²+σ_j²（相対1%）、縮小推定の全体 MSE が生の平均より小さく理論値に一致（相対1%）
- 本文の修正: なし
- 検算スクリプト側の修正: HPD の連立解法で brentq の右端が F(L)+0.95>1 の範囲に入り NaN → 右端を F⁻¹(0.05) で制限、posterior-mean-map 例1 の「0.008889+(0.6818−0.6667)²」を丸めた値で計算すると 0.009117（真値 0.0091185）なので、代入式の照合だけ許容を丸め誤差 2e-6 に広げた（本文の 0.009118 は真値と一致）
- 検算対象外: 定性的な記述（現場での使われ方、ジェフリーズの目安表、BIC 近似 2log B10≈BIC0−BIC1 の一般論、MCMC・経験ベイズの説明、図の見た目とスライダー挙動）

## 第7〜12章（重要度C）

- スクリプト: `verify/ch07.py`〜`verify/ch12.py` の末尾に「重要度Cの用語」の節を追加（既存チェックはそのまま。乱数は節ごとに seed=20261023+章番号）
- 対象: convergence-in-mean-square, lindeberg-levy-clt, berry-esseen-theorem（ch07）、complete-statistic, observed-fisher-information（ch08）、clopper-pearson-interval, likelihood-based-confidence-interval（ch09）、randomized-test（ch10）、bartlett-test, preliminary-test-problem, normality-test, one-way-test-variance-equality-levene（ch11）、variance-stabilizing-transform, dispersion-test（ch12）（14ファイル）
- 結果: 追加分 **OK 162 / NG 0**（ch07 31、ch08 23、ch09 31、ch10 9、ch11 36、ch12 32）。各ファイル全体も NG 0（ch07 125、ch08 213、ch09 361、ch10 175、ch11 252、ch12 243）。途中の NG は本文側1件、検算スクリプト側1件
- 方法:
  - 平均二乗収束: σ²=35/12、MSE 0.0292、チェビシェフ上限 0.1167、MSE＝分散＋偏り²（sympy）、反例 E[X_n²]=n、乱数で MSE と P(|X̄−3.5|≥0.5)≤上限
  - リンデバーグ・レヴィ: アーウィン・ホール分布の正確な裾（分数で厳密計算）0.1607/0.0223/0.0056 と正規近似 0.1587/0.0228/0.0062、{1−t²/(2n)}ⁿ→e^{−t²/2}（sympy）
  - ベリー・エッセン: ベルヌーイの ρ（sympy）、ρ/σ³=1・2.733、上限 0.0475/0.1298/0.0649、実際の最大誤差（各跳びの左右極限で評価）0.0398/0.0832/0.0420
  - 完備統計量: E[T(n−T)]=n(n−1)p(1−p)（sympy, n=3,7,10）、0.2333・0.21・0.225、E[g(T)] の多項式表示、X1−X2 の反例、V[τ̂]>CR下限
  - 観測情報量: −ℓ'' の式（sympy）、コーシーの θ̂=1.109（格子で全域最大も確認）・J=3.184・SE 0.560、I=1/2（数値積分）・0.632、E[J]=nI（乱数）、ポアソン J(λ̂)=n/x̄
  - クロッパー・ピアソン: 0.0123/0.3170（ベータ分位点・二項の裾の方程式・scipy binomtest exact が一致）、ウィルソン 0.0279/0.3010、被覆確率 0.989（正確計算と乱数）、全 p で ≥0.95、0.0711/0.0582
  - 尤度に基づく区間: −6.5017・−8.4224・[0.0174,0.2779]、ワルド [−0.0315,0.2315]、1.921・0.147・3.317、logit 変換で端点が写ること、n=200 の被覆率≈0.95（乱数）
  - 無作為化検定: 0.0207/0.0577/0.0370・γ=0.7928・サイズ 0.05・検出力 0.4164/0.1916/0.5683、乱数で棄却率 0.05
  - バートレット・ルビーン: 本文は要約値のみのため、すべての要約値（s² 0.0217/0.1017/0.6747、z̄ 0.1167/0.2500/0.5556、外れ値 12.0 の偏差 1.6667）を満たす 0.1 刻みのデータを総当たりで構成し、s_p²=0.2660・分子 12.694・C=1.0889・B=11.658・p=0.0029（scipy bartlett と一致）、W=2.710・F_0.05(2,15)=3.682・p=0.099（scipy levene と一致）、ブラウン・フォーサイス p=0.567 を確認（条件を満たす192組すべてで W=2.710）
  - 予備検定: 20万回の模擬実験でプール t 0.107・ウェルチ 0.051・二段階 0.081・F 非棄却 64%（許容 ±0.003、割合は ±0.01）
  - 正規性の検定: 0.8425・JB=14.042・p=0.0009、JB の式＝scipy jarque_bera、W≤1、歪度・尖度の分散 6/n・24/n（乱数）
  - 分散安定化変換: 変換の微分（sympy, 逆正弦・平方根・tanh⁻¹）、0.0018/0.0042/0.0050、変換後の分散を二項分布で正確計算＋20万回の乱数、Z=2.140・p=0.032・通常の z=2.137
  - 分散検定: x̄=5・176・s²=16・3.2・D=35.2・19.675・p=0.0002、ピアソン χ² と同形、H0 下の棄却率 0.05・E[D]=11（乱数）
- 本文の修正:
  - `content/ch12/variance-stabilizing-transform.md` 具体例: $p=0.3$ の $\arcsin\sqrt{\hat p}$ の分散「0.0052」→「0.0051」（二項分布での正確な値は 0.005139、20万回の乱数でも 0.00510〜0.00515）
- 検算スクリプト側の修正: 丸めた s² から B を再計算するチェック（11.649 で表示桁の差。本文の主張ではない）を削除
- 補足（修正せず）: 平均二乗収束の「0.0292/0.5²≈0.1167」、無作為化検定の「(0.05−0.0207)/0.0370≈0.7928」は、丸めた途中値で割ると 0.1168・0.7919 になるが、≈ で正確な値を示しているため本文は変えていない
- 検算対象外: 定性的な記述（現場での使われ方、レーマン・シェフェの定理や指数型分布族の完備性の一般論、C≤0.4748 の定数の出典、リリフォース検定の棄却限界、リンデバーグ・フェラー／リャプノフの条件、正規Q-Qプロットの図）

## 第13〜16章（重要度C）

- スクリプト: `verify/ch13.py`〜`verify/ch16.py` の末尾「重要度Cの用語」の節（既存チェックはそのまま。各ファイル実行で全体 NG 0: ch13 OK 233、ch14 OK 191、ch15 OK 182、ch16 OK 236）
- 対象: ch13 asymptotic-relative-efficiency, median-test, kolmogorov-smirnov-two-sample, runs-test, jonckheere-terpstra-test／ch14 positive-recurrence, continuous-time-markov-chain／ch15 counting-process, pure-birth-process, compound-poisson-process, mmc-queue, reflection-principle／ch16 partial-f-test（13ファイル）
- 結果: **OK 78 / NG 0**（ch13 33、ch14 12、ch15 25、ch16 8）
- 方法:
  - ARE: 正規（σ 記号のまま）とラプラス（尺度 b）で 12σ²(∫f²)²・4σ²f(m)² を sympy で積分（3/π, 2/π, 1.5, 2）、必要人数 104.7・157.1、中央値検定の効率 2/3、sim で t 検定 n=100 と順位和 n=105 の検出力がほぼ等しい
  - 中央値検定: M=13.5・表、χ²=7.2（p 0.0073）、イェーツ 5.0（0.0253）、フィッシャー 0.0230（scipy と超幾何の直接計算）、scipy median_test
  - 2標本KS: 合併16個の並び 12870 通りを全数え上げして P(D≥0.75)=0.0186・P(D≥0.625)=0.0870（ks_2samp exact とも一致）、コルモゴロフ分布の級数で λ=1.5 の漸近 p=0.0222（kstwobign とも）、1.358 は 0.95 分位、帰無分布が母集団分布によらないこと（正規・指数の sim）
  - ラン検定: R=5、E[R]=10.1・V[R]=3.879、z と p（連続修正あり・なし）、C(20,7)=77520 通りの全数え上げで P(R≤5)=0.0095、平均・分散・P(R=2k) の式が全数え上げと一致
  - JT 検定: U=14,16,13、J=43、E・V・z・p、34650 通りの全数え上げで正確な p=0.0020 と E[J]・V[J] の式、クラスカル・ウォリス p=0.0296
  - 正再帰: 定常分布 (1/3)(2/3)^n（打ち切り行列の固有ベクトルとも）、sim で平均再帰時間≈3
  - 連続時間マルコフ連鎖: π、p00(1)=0.8835（式・expm）、sympy で P'=PQ=QP、P(s+t)=P(s)P(t)、sim
  - 計数過程: N(t) の値、Po(2.4) の P=0.2090、{N(t)≥n}={S_n≤t}（ポアソン上側=ガンマ cdf）
  - 純出生: ユール過程の幾何分布が前進方程式を満たす（sympy）・打ち切り生成行列の expm と一致、0.3679/0.3996/3.667、sim 平均 2.72・P≥3 0.400
  - 複合ポアソン: MGF を sympy で微分して E・V、720/18000/134.16/z=1.342/0.0899、ガンマ（形4・尺度5）の sim で P(S>900)≈0.094
  - M/M/c: π0=1/7、C、Lq、Wq、W、L、リトルの公式、打ち切り生成行列の定常分布、M/M/1 比較 22.5・30・45 分、c=1 での還元（sympy）
  - 鏡像原理: 2^10 通りの数え上げで 232・45 通り、n=1..12 の全 m,b で道の対応と P(max≥m) の式、BM 0.3173
  - 部分 F 検定: 共通データ（8店舗）から S_e 4.7738/13.4167、F=9.052、F0.05(1,5)=6.608、p=0.0298、√F=3.009＝広告費の t 値、6.608=2.571²、R² 版でも同値、H0 下の sim 棄却率≈0.05
- 乱数（seed=20261023）: 許容誤差は各チェックに明記（概ね標準誤差の4倍以上）
- 本文の修正:
  - `content/ch13/asymptotic-relative-efficiency.md` 具体例: 「$100/0.637=157.1$」→「$100/0.6366=157.1$」（100/0.637=156.99 で 157.1 にならない。157.1 は 100/(2/π)=157.08）
- 検算対象外: 定性的な記述（現場での使われ方、零再帰・過渡の分類の一般論、爆発の条件、図の見た目）

## 第32章（シミュレーション）

- スクリプト: `verify/ch32.py`（実行: `.venv/bin/python verify/ch32.py`）
- 対象: uniform-random-number, linear-congruential-generator, inverse-transform-sampling, rejection-sampling, box-muller-method, monte-carlo-integration, importance-sampling, mcmc, metropolis-hastings, gibbs-sampling, burn-in-convergence, bootstrap, bootstrap-confidence-interval, jackknife（14ファイル）
- 結果: **OK 159 / NG 0**（途中の NG は検算スクリプト側1件のみ）
- 方法:
  - 線形合同法: 例1〜例2の数列・周期（16・8・4・12）を実行して照合、ハル・ドーベルの3条件と最大周期の一致を m≤64 の全 (a,c) で総当たり、偶奇の交互
  - 一様乱数: E[U]=1/2・V=1/12・U(a,b)（sympy）、サイコロ例、12個の和の E[(U−1/2)^4]=1/80・尖度1.8・E[Z^4]=2.9（キュムラントで厳密に）
  - 逆関数法: 指数・ワイブル・コーシーの F(F⁻¹(U))=U（sympy）、幾何の生成式の度数=pmf（等間隔格子）、例1〜3 の数値
  - 棄却法: 採択率 1/M（数値積分）、Be(2,2)・Be(2,5) の M と最大点（sympy）、半正規 M=√(2e/π)=1.3155 と採択条件 e^{−(y−1)²/2}（sympy）、試行回数の平均 M・分散 M(M−1)
  - ボックス・ミュラー: 逆変換の偏微分4つ・ヤコビアン −u1/(2π)・|J|=φ(z1)φ(z2)・順変換のヤコビアン −2π/U1（sympy）、P(R²>s)=e^{−s/2}、例1・例2 の数値、極座標法（採択率 π/4、W~U(0,1)、出力の正規性）
  - モンテカルロ積分・重点サンプリング: σ²=0.2420（数値積分とも）、n=242036・26968、5.6889（sympy）、P(Z>3) の重み e^{−3x+4.5}（sympy）、2次モーメント e^9P(Z>6)=7.994e-6（数値積分とも）、分散比 約1/218、g=(2/3)(1+x) の分散 0.0269、最適 g* で分散0
  - MCMC・MH・ギブス: 2状態連鎖の定常分布・ρ_k=0.7^k（P^k から厳密に）・5.667・1765、MH 3状態の推移行列・詳細つり合い・πP=π・状態1からの分布・固有値 −1/3、例2・例3 の採択確率、2変量正規の条件付き分布と AR(1) 形（sympy）、ギブスの採択比=1、手計算 1.9/0.92/0.916/1.4528、ESS 6000/1050/約100
  - バーンイン: 1+2Σφ^k=(1+φ)/(1−φ)（sympy）、43.7・0.097・526.3、R̂ の W・B・V̂ 2例
  - ブートストラップ: 待ち時間データの平均・中央値・s・SE、C(19,10)=92378、0.9^10、ベーシック・正規近似・t 区間
  - ジャックナイフ: 例1・例2 の全 θ̂_(i)・偏り・補正・V_jack=70.094、擬似値の恒等式、「σ̂² の補正=s²」「x̄ の V_jack=s²/n」を任意データ（記号5個）で sympy 証明
- 本文の「シード k」の値: np.random.default_rng(k) で同じ手順を再実行し、逆関数法(1)・棄却法(2)・ボックス・ミュラー(3)・モンテカルロ積分(4,5)・重点サンプリング(6)・MCMC の表(7、幅ごとに乱数を作り直す)・ギブス(8)・ブートストラップ(9、1つ目の標本・SE・偏り・分位点まで) をすべて表示桁で再現。MCMC 表の ESS（40/1295/2422/750）は再計算値 40.37/1295.55/2422.89/750.61 の切り捨て表示（「約」なので許容1）
- charts: 全14ファイルの front matter を yaml で読み照合（points-2d の16点＝例1の (x_n,x_{n+1}) と傾き5の平行線、bar-data の値＝σ̂²_(i)、lln-path p=0.785≈π/4、pdf N(0,1)/N(3,1)、clt-sim n=12、rejection a=2,b=5、gibbs rho=0.8・init=2・seed=8、mcmc-trace init=−6、acf-plot φ=0.9、bootstrap-hist の n・B・stat、scatter seed=3）
- 乱数（seed=20261023）: Be(2,5) の棄却法（採択率 許容0.005、KS<0.01）、極座標法（許容0.003、KS<0.01）、重点サンプリングの不偏性（3SE）と分散（相対5%）、2状態連鎖の割合（3SE）、独立連鎖 MH で Ga(2,1)（相対3%）、AR(1) の V[x̄]=σ²/n_eff（相対5%）
- 本文の修正:
  - `content/ch32/uniform-random-number.md` 例2: 「分散 $0.997$、4次モーメント $2.883$」→「分散 $0.996$、4次モーメント $2.865$」（ほかの章内のシード例はすべて default_rng で再現できたが、この値だけ再現できなかった。default_rng(1) の再現値に合わせた。どちらも理論値 1・2.9 のモンテカルロ誤差 3SE 以内）
- 検算スクリプト側の修正: ワイブルの sympy 検算で −log(1−U)>0 の仮定がなく簡約できなかった → v=−log(1−U) を正の記号に置換
- 検算対象外: 定性的な記述（現場での使われ方、エルゴード定理・収束条件の一般論、BCa 法・ブートストラップ t 法の精度の一般論、ジャックナイフが中央値で一致性をもたないこと、図の見た目とスライダー挙動）

## 第17〜21章（重要度C）

- スクリプト: `verify/ch17.py`〜`verify/ch21.py` の末尾に「重要度Cの用語」の節を追加（既存チェックはそのまま。乱数は節ごとに seed=20261023+章番号。statsmodels は不使用）
- 対象: dfbetas, breusch-pagan-test, robust-standard-error（ch17）、tobit-model, offset-term（ch18）、elastic-net, principal-component-regression, spline-regression（ch19）、studentized-range-distribution, effect-size-eta-squared（ch20）、sampling-weight（ch21）（11ファイル）
- 結果: 追加分 **OK 104 / NG 0**（ch17 26、ch18 18、ch19 29、ch20 22、ch21 9）。各ファイル全体も NG 0（ch17 200、ch18 217、ch19 310、ch20 220、ch21 210）。途中の NG は検算スクリプト側1件のみ
- 方法:
  - DFBETAS: 1点除外の当てはめ直しで 2.0833+1.5833x、c00=0.4226・c11=0.0119、店8除外 1.0000+1.8929x・s₍₈₎=0.5278、DFBETAS₁,₈=−5.3748・DFBETAS₀,₈=3.1573、目安 0.7071 を超えるのは店8だけ、他の最大は店1の 0.4353、係数変化の公式を全8点で照合、chart の8値
  - ブルーシュ・ペイガン: 補助回帰 −2.6321+1.1898x・R²=0.6269・nR²=5.0150・p=0.0251（不均一分散の項と同値）、もとの版 ESS/2=4.0123・p=0.0452、乱数（n=200、2万回）で H0 下の棄却率≈0.05
  - 頑健標準誤差: サンドイッチ行列と単回帰の式（145.5755/42²=0.08253）が一致、SE 0.2940/0.2873/0.3317/0.4469・t 値 6.38/6.53/5.66/4.20・t₀.₀₂₅(6)=2.4469、乱数で HC0 の平均≈真の分散
  - トービット: Φ(0.5)・φ(0.5)・0.3085・1.3956（数値積分・100万点の乱数とも）・2.0183・0.3457、∂E[y]/∂(xᵀβ)=Φ(z)（sympy）、本文の尤度の最尤推定でパラメータ復元、正の値だけの OLS が0方向に偏ること（乱数）
  - オフセット項: ポアソン回帰をニュートン法で独立実装（scipy.optimize とも照合）、0.072・−2.6311・当てはめ値・逸脱度 3.1598、γ̂=0.6553（SE 0.1910）・0.0972、差 3.0626・p=0.0801
  - エラスティックネット: (0.3958, 0.2958) を連立方程式・座標降下法・格子探索で確認、Lasso λ=10 の (0.7368, 0.1368)・比 5.4、EN の比 1.34、直交計画の公式を4組の (λ1, λ2) で数値最小化と照合、chart method=lasso
  - 主成分回帰: 固有値 19・1、γ̂1=1.3621、(0.9632, 0.9632)、m=p で OLS、分散 0.5263σ²/0.0263σ²、RSS 0.0284→0.2084、リッジの λ_k/(λ_k+λ) 縮小、乱数で分散と偏り
  - スプライン回帰: 切断べき基底の OLS で 0.9636+1.0318x−0.6655(x−4)₊、RSS 1.2214→0.0616、F=94.08・p=0.0002、2次式 0.2155、3次基底の連続性（sympy）、chart の当てはめ値8点
  - スチューデント化範囲: scipy.stats.studentized_range で 3.199/3.948/4.415/4.755・3.578、√2·t₀.₀₂₅(9)、ν→∞ の 3.314 を範囲の分布の積分式から求根、P(Q>5.892)=0.0062、乱数で W/S の定義どおりの上側5%点、chart params（k=3, df=9, alpha=0.05）とスライダー範囲
  - 効果量: 肥料データから S_A・S_E・S_T、η²=0.680・ρ_A=0.609・ω²=0.588、ダミー回帰の R²=η²、乱数で純変動の不偏性と η² の過大傾向、偏 η² の合計が1を超える例
  - 標本ウェイト: ウェイト・合計 750・41.667・46.111・計画効果 1.053・有効標本 42.7・20.833、乱数で HT 推定量の不偏性とキッシュの計画効果（有限母集団修正込みの正確値 1.057）
- 本文の修正: なし
- 検算スクリプト側の修正: η² の「2×9.571/(2×9.571+9)=19.142/28.142」を丸め前の F で照合して NG（2F=19.1429）→ 本文は丸めた F=9.571 で計算しているため、その値で照合
- 補足（修正せず）: スプラインの「1.0318−0.6655=0.3664」は丸めた値どうしの差では 0.3663 だが、丸め前の傾き 0.36636 を正しく示しているため本文は変えていない
- 検算対象外: 定性的な記述（現場での使われ方、平滑化スプラインの解が自然3次スプラインになる定理、Lasso が p>n で高々 n 個しか選べない性質、ケンカー版の漸近性の一般論、図の見た目とスライダー挙動）

## 第22〜26章（重要度C）

- スクリプト: `verify/ch22.py`〜`verify/ch26.py` の末尾に「重要度Cの用語」の節を追加（既存チェックはそのまま。乱数は各ファイル既存の rng（seed=20261023）を継続使用。sklearn は不使用）
- 対象: biplot, low-rank-approximation（ch22）、kernel-method, decision-tree, naive-bayes-classifier（ch23）、mixture-model-clustering, standardization-before-clustering（ch24）、graphical-lasso（ch25）、nonmetric-mds, multiple-correspondence-analysis, goodness-of-fit-indices（ch26）（11ファイル）
- 結果: 追加分 **OK 93 / NG 0**（ch22 23、ch23 17、ch24 14、ch25 15、ch26 24）。各ファイル全体も NG 0（ch22 181、ch23 181、ch24 191、ch25 163、ch26 192）
- 方法:
  - バイプロット: 4教科データ（主成分分析の項目と同じ X）を標準化して SVD、α=0 の G=√(n−1)U₂・H=V₂D₂/√(n−1) が chart の12点と小数第4位で一致、H=相関行列の負荷量、どの α でも GH'=X₂、生徒3の数学 0.560→79.27 点・国語 −0.892→62.42 点、累積寄与率 0.9697、国語・英語の内積 0.951 と相関 0.902
  - 低ランク近似: A=[[4,0],[3,−5]] の特異値 √40・√10、A₁=[[2,−2],[4,−4]]、誤差10=d₂²・80%、スペクトルノルム誤差=d₂、乱数の階数1行列2000個より誤差が小さいこと、4教科の d²=22.751/4.401/0.708/0.140・誤差 5.249/0.848、元の2列で回帰する方が誤差大
  - カーネル法: (x'y)²=25=φ(x)'φ(y)（sympy で恒等式）、e^{−2.5}=0.0821・e^{−0.625}=0.5353、乱数でグラム行列の半正定値性、−‖x−y‖² は非半正定値、1次元の例が x² で分離可能
  - 決定木: ジニ 0.32/0.18、20/49・0.286・0.214、エントロピー 0.722・0.863、情報利得 0.278/0.396
  - ナイーブベイズ: 0.9926・0.9709、対数和でも同値、平滑化 0.03125、正規分布版=対角共分散の QDA（乱数500点）
  - 混合モデル: 負担率 0.910/0.426/0.030 と各密度値、chart params、境目の σ₂ 依存（brentq、0.1刻み）、EM の自作実装で真値回復（n=4000）、σ→0 で k-means の極限
  - 標準化: 生の距離 100.02/28.28、標準化後 1.020/2.010、標準化ユークリッド距離との一致、単位変更への不変性、平均を引くだけでは不変
  - グラフィカルLasso: ADMM を自作し λ=0.2/0.4/0.5/0.6 の解、Ŵ の値（0.4,0.3,0.2／0.2,0.1,0.02）、偏相関（0.364,0.245,0.092／0.199,0.098／0.1）、KKT 条件、近傍500点で目的関数が最大、境目 λ=(0.1+√0.41)/2≈0.370（sympy と ADMM の前後±0.005）、標本偏相関 0.504/0.355/0.144（共分散選択と同じ S）、chart の辺とラベル
  - 非計量MDS: PAVA を自作し距離・ディスパリティ・Stress=0.024、B=0.85 で Stress=0、拡大縮小不変
  - 多重コレスポンデンス分析: 170×6 の指示行列から固有値 0.7196/0.5316/0.4684/0.2804、合計 (J−Q)/Q=2、(1±γ)/2（γ はコレスポンデンス分析の項目と同じ 0.4393/0.0632）、ベンゼクリ補正 0.1929/0.0040=γ²・約98%、バート行列の固有値=λ²、乱数の Q=3 でも全イナーシャ一定、第1軸の符号
  - 適合度指標: SEM の項目と同じ相関行列で1因子モデルを最尤推定（λ̂・ψ̂ が SEM の項目と一致）、χ²=2.845・χ₀²=187.105・p=0.241、RMSEA 0.046・CFI 0.995・TLI 0.986・GFI 0.99269・AGFI 0.963・SRMR 0.023
- 本文の修正:
  - `content/ch25/graphical-lasso.md`: KKT 条件の符号 $\hat w_{jk}=s_{jk}-\lambda\,\mathrm{sign}(\hat\omega_{jk})$ → $s_{jk}+\lambda\,\mathrm{sign}(\hat\omega_{jk})$（正の相関で ω̂<0 となり ŵ=s−λ になる。本文の数値例はもともと正しい）
  - `content/ch22/biplot.md`: h_数学 の第2成分 −0.398 → 実際は −0.39750 で、内積も 0.5605（「≈0.561」は丸め値どうしの計算）。g₃・h_数学 を chart と同じ4桁（−0.1860, −1.8363）・（0.9114, −0.3975）にし、内積と復元式を 0.560 に修正（79.27 点は不変）
  - `content/ch24/mixture-model-clustering.md`: 「σ₂ を大きくすると境目が左へ動く」は σ₂≈2.2 までで、それ以上ではわずかに右へ戻る（スライダー上限3で 1.874）→ caption と本文に「σ₂≈2.2 まで」を補足
  - `content/ch26/goodness-of-fit-indices.md`: AGFI の式 $1-\frac{20}{4}(1-0.993)$ は 0.965 になるため、丸め前の GFI $0.99269$ を使う形に修正（結果 0.963 は不変）
- 検算スクリプト側の修正: ch23 の節に r() が未定義、ch24 の allclose がスカラーを展開しない → それぞれ節内で対応
- 検算対象外: 定性的な記述（現場での使われ方、マーサーの定理の一般論、剪定・過学習の一般論、ストレスの目安値、適合度指標の目安値の妥当性、図の見た目とスライダー挙動）
