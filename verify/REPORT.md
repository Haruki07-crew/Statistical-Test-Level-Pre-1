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
