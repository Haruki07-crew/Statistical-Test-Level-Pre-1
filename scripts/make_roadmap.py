"""学習ロードマップ（content/roadmap.yaml）を生成する。

使い方:
    .venv/bin/python scripts/make_roadmap.py          # 生成して検査
    .venv/bin/python scripts/check_roadmap.py         # 検査だけ

方針（詳しくは spec/research/roadmap.md）:
- 各用語の段階（stage 1〜4）・前提用語・学習時間・理由は、下の TABLE に手で決めて書く。
- score は TABLE の値から機械的に計算する。
    raw = 重要度の基礎点 + 章の頻出度 + 依存の多さ（前提として参照される回数・related で参照される回数）+ 個別補正
  これを段階ごとの帯（例: 段階1は 78〜99）に線形に割り当てる。
- 並び順は段階ごとのトポロジカル順。前提が先に来る範囲で、score の高いもの
  （その用語を前提にする同じ段階の用語の score も考慮）を先に出す。
- see（統合先）がある用語は、統合先を自動で前提に入れる。
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
OUT = CONTENT / "roadmap.yaml"

STAGES = [
    (1, "まず固める基礎", "確率・期待値と分散・主要な分布・推定と検定の基本手順。ほかのすべての章がこの上に乗る。"),
    (2, "得点源", "回帰・分散分析・時系列・主成分分析・ベイズ・マルコフ連鎖など毎回のように出る分野と、推定・検定の理論。"),
    (3, "差がつく応用", "多変量解析の発展、標本調査、欠測と生存時間、確率過程の応用など、押さえると周りと差がつく項目。"),
    (4, "余裕があれば", "出題がまれな発展事項と、ほかの項目に統合済みの用語。直前に眺める程度でよい。"),
]

# 段階ごとの score の帯
BANDS = {1: (78, 99), 2: (60, 86), 3: (36, 64), 4: (8, 38)}
IMP_BASE = {"A": 70, "B": 55, "C": 40}
# 章の頻出度（調査結果から。15=ほぼ毎回、10=よく出る、5=出題が少なめ・後回しでよいと言われる分野）
CH_WEIGHT = defaultdict(lambda: 10)
CH_WEIGHT.update({n: 15 for n in (5, 6, 8, 9, 10, 11, 12, 16, 22, 27, 31)})
CH_WEIGHT.update({n: 5 for n in (4, 7, 13, 15, 19, 21, 25, 26, 29)})
CH_WEIGHT[33] = 8

# id | stage | minutes | prereqs（空白区切り） | 個別補正 | 理由（60字以内、$ と \ は使わない）
TABLE = """
conditional-probability|1|30||5|確率計算すべての出発点。ベイズの定理や条件付き分布の前提
law-of-total-probability|1|20|conditional-probability|0|場合分けで確率を求める定番の道具。ベイズの定理の分母になる
independence-of-events|1|20|conditional-probability|0|独立の定義は全分野で使う。確率計算の頻出論点
bayes-theorem|1|30|law-of-total-probability|5|検査の陽性的中率などの計算が頻出。ベイズ法の土台
conditional-independence|2|20|independence-of-events|0|グラフィカルモデルや因果推論の前提。独立との違いが問われる
inclusion-exclusion-principle|3|20|independence-of-events|-5|和事象の確率計算で使う。出題は限られるが短時間で身につく
probability-mass-and-density-function|1|30|conditional-probability|5|分布を扱うすべての計算の出発点
cumulative-distribution-function|1|20|probability-mass-and-density-function|0|分位点・順序統計量・変数変換で毎回使う
survival-function|2|15|cumulative-distribution-function|-3|生存時間解析とハザード関数の前提
hazard-function|2|25|survival-function|0|指数分布やワイブル分布と結び付けて問われる。生存時間解析の基礎
joint-probability-function|1|20|probability-mass-and-density-function|0|2変数の離散分布の基本。周辺・条件付き分布の前提
marginal-distribution|1|20|joint-probability-function|0|同時分布から一方を取り出す操作。多変量の計算で必須
marginal-probability-function|1|10|marginal-distribution|-8|周辺分布の離散版。和をとるだけなので短時間で済む
conditional-probability-function|1|20|marginal-distribution conditional-probability|0|条件付き期待値やベイズ法の計算の前提
joint-probability-density-function|1|25|joint-probability-function|0|連続2変数の計算の基本。積分範囲の扱いが問われる
marginal-probability-density-function|1|15|joint-probability-density-function marginal-distribution|-8|同時密度を積分して得る。2変量正規分布などで使う
conditional-probability-density-function|1|25|marginal-probability-density-function|0|条件付き期待値や2変量正規の条件付き分布の前提
moment-generating-function|1|40|moments|5|期待値・分散の導出や再生性の証明で頻出
probability-generating-function|1|30|moments|0|離散分布の期待値・分散を素早く出す道具。計算問題で出る
characteristic-function|4|20|moment-generating-function|-5|母関数の理論的な拡張。計算問題での出番は少ない
moments|1|25|probability-mass-and-density-function|3|期待値・分散・歪度をまとめる考え方。母関数の前提
expectation-variance-properties|1|40|moments|10|線形性や分散の公式は全分野の計算で使う最重要事項
quantile-function|2|15|cumulative-distribution-function|-3|分位点の定義。逆関数法や分位点回帰の前提
coefficient-of-variation|4|10|expectation-variance-properties|-5|単位の違うばらつきの比較。定義を知っていれば解ける
covariance-and-correlation|1|30|expectation-variance-properties|8|回帰・多変量解析の土台。和の分散の計算で頻出
spurious-correlation|3|15|covariance-and-correlation|-3|相関と因果の区別を問う概念問題。偏相関と組で押さえる
partial-correlation-coefficient|2|30|covariance-and-correlation|0|第3の変数の影響を除いた相関。公式による計算が出る
conditional-expectation|1|35|conditional-probability-density-function expectation-variance-properties|5|繰り返し期待値の公式は計算問題の定番
conditional-variance|1|30|conditional-expectation|0|全分散の公式は混合分布や階層モデルで頻出
skewness-kurtosis|2|20|moments|-3|分布の形を表す指標。定義と正規分布での値が問われる
weighted-geometric-harmonic-mean|4|15|expectation-variance-properties|-5|平均の種類の使い分け。成長率や速度の平均で出る
mean-vector|1|10|expectation-variance-properties|-8|多変量を扱うための記法。分散共分散行列と組で覚える
variance-covariance-matrix|1|30|covariance-and-correlation mean-vector|3|多変量正規・回帰・主成分分析のすべてで使う
correlation-matrix|2|15|variance-covariance-matrix|-3|標準化した分散共分散行列。主成分分析の入力になる
change-of-variables|1|40|probability-mass-and-density-function|5|密度の変換とヤコビアンは導出問題の定番
log-transformation|3|15|change-of-variables|-3|右に裾の長いデータの処理。対数正規分布と結び付く
power-transformation|4|10|change-of-variables|0|Box-Cox変換の特別な場合として押さえれば足りる
box-cox-transformation|2|20|change-of-variables|0|新しい出題範囲表に明記。変換パラメータの意味が問われる
logit-logistic-transformation|2|20|change-of-variables|0|ロジスティック回帰とオッズの前提になる変換
probit-transformation|3|15|logit-logistic-transformation standard-normal-distribution|-3|プロビットモデルの前提。ロジットとの違いを押さえる
sum-of-random-variables|1|30|moment-generating-function|5|再生性は正規・ポアソン・ガンマなどの計算で頻出
bernoulli-distribution|1|15|expectation-variance-properties|0|二項分布など離散分布すべての出発点
binomial-distribution|1|30|bernoulli-distribution|8|最頻出の分布。比率の推定・検定の土台
hypergeometric-distribution|1|25|binomial-distribution|0|非復元抽出の分布。期待値・分散の計算が出る
finite-population-correction|3|15|hypergeometric-distribution|-5|非復元抽出の分散の補正。標本調査の公式に直結
fisher-exact-test|3|25|hypergeometric-distribution two-by-two-table|0|小標本の2×2表の検定。超幾何分布の応用として出る
poisson-distribution|1|30|binomial-distribution|8|まれな事象の分布。確率過程・回帰・ベイズで繰り返し登場
geometric-distribution|1|25|bernoulli-distribution|0|無記憶性の計算が頻出。台の流儀の違いに注意
negative-binomial-distribution|1|25|geometric-distribution|0|幾何分布の一般化。過分散のモデルにも使う
multinomial-distribution|1|25|binomial-distribution|0|適合度検定と分割表の土台になる分布
normal-distribution|1|40|expectation-variance-properties|12|あらゆる推定・検定の前提となる最重要分布
standard-normal-distribution|1|20|normal-distribution|5|標準化と数表の読み方は毎回の計算で必須
uniform-distribution|1|15|expectation-variance-properties|-3|乱数生成や順序統計量の例題で使う基本分布
exponential-distribution|1|30|expectation-variance-properties|3|無記憶性・ポアソン過程・生存時間で頻出
gamma-distribution|1|30|exponential-distribution|3|指数分布の和やベイズの共役事前分布で頻出
beta-distribution|1|25|gamma-distribution|0|比率のベイズ推定と順序統計量で使う
cauchy-distribution|3|15|t-distribution|-3|期待値が存在しない分布の代表。反例として問われる
lognormal-distribution|3|15|normal-distribution|-3|対数をとると正規分布になる分布。期待値の式が出る
bivariate-normal-distribution|1|35|normal-distribution covariance-and-correlation|5|条件付き分布や回帰との関係が頻出
multivariate-normal-distribution|1|30|bivariate-normal-distribution variance-covariance-matrix|0|回帰・判別分析・グラフィカルモデルの前提
normal-mixture-distribution|3|20|normal-distribution conditional-variance|-3|EMアルゴリズムや混合モデルの土台
chi-squared-distribution|1|30|standard-normal-distribution gamma-distribution|8|分散の推定・検定と適合度検定の土台
noncentral-chi-squared-distribution|3|15|chi-squared-distribution|-5|検出力の計算で使う。定義を押さえれば十分
t-distribution|1|30|chi-squared-distribution|8|母平均の推定・検定に必須の分布
t-statistic|1|20|t-distribution|5|t検定・回帰係数の検定のすべてで使う統計量
noncentral-t-distribution|3|15|t-distribution|-3|t検定の検出力計算の前提
f-distribution|1|25|chi-squared-distribution|5|分散比の検定と分散分析・回帰のF検定の土台
noncentral-f-distribution|4|10|f-distribution noncentral-chi-squared-distribution|0|分散分析の検出力で使う。出題はまれ
markov-inequality|3|15|expectation-variance-properties|-3|チェビシェフの不等式の元になる不等式
chebyshev-inequality|1|20|expectation-variance-properties|0|確率の上限評価と大数の法則の証明で使う
convergence-in-probability|1|20|chebyshev-inequality|0|一致性と大数の法則を理解するための基本概念
almost-sure-convergence|2|20|convergence-in-probability|-3|新しい出題範囲表に明記。確率収束との強弱関係が問われる
convergence-in-distribution|1|20|convergence-in-probability|0|中心極限定理と漸近正規性の前提
convergence-in-mean-square|4|10|convergence-in-probability|0|収束の強弱関係の一つとして知っていれば十分
weak-law-of-large-numbers|1|20|convergence-in-probability|0|標本平均が母平均に近づく根拠。一致性の基礎
strong-law-of-large-numbers|4|10|weak-law-of-large-numbers almost-sure-convergence|-5|弱法則との違いを知っていれば足りる
central-limit-theorem|1|35|convergence-in-distribution normal-distribution|12|正規近似の根拠。推定・検定の多くを支える最重要定理
slutsky-theorem|2|20|central-limit-theorem|0|新しい出題範囲表に明記。漸近分布の組み合わせで使う
continuous-mapping-theorem|2|15|convergence-in-distribution|-3|新しい出題範囲表に明記。デルタ法と組で使う
delta-method|2|30|central-limit-theorem|5|推定量の関数の漸近分散を出す定番。計算問題で頻出
continuity-correction|1|15|normal-approximation-binomial-poisson|0|離散分布の正規近似の精度を上げる工夫。計算で出る
law-of-small-numbers|3|15|poisson-distribution|-3|二項分布をポアソン分布で近似する根拠
lindeberg-levy-clt|4|5|central-limit-theorem|0|中心極限定理の項目に統合済み。そちらで確認すれば十分
berry-esseen-theorem|4|10|central-limit-theorem|0|正規近似の誤差の評価。範囲表になく優先度は低い
normal-approximation-binomial-poisson|1|25|central-limit-theorem binomial-distribution poisson-distribution|5|比率やポアソンの検定・区間推定の計算の土台
extreme-value-distribution|3|20|order-statistics|-3|最大値の極限分布。順序統計量と組で出る
point-estimation|1|15|expectation-variance-properties|0|推定の用語の出発点。推定量と推定値の区別
unbiased-estimator|1|25|point-estimation|8|不偏分散など計算問題で頻出。推定量の評価の基本
consistent-estimator|1|20|unbiased-estimator weak-law-of-large-numbers|0|大標本での推定量の性質。定義が問われる
efficient-estimator|2|20|cramer-rao-inequality|0|クラメール・ラオの下限との関係で問われる
mean-squared-error|1|25|unbiased-estimator|5|バイアスと分散の分解は推定量の比較の定番
likelihood-function|1|30|point-estimation probability-mass-and-density-function|8|最尤法・検定・ベイズ・モデル選択すべての土台
maximum-likelihood-estimator|1|45|likelihood-function|10|各分布での導出が頻出。推定の最重要手法
score-function|2|20|maximum-likelihood-estimator|0|フィッシャー情報量とスコア検定の前提
fisher-information|2|35|score-function|5|計算問題で頻出。漸近分散と検定の基礎
cramer-rao-inequality|2|30|fisher-information unbiased-estimator|3|不偏推定量の分散の下限。有効性の判定で問われる
sufficient-statistic|2|30|likelihood-function|3|データの要約の考え方。分解定理と組で頻出
factorization-theorem|2|25|sufficient-statistic|0|十分統計量を見つける定番の方法
method-of-moments|1|30|moments point-estimation|5|最尤法と並ぶ推定法。計算問題で頻出
asymptotic-normality-of-mle|2|30|fisher-information central-limit-theorem|5|最尤推定量の分散の近似と信頼区間の根拠
invariance-of-mle|3|10|maximum-likelihood-estimator|-5|関数の最尤推定量がすぐ求まる便利な性質
rao-blackwell-theorem|3|20|sufficient-statistic mean-squared-error|-3|十分統計量で推定量を改良する定理
complete-statistic|4|15|sufficient-statistic|0|一様最小分散不偏推定の理論。準1級では深入り不要
exponential-family|3|25|sufficient-statistic|0|十分統計量や一般化線形モデルを統一的に見る枠組み
observed-fisher-information|4|5|fisher-information|0|フィッシャー情報量の項目に統合済み
order-statistics|2|30|cumulative-distribution-function|3|最大値・最小値の分布は計算問題でよく出る
standard-error|1|15|unbiased-estimator|3|区間推定・検定・回帰出力の読み取りで常に使う
confidence-interval|1|25|standard-error normal-distribution|10|区間推定の意味は毎回問われる。解釈の誤りに注意
confidence-coefficient|1|10|confidence-interval|0|信頼区間の解釈と合わせて押さえる基本用語
pivotal-quantity|1|20|confidence-interval|0|信頼区間を作る統一的な方法
ci-mean-known-variance|1|20|pivotal-quantity standard-normal-distribution|0|区間推定の最も基本的な計算
ci-mean-unknown-variance|1|25|ci-mean-known-variance t-statistic|5|t分布を使う区間推定。計算問題で頻出
ci-variance|1|20|pivotal-quantity chi-squared-distribution|0|カイ二乗分布を使う区間推定の定番
ci-proportion|1|20|pivotal-quantity normal-approximation-binomial-poisson|5|世論調査型の計算問題で頻出
wald-confidence-interval|2|20|ci-proportion asymptotic-normality-of-mle|0|最尤推定量から作る近似区間。比率の区間の基本形
wilson-interval|4|15|wald-confidence-interval|-3|ワルド区間の改良版。違いを知っておく
ci-two-sample-mean-difference|1|25|ci-mean-unknown-variance pooled-variance|0|2群比較の区間推定。プールした分散と組で出る
ci-paired-difference|3|15|ci-mean-unknown-variance|-3|差をとって1標本に帰着させる考え方
ci-variance-ratio|2|20|ci-variance f-distribution|0|新しい出題範囲表に明記。F分布の使い方が問われる
required-sample-size|1|25|ci-proportion|5|区間の幅から標本サイズを逆算する計算が頻出
ci-two-sample-proportion|3|15|ci-proportion|-3|2群の比率の差の区間推定。検定と組で出る
clopper-pearson-interval|4|10|ci-proportion binomial-distribution|0|正確な比率の区間。名前と考え方を知る程度でよい
likelihood-based-confidence-interval|4|15|wilks-theorem|0|尤度比から作る区間。出題はまれ
confidence-interval-test-duality|2|15|confidence-interval rejection-region|0|区間と検定を互いに読み替える問題が出る
one-sided-confidence-limit|3|10|confidence-interval one-sided-two-sided-test|-3|片側の上限・下限。片側検定と対応させて覚える
null-and-alternative-hypothesis|1|15|normal-distribution|5|検定の出発点。仮説の立て方は毎回問われる
type-i-and-ii-errors|1|20|null-and-alternative-hypothesis|5|過誤の確率計算と定義は頻出
significance-level|1|10|type-i-and-ii-errors|0|第1種の過誤の上限。p値との関係が問われる
p-value|1|20|significance-level|8|解釈と計算が頻出。誤解しやすい点が狙われる
rejection-region|1|15|significance-level|0|検定統計量の値で判定する仕組み
power-function|1|25|type-i-and-ii-errors rejection-region|5|検出力の計算は頻出。標本サイズ設計の前提
one-sided-two-sided-test|1|15|rejection-region|0|棄却域の置き方とp値の計算に直結
neyman-pearson-lemma|2|30|power-function likelihood-function|3|最強力検定の作り方。導出問題で出る
uniformly-most-powerful-test|2|20|neyman-pearson-lemma|0|片側検定が最強力になる条件が問われる
monotone-likelihood-ratio|4|15|uniformly-most-powerful-test|-3|一様最強力検定が存在する条件
likelihood-ratio-test|2|30|maximum-likelihood-estimator rejection-region|5|多くの検定を統一的に導く方法。頻出
wilks-theorem|2|20|likelihood-ratio-test chi-squared-distribution|3|尤度比の漸近カイ二乗分布。自由度の数え方が問われる
wald-test|2|20|asymptotic-normality-of-mle|0|最尤推定量を使う検定。回帰の出力とも結び付く
score-test|2|20|score-function fisher-information|0|3つの漸近検定の一つ。比率の検定と同じ形になる
multiple-testing|2|15|significance-level|0|多数の検定で第1種の過誤が増える問題
bonferroni-correction|2|15|multiple-testing|0|多重検定の最も基本的な補正。計算が簡単で出やすい
holm-method|3|15|bonferroni-correction|-3|ボンフェローニの改良版。手順を押さえる
false-discovery-rate|3|20|multiple-testing|-3|大規模な多重検定の考え方。BH法の手順が問われる
sample-size-determination-test|4|5|required-sample-size power-function|0|必要な標本サイズの項目に統合済み
randomized-test|4|10|neyman-pearson-lemma|0|離散分布で有意水準を合わせる理論上の工夫
z-test-known-variance|1|20|rejection-region standard-normal-distribution|3|検定の手順を学ぶ最初の例
one-sample-t-test|1|25|z-test-known-variance t-statistic|8|計算問題で頻出の基本検定
two-sample-t-test|1|25|one-sample-t-test pooled-variance|5|2群の平均比較。出力の読み取りも出る
pooled-variance|1|15|unbiased-estimator|0|2標本t検定と分散分析で使う
welch-test|1|20|two-sample-t-test|3|等分散を仮定しない2群比較。自由度の近似が問われる
paired-t-test|1|20|one-sample-t-test|3|対応のあるデータは差の1標本検定に帰着する
chi-square-test-variance|1|20|chi-squared-distribution rejection-region|0|母分散の検定。カイ二乗分布の使い方の定番
f-test-equal-variances|1|20|f-distribution rejection-region|0|分散比の検定。ウェルチ検定の選択と結び付く
bartlett-test|4|10|f-test-equal-variances|0|3群以上の等分散の検定。名前と用途で足りる
test-ci-relationship-mean-difference|4|5|confidence-interval-test-duality|0|信頼区間と検定の対応の項目に統合済み
test-correlation-coefficient|3|20|covariance-and-correlation t-statistic|0|相関の有無をt統計量で検定する
fisher-z-transformation|3|20|test-correlation-coefficient|0|相関係数の区間推定と2群比較で使う
power-of-t-test|3|20|power-function one-sample-t-test noncentral-t-distribution|0|非心t分布を使う検出力の計算
preliminary-test-problem|4|10|welch-test f-test-equal-variances|0|等分散の予備検定を挟む問題点。概念だけ押さえる
normality-test|4|10|normal-qq-plot|0|正規性の確認方法。範囲表になく優先度は低い
hotelling-t2|4|15|multivariate-normal-distribution two-sample-t-test|0|多変量の平均の検定。準1級では主題にならない
one-way-test-variance-equality-levene|4|5|bartlett-test|0|バートレット検定の項目に統合済み
proportion-test|1|20|normal-approximation-binomial-poisson rejection-region|5|比率の検定は計算問題の定番
binomial-test|1|15|binomial-distribution p-value|0|二項分布で直接p値を出す正確な検定
poisson-parameter-test|2|20|poisson-distribution p-value|0|件数データの検定。正規近似と正確法の両方が出る
two-sample-proportion-test|1|20|proportion-test|3|2群の比率比較。プールした比率を使う
goodness-of-fit-test|1|25|pearson-chi-square-statistic|5|観測度数と期待度数の比較。頻出の定番
pearson-chi-square-statistic|1|20|expected-frequency chi-squared-distribution|5|適合度と独立性の検定に共通する統計量
independence-test|1|25|goodness-of-fit-test contingency-table|5|分割表の検定の代表。自由度の計算が頻出
homogeneity-test|2|15|independence-test|0|独立性の検定と同じ計算で意味が異なる
g-statistic|2|20|pearson-chi-square-statistic likelihood-ratio-test|0|尤度比による適合度検定。ピアソン統計量と比べる
expected-frequency|1|15|multinomial-distribution|0|カイ二乗検定で最初に計算する量
degrees-of-freedom-gof|1|15|goodness-of-fit-test|3|推定した母数の数だけ自由度が減る点が頻出
chi-square-approximation-condition|4|10|pearson-chi-square-statistic|-3|期待度数が小さいときの注意点
variance-stabilizing-transform|4|15|delta-method|0|デルタ法の応用。出題はまれ
poisson-two-sample-test|3|20|poisson-parameter-test|0|2群のポアソン率の比較。条件付き二項検定に帰着する
kolmogorov-smirnov-test|4|15|cumulative-distribution-function|0|経験分布を使う適合度検定。範囲表にない
dispersion-test|4|10|poisson-distribution|0|ポアソン分布の過分散の確認。出題はまれ
rank-and-ties|2|15|order-statistics|0|順位を使う検定に共通する準備
nonparametric-methods-overview|4|10|rank-and-ties|-3|分布を仮定しない方法の長所と短所
sign-test|3|15|binomial-test|0|二項検定に帰着する最も簡単なノンパラメトリック検定
wilcoxon-signed-rank-test|3|25|sign-test rank-and-ties|0|対応のあるデータの順位検定。統計量の計算が出る
wilcoxon-rank-sum-test|2|25|rank-and-ties|3|2群の順位検定の代表。期待値と分散の計算が出る
mann-whitney-u-test|4|5|wilcoxon-rank-sum-test|0|順位和検定の項目に統合済み
kruskal-wallis-test|3|20|wilcoxon-rank-sum-test|0|一元配置分散分析のノンパラメトリック版
friedman-test|3|20|kruskal-wallis-test|-3|乱塊法に対応するノンパラメトリック検定
spearman-rank-correlation|2|20|rank-and-ties covariance-and-correlation|3|順位相関の計算式が頻出
kendall-rank-correlation|3|20|concordant-discordant-pairs|0|一致対と不一致対の数から求める順位相関
permutation-test|3|20|p-value|0|並べ替えでp値を出す考え方。シミュレーションと結び付く
normal-approximation-rank-tests|3|15|wilcoxon-rank-sum-test central-limit-theorem|0|順位統計量の期待値と分散による正規近似
asymptotic-relative-efficiency|4|10|wilcoxon-rank-sum-test|0|検定の効率の比較。代表的な数値を知る程度で足りる
median-test|4|10|sign-test|0|中央値の比較の検定。範囲表になく優先度は低い
kolmogorov-smirnov-two-sample|4|5|kolmogorov-smirnov-test|0|コルモゴロフ・スミルノフ検定の項目に統合済み
runs-test|4|10|rank-and-ties|0|並びの無作為性の検定。出題はまれ
concordant-discordant-pairs|3|10|rank-and-ties|-3|ケンドールの順位相関の計算の前提
jonckheere-terpstra-test|4|10|kruskal-wallis-test|0|順序のある対立仮説の検定。出題はまれ
markov-property|4|5|markov-chain|0|マルコフ連鎖の項目に統合済み
markov-chain|2|25|conditional-probability|5|状態の推移を扱う基本モデル。毎回のように出る分野
transition-probability-matrix|2|25|markov-chain|5|行列計算の問題が頻出
n-step-transition-probability|2|20|transition-probability-matrix|0|行列のべき乗で求める計算が出る
chapman-kolmogorov-equation|3|15|n-step-transition-probability|-3|n段階推移の計算の根拠
stationary-distribution|2|30|transition-probability-matrix|8|連立方程式を解いて求める問題が頻出
irreducibility|2|15|communicating-class|0|定常分布が一つに決まる条件
communicating-class|2|15|transition-probability-matrix|-3|既約性や再帰性を判定する準備
periodicity|2|15|irreducibility|0|極限分布が存在する条件として問われる
recurrence-and-transience|2|20|communicating-class|0|状態の分類の基本。判定問題が出る
absorbing-state|2|20|transition-probability-matrix|0|吸収確率の計算問題の前提
ergodicity|2|20|periodicity recurrence-and-transience stationary-distribution|0|定常分布への収束の条件。MCMCの根拠になる
detailed-balance|2|20|stationary-distribution|0|定常分布を簡単に求める条件。MCMCで再登場
limiting-distribution|3|15|ergodicity|0|定常分布との違いを押さえる
absorption-probability|3|25|absorbing-state|0|一歩目で場合分けする計算。破産問題と同じ形
first-passage-time|3|20|recurrence-and-transience|0|平均再帰時間と定常分布の関係が問われる
positive-recurrence|4|5|recurrence-and-transience|0|再帰と過渡の項目に統合済み
continuous-time-markov-chain|4|15|markov-chain exponential-distribution|0|出生死滅過程の一般形。深入りは不要
stochastic-process|2|10|markov-chain|-5|確率過程の用語の整理。ポアソン過程の前提
counting-process|4|5|poisson-process|0|ポアソン過程の項目に統合済み
poisson-process|2|30|poisson-distribution exponential-distribution stochastic-process|5|範囲表に明記。到着回数と間隔の計算が頻出
interarrival-time|3|15|poisson-process|0|指数分布との対応が問われる
superposition-thinning-poisson|3|15|poisson-process|0|合成や間引きをしてもポアソン過程になる性質
random-walk|2|20|bernoulli-distribution stochastic-process|0|確率過程の基本例。破産問題の前提
gambler-ruin-problem|3|25|random-walk absorption-probability|0|吸収確率の典型問題
brownian-motion|3|20|random-walk normal-distribution|0|ランダムウォークの連続版。性質を問う問題が出る
martingale|4|15|conditional-expectation|0|範囲表になく優先度は低い
birth-death-process|3|25|poisson-process|0|待ち行列の土台。定常分布の計算が出る
pure-birth-process|4|5|birth-death-process|0|出生死滅過程の項目に統合済み
queueing-theory|4|5|mm1-queue|0|M/M/1待ち行列の項目に統合済み
mm1-queue|3|25|birth-death-process|0|待ち行列の代表。平均系内人数の計算が出る
traffic-intensity|3|10|mm1-queue|0|M/M/1が安定する条件。計算の最初に出る
littles-law|3|10|mm1-queue|0|平均人数と平均待ち時間をつなぐ公式
compound-poisson-process|4|15|poisson-process conditional-variance|0|保険数理型の応用。出題はまれ
independent-stationary-increments|3|10|poisson-process|-3|ポアソン過程とブラウン運動に共通する性質
mmc-queue|4|15|mm1-queue|0|窓口が複数ある待ち行列。発展的
reflection-principle|4|15|random-walk|0|ランダムウォークの経路の数え方。発展的
multiple-regression-model|2|30|multivariate-normal-distribution|10|回帰はほぼ毎回出る。すべての回帰手法の土台
least-squares-method|2|30|multiple-regression-model|10|推定量の導出と性質が頻出
normal-equations|2|25|least-squares-method|5|最小二乗推定量を行列で導く式
hat-matrix|2|25|normal-equations|3|てこ比や残差の性質の前提
gauss-markov-theorem|2|20|normal-equations|3|最小二乗推定量が最良線形不偏である根拠
coefficient-of-determination|2|20|least-squares-method|8|出力表の読み取りと平方和の分解で頻出
adjusted-r-squared|2|15|coefficient-of-determination|3|変数の数を考慮した決定係数。計算が出る
regression-coefficient-t-test|2|25|normal-equations t-statistic|8|出力表の読み取り問題で頻出
overall-f-test-regression|2|20|coefficient-of-determination f-distribution|5|決定係数とF値の関係が問われる
standardized-partial-regression-coefficient|3|15|least-squares-method|0|単位の違う説明変数の効果を比べる
multicollinearity|2|20|multiple-regression-model correlation-matrix|5|係数が不安定になる理由が問われる
variance-inflation-factor|2|15|multicollinearity coefficient-of-determination|3|多重共線性の数値指標。計算が出る
dummy-variable|2|15|multiple-regression-model|0|質的変数を回帰に入れる方法。係数の解釈が問われる
interaction-term|3|15|dummy-variable|0|効果がほかの変数で変わることを表す項
confidence-and-prediction-interval-regression|2|20|regression-coefficient-t-test|0|信頼区間と予測区間の違いが問われる
partial-f-test|3|15|overall-f-test-regression|0|説明変数の一部を外してよいかの検定
residual|2|15|least-squares-method|3|回帰診断の出発点
standardized-residual|2|15|residual hat-matrix|0|外れ値の目安として使う
studentized-residual|3|15|standardized-residual|0|その点を除いた分散で割る残差
leverage|2|20|hat-matrix|3|ハット行列の対角成分。頻出
cooks-distance|2|20|leverage standardized-residual|3|新しい出題範囲表に明記。影響の大きい点の判定
dffits|3|15|cooks-distance|-3|影響度を測る別の指標
dfbetas|4|5|dffits|0|DFFITSの項目に統合済み
outlier-and-influential-point|2|15|leverage residual|0|外れ値と影響点の区別が問われる
residual-plot|2|15|residual|3|図の読み取り問題で出る
normal-qq-plot|2|15|residual quantile-function|3|正規性の確認。図の読み取りで出る
heteroscedasticity|3|20|residual-plot|0|誤差の分散が一定でない場合の問題と対処
serial-correlation|3|15|residual|0|誤差の自己相関。ダービン・ワトソン比の前提
durbin-watson-statistic|3|15|serial-correlation|0|系列相関の検定統計量。値の範囲と解釈が出る
breusch-pagan-test|4|10|heteroscedasticity|0|不均一分散の検定。範囲表にない
robust-standard-error|4|15|heteroscedasticity|0|不均一分散でも使える標準誤差。発展的
logistic-regression|2|30|logit-logistic-transformation maximum-likelihood-estimator odds-and-odds-ratio|8|2値の結果を扱う回帰の代表。出力の解釈が頻出
odds-and-odds-ratio|2|20|conditional-probability|5|ロジスティック回帰と分割表に共通する尺度
generalized-linear-model|2|25|logistic-regression multiple-regression-model|5|線形回帰・ロジスティック・ポアソン回帰を統一する枠組み
link-function|2|15|generalized-linear-model|0|平均と線形予測子をつなぐ関数。正準リンクが問われる
exponential-dispersion-family|3|15|generalized-linear-model exponential-family|0|一般化線形モデルで使う分布の共通の形
probit-model|3|15|logistic-regression probit-transformation|-3|ロジットモデルとの違いを押さえる
poisson-regression|2|20|generalized-linear-model poisson-distribution|5|新しい出題範囲表に明記。件数データの回帰
deviance|2|20|generalized-linear-model likelihood-ratio-test|3|モデルの当てはまりと比較の尺度
logistic-likelihood-estimation|3|20|logistic-regression|0|対数尤度の形と推定の仕組み
iteratively-reweighted-least-squares|3|15|logistic-likelihood-estimation weighted-least-squares|-3|一般化線形モデルの数値解法
multinomial-logit-model|3|15|logistic-regression|0|3つ以上のカテゴリを扱う回帰
ordered-logit-model|3|15|multinomial-logit-model|0|順序のあるカテゴリを扱う回帰
tobit-model|4|15|censored-data|0|打ち切りのある目的変数の回帰。発展的
overdispersion|3|15|poisson-regression|0|ポアソン回帰の仮定が崩れる状況
offset-term|4|10|poisson-regression|0|観測期間や人数の違いを調整する項
weighted-least-squares|3|20|least-squares-method heteroscedasticity|0|不均一分散への対処法
generalized-least-squares|3|20|weighted-least-squares serial-correlation|0|誤差の相関を考慮した推定
ridge-regression|2|25|regularization multicollinearity|5|正則化の代表。推定量の式が出る
lasso|2|20|regularization|3|変数選択を兼ねる正則化。リッジとの違いが頻出
elastic-net|4|10|lasso ridge-regression|0|リッジとLassoの組み合わせ。名前と性質で足りる
regularization|2|20|least-squares-method overfitting|3|過学習を抑える罰則の考え方。新しい範囲表に明記
bias-in-penalized-estimation|4|5|ridge-regression|0|リッジ回帰の項目に統合済み
nonlinear-regression|3|15|least-squares-method|-3|パラメータについて非線形なモデル
polynomial-regression|3|15|multiple-regression-model|-3|線形回帰の枠内で曲線を当てはめる
quantile-regression|3|15|quantile-function least-squares-method|-3|平均ではなく分位点を予測する回帰
instrumental-variable-method|3|25|endogeneity|0|内生性への対処の代表的な方法
endogeneity|3|20|least-squares-method|0|説明変数と誤差の相関が推定を偏らせる問題
two-stage-least-squares|3|20|instrumental-variable-method|0|操作変数法の具体的な計算手順
variable-selection|2|20|adjusted-r-squared aic|3|ステップワイズや情報量規準による説明変数の選び方
omitted-variable-bias|3|15|endogeneity|0|必要な変数を落とすと係数が偏る仕組み
principal-component-regression|4|10|principal-component-analysis multicollinearity|0|主成分を説明変数にする回帰。発展的
spline-regression|4|10|polynomial-regression|0|区分的な多項式の回帰。範囲表にない
neural-network-model|4|15|logistic-regression|0|範囲表にあるが準1級では概念程度でよい
one-way-anova|2|30|f-distribution two-sample-t-test|8|分散分析表の計算が頻出
sum-of-squares-decomposition|2|20|one-way-anova|5|分散分析と回帰に共通する考え方
two-way-anova-with-replication|2|30|main-effect-and-interaction|5|交互作用を含む分散分析表の計算が出る
two-way-anova-without-replication|2|25|one-way-anova|3|乱塊法と同じ形の分散分析
main-effect-and-interaction|2|20|two-way-anova-without-replication|5|交互作用の解釈が頻出
fixed-and-random-effects|3|20|one-way-anova|0|母数模型と変量模型の違い
expected-mean-square|3|20|fixed-and-random-effects|0|F比の分母の選び方の根拠
fisher-three-principles|2|15||3|反復・無作為化・局所管理。概念問題で頻出
randomized-block-design|2|20|fisher-three-principles two-way-anova-without-replication|0|局所管理の代表的な計画
latin-square-design|2|20|randomized-block-design|0|2つの局外要因を同時に除く計画
orthogonal-array|2|25|factorial-design|0|直交表への割り付けが出る
factorial-design|2|15|main-effect-and-interaction|-3|複数の要因を組み合わせる計画
split-plot-design|3|20|randomized-block-design|0|誤差が2種類ある計画
confounding|3|15|factorial-design|0|効果が分離できない状態。一部実施計画と因果推論に結び付く
tukey-multiple-comparison|2|20|one-way-anova multiple-testing|0|分散分析後の対比較の定番
studentized-range-distribution|4|5|tukey-multiple-comparison|0|テューキーの多重比較の項目に統合済み
effect-size-eta-squared|4|10|sum-of-squares-decomposition|0|分散分析の寄与率。決定係数と同じ考え方
fractional-factorial-design|3|20|orthogonal-array confounding|0|実験回数を減らす計画。交絡との関係が要点
incomplete-block-design|4|15|randomized-block-design|0|ブロックに全処理が入らない計画。発展的
simple-random-sampling|2|20|hypergeometric-distribution|3|標本調査の基本。推定量の分散の式が出る
horvitz-thompson-estimator|3|20|inclusion-probability|0|不等確率抽出での推定量の基本形
inclusion-probability|3|15|simple-random-sampling|0|ホルビッツ・トンプソン推定量の前提
stratified-sampling|2|25|simple-random-sampling|5|分散が小さくなる理由と計算が出る
proportional-allocation|3|15|stratified-sampling|0|層の大きさに比例させる配分
neyman-allocation|3|20|proportional-allocation|0|分散を最小にする配分。計算が出る
cluster-sampling|3|20|simple-random-sampling|0|集落単位で抽出する方法。集落内の似かよりが効く
two-stage-sampling|3|20|cluster-sampling|0|集落の中からさらに抽出する方法
systematic-sampling|4|10|simple-random-sampling|-3|等間隔の抽出。周期性に注意
ratio-estimation|3|20|simple-random-sampling|0|補助変数を使う推定。分散の近似が出る
regression-estimation|4|15|ratio-estimation|0|比推定の一般化。発展的
design-effect|4|15|cluster-sampling|0|単純無作為抽出との分散の比
sampling-and-nonsampling-error|2|10|simple-random-sampling|0|概念問題で頻出
probability-proportional-to-size-sampling|4|15|horvitz-thompson-estimator|0|規模に比例させる抽出。発展的
nonresponse-bias|3|10|sampling-and-nonsampling-error|-3|非標本誤差の代表。欠測データと結び付く
sampling-weight|4|5|horvitz-thompson-estimator|0|ホルビッツ・トンプソン推定量の項目に統合済み
principal-component-analysis|2|30|variance-covariance-matrix eigenvalue-and-eigenvector|10|頻出の多変量手法。固有値問題として出る
principal-component|4|5|principal-component-analysis|0|主成分分析の項目に統合済み
eigenvalue-and-eigenvector|2|25|variance-covariance-matrix|5|主成分・因子分析・判別分析の計算の土台
contribution-ratio|2|15|principal-component-analysis|3|固有値の比で求める。計算が頻出
cumulative-contribution-ratio|2|10|contribution-ratio|0|主成分の数を決める目安
principal-component-loading|2|15|principal-component-analysis|0|主成分と元の変数の相関。解釈で出る
principal-component-score|2|15|principal-component-analysis|0|各個体の主成分の値。計算が出る
pca-on-correlation-matrix|2|15|principal-component-analysis correlation-matrix|3|単位の違う変数で使う。共分散行列との違いが問われる
scree-plot|3|10|cumulative-contribution-ratio|0|主成分の数を図で決める方法
kaiser-criterion|4|10|pca-on-correlation-matrix|-3|固有値1以上を採る目安
singular-value-decomposition|3|20|eigenvalue-and-eigenvector|0|主成分分析やコレスポンデンス分析の計算の土台
lagrange-multiplier-pca|4|20|principal-component-analysis|-3|主成分を分散の最大化から導く手順
biplot|4|10|principal-component-score|0|得点と負荷量を重ねた図
low-rank-approximation|4|10|singular-value-decomposition|0|特異値分解による行列の近似。発展的
linear-discriminant-analysis|2|25|multivariate-normal-distribution mahalanobis-distance|5|判別分析の基本。範囲表に明記
fisher-linear-discriminant-function|2|20|linear-discriminant-analysis|3|群間と群内の分散の比を最大にする関数
mahalanobis-distance|2|20|variance-covariance-matrix|5|判別や外れ値の検出で使う距離
quadratic-discriminant-analysis|3|15|linear-discriminant-analysis|0|共分散が群ごとに違う場合の判別。範囲表に明記
bayes-discriminant-rule|3|20|bayes-theorem linear-discriminant-analysis|0|事後確率が最大の群に判別する考え方
misclassification-rate|2|15|linear-discriminant-analysis|0|判別の性能の基本的な指標
confusion-matrix|2|15|misclassification-rate|3|感度や適合率を計算する表。頻出
sensitivity-and-specificity|2|15|confusion-matrix|5|医療検査型の問題で頻出
roc-curve-and-auc|2|20|sensitivity-and-specificity|3|判別性能の評価。図の読み取りが出る
precision-recall-f-measure|3|15|confusion-matrix|0|不均衡なデータで使う評価指標
k-nearest-neighbors|3|10|distance-measures|-3|近い点の多数決で判別する方法
support-vector-machine|3|20|linear-discriminant-analysis|0|範囲表に明記。マージン最大化の考え方
kernel-method|4|15|support-vector-machine|0|非線形への拡張。発展的
decision-tree|4|10|misclassification-rate|0|範囲表にない。概念だけでよい
naive-bayes-classifier|4|10|bayes-discriminant-rule conditional-independence|0|条件付き独立を仮定した判別。発展的
hierarchical-clustering|2|20|distance-measures|5|クラスター分析の基本。併合の手順をたどる問題が出る
dendrogram|2|15|hierarchical-clustering|3|図の読み取りが頻出
single-linkage|2|15|hierarchical-clustering|0|手計算で併合をたどる問題が出る
complete-linkage|3|10|single-linkage|0|最短距離法との違いが問われる
average-linkage|3|10|single-linkage|0|最短・最長距離法の中間の方法
ward-method|2|20|hierarchical-clustering|3|平方和の増加で併合する代表的な方法
centroid-method|4|10|average-linkage|-3|重心間の距離で併合する方法
lance-williams-formula|4|15|single-linkage complete-linkage|-3|各方法の距離の更新を一つの式で表す
k-means-method|2|20|distance-measures|5|非階層的手法の代表。反復の手順が出る
non-hierarchical-clustering|4|5|k-means-method|0|k-means法の項目に統合済み
distance-measures|2|15||3|ユークリッド距離など。クラスター分析と判別の前提
similarity-measures|4|10|distance-measures|-3|2値データなどで使う類似度
number-of-clusters|3|10|dendrogram k-means-method|-3|クラスター数の決め方の目安
mixture-model-clustering|4|15|normal-mixture-distribution em-algorithm|0|確率モデルに基づくクラスタリング。発展的
standardization-before-clustering|4|5|distance-measures|0|距離の項目に統合済み
factor-analysis-model|3|25|principal-component-analysis|5|範囲表に明記。主成分分析との違いが問われる
common-factor-unique-factor|3|15|factor-analysis-model|0|因子分析モデルの構成要素
factor-loading|3|20|common-factor-unique-factor|3|共分散の構造の計算と解釈で出る
communality|3|15|factor-loading|3|分散の分解の計算問題で出る
factor-rotation|3|15|factor-loading|0|解釈しやすくするための回転
varimax-rotation|4|10|factor-rotation|-3|直交回転の代表
promax-rotation|4|10|factor-rotation|0|斜交回転の代表。名前と性質で足りる
number-of-factors|4|10|factor-analysis-model|0|因子数の決め方の目安
factor-score|4|10|factor-loading|0|各個体の因子の値の推定
maximum-likelihood-factor-analysis|4|15|factor-analysis-model maximum-likelihood-estimator|0|因子分析の推定法の一つ。発展的
graphical-model|3|20|conditional-independence|3|条件付き独立をグラフで表す考え方
undirected-graphical-model|3|20|graphical-model precision-matrix|0|精度行列の零成分と辺の対応が要点
precision-matrix|3|20|variance-covariance-matrix partial-correlation-coefficient|3|条件付き独立と零成分の対応で出る
covariance-selection|4|15|undirected-graphical-model|0|グラフの辺を選ぶ手続き。発展的
directed-acyclic-graph|3|20|graphical-model|3|新しい範囲表で因果推論の土台として明記
graphical-lasso|4|10|covariance-selection lasso|0|罰則で疎な精度行列を推定する。発展的
multidimensional-scaling|3|20|distance-measures eigenvalue-and-eigenvector|0|距離から座標を復元する方法
classical-mds|4|5|multidimensional-scaling|0|多次元尺度構成法の項目に統合済み
nonmetric-mds|4|10|multidimensional-scaling|0|順位だけを使う尺度構成法。発展的
canonical-correlation-analysis|3|20|correlation-matrix eigenvalue-and-eigenvector|0|2組の変数の相関を最大にする方法
canonical-variate|3|15|canonical-correlation-analysis|0|正準相関分析で得る合成変数
correspondence-analysis|3|20|contingency-table singular-value-decomposition|0|分割表を図にする方法
quantification-theory|3|15|dummy-variable|0|質的データを数量化する手法群
quantification-type-1|4|10|quantification-theory|-3|ダミー変数による回帰と同じもの
quantification-type-3|4|10|quantification-theory correspondence-analysis|0|コレスポンデンス分析とほぼ同じ手法
multiple-correspondence-analysis|4|5|quantification-type-3|0|数量化III類の項目に統合済み
structural-equation-modeling|3|20|path-analysis factor-analysis-model|0|潜在変数を含む因果モデル
path-analysis|3|20|multiple-regression-model|3|範囲表に明記。直接効果と間接効果の計算が出る
independent-component-analysis|4|10|principal-component-analysis|0|範囲表になく優先度は低い
goodness-of-fit-indices|4|5|structural-equation-modeling|0|構造方程式モデリングの項目に統合済み
time-series|2|10||0|時系列解析の用語の出発点
weak-stationarity|2|20|time-series covariance-and-correlation|5|時系列モデルの前提。定義が問われる
strict-stationarity|4|10|weak-stationarity|0|弱定常との違いを知っていれば足りる
autocovariance-autocorrelation|2|20|weak-stationarity|5|時系列の性質を表す基本量。計算が頻出
correlogram|3|10|autocovariance-autocorrelation|0|自己相関の図。モデルの見分けに使う
partial-autocorrelation|2|20|autocovariance-autocorrelation partial-correlation-coefficient|3|ARモデルの次数を見分ける指標
white-noise|2|10|weak-stationarity|3|時系列モデルの誤差の基本
ar-model|2|30|white-noise|8|時系列で最頻出のモデル。自己相関の計算が出る
ma-model|2|25|white-noise|5|ARと並ぶ基本モデル。自己相関の打ち切りが要点
arma-model|2|20|ar-model ma-model|3|ARとMAの組み合わせ
arima-model|2|20|arma-model unit-root|3|差分をとって非定常な系列を扱う
yule-walker-equations|2|25|ar-model autocovariance-autocorrelation|5|AR係数と自己相関の関係。計算問題で頻出
stationarity-condition|2|20|ar-model|3|ARモデルが定常になる条件が問われる
invertibility|3|15|ma-model|0|MAモデルの反転可能性の条件
unit-root|2|15|stationarity-condition|0|非定常の代表。差分の必要性につながる
spectrum|3|20|autocovariance-autocorrelation|0|周波数で見た時系列の性質
seasonal-adjustment|4|10|time-series|-3|季節変動を取り除く考え方
forecasting|3|15|arma-model|0|時系列モデルによる予測の計算
state-space-model|3|20|ar-model|0|範囲表に明記。観測と状態を分けるモデル
contingency-table|1|15||0|カテゴリデータを集計する表。独立性の検定の前提
two-by-two-table|1|15|contingency-table|0|オッズ比や正確検定の舞台になる最も基本の表
relative-risk|2|15|two-by-two-table|3|オッズ比との違いが問われる
risk-difference|4|10|relative-risk|-3|リスクの差による効果の表し方
odds-ratio-interval|2|20|odds-and-odds-ratio delta-method|3|対数オッズ比の標準誤差による区間推定
sampling-schemes|3|15|two-by-two-table multinomial-distribution|0|分割表の集め方による分布の違い
yates-correction|2|10|independence-test continuity-correction|0|新しい出題範囲表に明記。2×2表の連続修正
mcnemar-test|2|20|two-by-two-table|0|対応のある2値データの検定
loglinear-model|3|25|contingency-table poisson-regression|0|分割表の構造をモデルで表す
hierarchical-loglinear-model|3|15|loglinear-model|0|範囲表に明記。対数線形モデルの選び方
simpsons-paradox|2|15|two-by-two-table|3|層別で結論が逆転する現象。概念問題で出る
mantel-haenszel-test|3|20|three-way-table|0|層別した2×2表をまとめて検定する
mantel-haenszel-estimator|3|15|mantel-haenszel-test|-3|層に共通するオッズ比の推定
ordinal-contingency-table|4|10|contingency-table|0|順序のあるカテゴリの分割表。発展的
three-way-table|3|10|simpsons-paradox|0|層別した分割表の扱い
residual-analysis-contingency|3|15|independence-test|0|範囲表に明記。どのセルが効いているかを調べる
phi-coefficient-and-association|3|10|independence-test|0|分割表の関連の強さを表す指標
cohort-and-case-control-study|3|15|relative-risk odds-and-odds-ratio|0|研究デザインとオッズ比・相対リスクの使い分け
missing-data-mechanism|3|15||0|欠測データの扱いの出発点
mcar|3|10|missing-data-mechanism|0|完全にランダムな欠測の定義
mar|3|10|mcar|0|ほかの観測値で説明できる欠測。代入法の前提
mnar|3|10|mar|0|欠測自体が値に依存する場合。偏りの原因
complete-case-analysis|3|10|mcar|0|新しい範囲表に明記。欠測を除く方法の注意点
single-imputation|3|10|missing-data-mechanism|0|新しい範囲表に明記。一つの値で補う方法の問題点
multiple-imputation|3|20|single-imputation mar|0|代入の不確かさを反映させる方法
rubins-rules|3|15|multiple-imputation|0|多重代入の結果をまとめる公式
em-algorithm|3|25|maximum-likelihood-estimator|3|欠測や潜在変数がある場合の最尤推定の手順
censored-data|3|15|survival-function|0|生存時間解析の前提。打ち切りの扱い
truncated-data|4|10|censored-data|0|打ち切りとの違いを知っていれば足りる
kaplan-meier-estimator|3|25|censored-data|3|新しい範囲表に明記。生存関数の推定の計算が出る
greenwood-formula|4|15|kaplan-meier-estimator|0|カプラン・マイヤー推定量の分散の式
log-rank-test|3|20|kaplan-meier-estimator|0|生存曲線の比較の検定
cox-proportional-hazards-model|3|25|hazard-function censored-data|0|生存時間の回帰の代表
partial-likelihood|4|15|cox-proportional-hazards-model|0|比例ハザードモデルの推定の仕組み
hazard-ratio|3|10|cox-proportional-hazards-model|0|比例ハザードモデルの係数の解釈
nelson-aalen-estimator|4|10|kaplan-meier-estimator|0|累積ハザードの推定。発展的
ignorability|4|10|mar|0|欠測の無視可能性の理論。発展的
model-selection|2|10|likelihood-function|0|情報量規準や交差検証の出発点
aic|2|25|model-selection kullback-leibler-divergence|8|モデル比較の最頻出の規準
bic|2|15|aic|5|新しい範囲表に明記。AICとの違いが問われる
kullback-leibler-divergence|2|20|likelihood-function|3|分布の近さの尺度。AICの導出の土台
expected-log-likelihood|3|15|kullback-leibler-divergence|0|AICが推定している量
cross-validation|2|20|overfitting|3|予測の良さを評価する定番の方法
leave-one-out-cv|3|10|cross-validation|0|1個ずつ抜く交差検証
overfitting|2|15|model-selection|3|複雑すぎるモデルの問題。正則化の動機
bias-variance-tradeoff|2|20|mean-squared-error overfitting|3|新しい範囲表に明記。予測誤差の分解
mallows-cp|2|15|adjusted-r-squared|0|新しい範囲表に明記。回帰の変数選択の規準
information-criteria-comparison|4|5|model-selection|0|モデル選択の項目に統合済み
generalization-error|3|10|overfitting|0|未知のデータに対する誤差の考え方
training-test-split|4|5|cross-validation|0|交差検証法の項目に統合済み
aicc|4|10|aic|0|小標本向けのAIC。発展的
tic-ebic|4|5|aic|0|AICの項目に統合済み
prior-distribution|2|15|bayes-theorem|5|ベイズ法の出発点。頻出分野
posterior-distribution|2|20|prior-distribution likelihood-function|8|事後分布の導出は毎回のように出る
conjugate-prior|2|20|posterior-distribution|5|事後分布が同じ形になる組み合わせ。計算の定番
beta-binomial-model|2|25|conjugate-prior beta-distribution|5|比率のベイズ推定の代表。計算が頻出
gamma-poisson-model|2|20|conjugate-prior gamma-distribution|3|件数データのベイズ推定の代表
normal-normal-model|2|25|conjugate-prior|3|事後平均が重み付き平均になる計算が出る
noninformative-prior|3|10|prior-distribution|0|事前情報がない場合の事前分布
jeffreys-prior|3|15|noninformative-prior fisher-information|0|フィッシャー情報量から作る事前分布
posterior-mean-map|2|15|posterior-distribution|5|ベイズ推定値の計算が頻出
credible-interval|2|15|posterior-distribution confidence-interval|3|信頼区間との解釈の違いが問われる
predictive-distribution|2|20|posterior-distribution|0|次の観測値の分布。計算問題で出る
bayes-factor|3|15|marginal-likelihood|0|ベイズ流のモデル比較
marginal-likelihood|3|15|posterior-distribution|0|ベイズファクターやBICの土台
hierarchical-bayes-model|3|20|conjugate-prior|0|母数にさらに分布を置くモデル
empirical-bayes|4|15|hierarchical-bayes-model marginal-likelihood|0|データから事前分布を推定する方法。発展的
bayes-decision-theory|4|15|posterior-mean-map|0|損失関数とベイズ推定の関係。発展的
uniform-random-number|2|10|uniform-distribution|0|シミュレーションの出発点
linear-congruential-generator|4|10|uniform-random-number|-3|擬似乱数の作り方の基本
inverse-transform-sampling|2|20|uniform-random-number cumulative-distribution-function|5|乱数生成の定番。計算問題で頻出
rejection-sampling|2|20|uniform-random-number|0|採択確率の計算が出る
box-muller-method|2|15|uniform-random-number change-of-variables|0|正規乱数の作り方。変数変換の応用
monte-carlo-integration|2|15|weak-law-of-large-numbers uniform-random-number|3|乱数で積分を近似する。分散の評価が出る
importance-sampling|3|20|monte-carlo-integration|0|分布を変えて推定の精度を上げる方法
variance-reduction|4|15|monte-carlo-integration|0|対照変量法などの工夫。発展的
mcmc|3|20|ergodicity posterior-distribution|3|事後分布からのサンプリングの考え方
metropolis-hastings|3|20|mcmc detailed-balance|0|MCMCの代表的な手法。採択確率の式が出る
gibbs-sampling|3|20|mcmc conditional-probability-density-function|0|条件付き分布から順に生成するMCMC
burn-in-convergence|3|10|mcmc|-3|MCMCの初期値の影響と収束の確認
bootstrap|2|20|standard-error|5|再標本化で標準誤差を求める。頻出
bootstrap-confidence-interval|3|15|bootstrap confidence-interval|0|ブートストラップで作る信頼区間
jackknife|3|15|bootstrap|0|1個ずつ抜く再標本化。バイアス補正の式が出る
permutation-simulation|4|5|permutation-test|0|並べ替え検定の項目に統合済み
causal-effect|3|20|confounding|3|新しい範囲表で追加された因果推論の出発点
backdoor-criterion|3|20|causal-effect d-separation|0|交絡を調整する変数の選び方
d-separation|3|20|directed-acyclic-graph conditional-independence|0|DAGから条件付き独立を読み取る規則
frontdoor-criterion|4|15|backdoor-criterion|0|バックドア基準が使えない場合の調整法。発展的
"""


def load_plan():
    plan = yaml.safe_load((CONTENT / "plan.yaml").read_text(encoding="utf-8"))
    terms = {}
    order = []
    for ch in plan["chapters"]:
        for t in ch["terms"]:
            terms[t["id"]] = dict(t, chapter=int(ch["no"]))
            order.append(t["id"])
    return terms, order


def load_related_indegree(terms):
    indeg = defaultdict(int)
    for path in CONTENT.glob("ch*/*.md"):
        text = path.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---", text, re.S)
        if not m:
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        for r in fm.get("related") or []:
            if r in terms and r != fm.get("id"):
                indeg[r] += 1
    return indeg


def parse_table():
    rows = {}
    for line in TABLE.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 6:
            sys.exit(f"TABLE の列数が違う: {line}")
        tid, stage, minutes, pre, adj, reason = parts
        if tid in rows:
            sys.exit(f"TABLE に重複: {tid}")
        rows[tid] = {
            "stage": int(stage),
            "minutes": int(minutes),
            "prereq": pre.split(),
            "adj": int(adj),
            "reason": reason.strip(),
        }
    return rows


def build():
    terms, plan_order = load_plan()
    rows = parse_table()
    missing = [t for t in plan_order if t not in rows]
    extra = [t for t in rows if t not in terms]
    if missing or extra:
        sys.exit(f"TABLE と plan.yaml が一致しない。足りない: {missing} 余分: {extra}")

    warnings = []
    # see の統合先を前提に入れ、段階の矛盾を確認する
    for tid, r in rows.items():
        see = terms[tid].get("see")
        if see and see not in r["prereq"]:
            r["prereq"].insert(0, see)
        r["prereq"] = list(dict.fromkeys(r["prereq"]))[:3]
        for p in r["prereq"]:
            if p not in rows:
                sys.exit(f"{tid} の前提 {p} が plan.yaml にない")
            if rows[p]["stage"] > r["stage"]:
                warnings.append(f"{tid}(段階{r['stage']}) の前提 {p} が段階{rows[p]['stage']}")
    if warnings:
        sys.exit("段階の矛盾:\n" + "\n".join(warnings))

    # score の計算
    dependents = defaultdict(int)
    for r in rows.values():
        for p in r["prereq"]:
            dependents[p] += 1
    rel_in = load_related_indegree(terms)
    raw = {}
    for tid, r in rows.items():
        t = terms[tid]
        hub = min(10, 2 * dependents[tid] + rel_in[tid] // 3)
        raw[tid] = IMP_BASE[t["importance"]] + CH_WEIGHT[t["chapter"]] + hub + r["adj"]
    score = {}
    for s, (lo, hi) in BANDS.items():
        ids = [t for t in rows if rows[t]["stage"] == s]
        mn = min(raw[t] for t in ids)
        mx = max(raw[t] for t in ids)
        for t in ids:
            score[t] = round(lo + (hi - lo) * (raw[t] - mn) / (mx - mn)) if mx > mn else hi

    # 並び順: 段階ごとに、前提を先に出す。準備ができたものの中で
    # 「自分と、同じ段階で自分に依存する用語」の score の最大値が高いものを先に出す。
    children = defaultdict(list)
    for tid, r in rows.items():
        for p in r["prereq"]:
            children[p].append(tid)
    plan_idx = {t: i for i, t in enumerate(plan_order)}

    eff_cache = {}

    def eff(tid, seen=()):
        if tid in eff_cache:
            return eff_cache[tid]
        if tid in seen:
            sys.exit(f"前提が循環している: {' -> '.join(seen + (tid,))}")
        best = score[tid]
        for c in children[tid]:
            if rows[c]["stage"] == rows[tid]["stage"]:
                best = max(best, eff(c, seen + (tid,)) - 1)
        eff_cache[tid] = best
        return best

    ordered = []
    done = set()
    for s in sorted(BANDS):
        pending = {t for t in rows if rows[t]["stage"] == s}
        while pending:
            ready = [t for t in pending if all(p in done for p in rows[t]["prereq"])]
            if not ready:
                sys.exit(f"段階{s}で前提が循環している: {sorted(pending)[:10]}")
            ready.sort(key=lambda t: (-eff(t), -score[t], terms[t]["chapter"], plan_idx[t]))
            t = ready[0]
            ordered.append(t)
            done.add(t)
            pending.remove(t)

    return terms, rows, score, ordered


def q(s):
    """YAML の素の文字列として安全ならそのまま、だめなら二重引用符で囲む。"""
    try:
        if yaml.safe_load(f"k: {s}") == {"k": s}:
            return s
    except yaml.YAMLError:
        pass
    return json.dumps(s, ensure_ascii=False)


def write(rows, score, ordered):
    lines = [
        "# 学習ロードマップ（scripts/make_roadmap.py が生成。手で編集しない）",
        "# 根拠と参考URLは spec/research/roadmap.md",
        "stages:",
    ]
    for no, title, desc in STAGES:
        lines.append(f"  - {{'no': {no}, title: {q(title)}, desc: {q(desc)}}}")
    lines.append("terms:")
    for t in ordered:
        r = rows[t]
        lines += [
            f"  - id: {t}",
            f"    stage: {r['stage']}",
            f"    score: {score[t]}",
            f"    prereq: [{', '.join(r['prereq'])}]",
            f"    minutes: {r['minutes']}",
            f"    reason: {q(r['reason'])}",
        ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    terms, rows, score, ordered = build()
    write(rows, score, ordered)
    from collections import Counter
    cnt = Counter(rows[t]["stage"] for t in ordered)
    mins = Counter()
    for t in ordered:
        mins[rows[t]["stage"]] += rows[t]["minutes"]
    print(f"{OUT.relative_to(ROOT)} に {len(ordered)} 語を書いた")
    for s in sorted(cnt):
        print(f"  段階{s}: {cnt[s]} 語, 約 {mins[s] / 60:.1f} 時間")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_roadmap
    sys.exit(check_roadmap.main())


if __name__ == "__main__":
    main()
