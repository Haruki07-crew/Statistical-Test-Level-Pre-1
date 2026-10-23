# 問題の校閲記録（第1〜8章）

点検日：2026-10-23　対象：content/questions/ch01〜ch08.yaml（243問）

## 直したもの

- 全章の numeric 問題（ch01〜08 で計87か所）：「小数第N位まで」に丸め方の指定がなく、切り捨てで答えると tolerance の境目になる（例 q05-poisson-distribution-2 の 0.1465 → 0.147／0.146、q05-binomial-distribution-3 の 0.0287）→ すべて「小数第N位まで（四捨五入）」に統一した。
- q06-normal-mixture-distribution-1：小数第2位の指定なのに tolerance 0.01（4.35・4.37 も正解扱い）→ 0.005 にした。
- q06-noncentral-f-distribution-1：同上（2.8、tolerance 0.01）→ 0.005 にした。
- q07-extreme-value-distribution-1：n=1000 では近似値 log1000+γ=7.48497 がちょうど丸めの境目で、γ の桁数によって 7.48 と 7.49 に分かれる（正確な値は 7.4855）。tolerance も 0.01 で第2位の指定と合わない → n=500 に変更（近似 6.7918、正確 6.7928 でどちらも 6.79）、tolerance 0.005。verify/questions_ch07.py の検算も合わせて直した。
- q03-coefficient-of-variation-2：X が定数（σ=0）だと CV はどちらも0で等しくなり、命題が正しくなる場合がある → 「（定数でない）」を加えた。
- q08-mean-squared-error-3：解説の「MSE は 1/(n+1)<1/n<1/(n-1) の順に小さくなります」が、c の大小なのか MSE の大小なのか読み取れない → 「MSE は c=1/(n+1) のときが最も小さく、1/n、1/(n-1) の順に大きくなる」に書き換えた。
- q08-complete-statistic-1：ヒントの「(3/4)^6 を計算する」が答えそのもの → 「与えられた式に n と T を代入する」に書き換えた。計算が1段階しかないので、難易度を応用から標準に下げた。
- q08-fisher-information-1：ヒントが公式 I(p)=1/{p(1-p)} そのもので、選択肢を選ぶだけになっていた → 対数尤度を2回微分する方針のヒントにした。
- q06-t-distribution-2：ヒントが分散の公式 ν/(ν-2) そのもの → T=Z/√(W/ν) から E[T²] を求める方針のヒントにした。
- q05-geometric-distribution-1：ヒントが公式 q/p そのもの → 級数で計算するか、試行回数 Y=X+1 から考える方針と、定義の確認だけを示すヒントにした。
- ch04・ch07・ch08 の二項分布の記号：`B(n,p)` と書いていたが、用語の本文（content/ch05/binomial-distribution.md）と ch05 の問題は `\mathrm{Bin}(n,p)` を使っている → `\mathrm{Bin}` にそろえた（q04-sum-of-random-variables-1、q07-continuity-correction-2/3、q07-law-of-small-numbers-1/2、q07-normal-approximation-binomial-poisson-1/3、q08-exponential-family-1、q08-order-statistics-2）。

## 直さなかったが気になる点

- 章をまたいでほぼ同じ問題がある：q03-quantile-function-1 と q06-exponential-distribution-1（レート0.5の指数分布の中央値。答えはどちらも 1.386。q04-logit-logistic-transformation-1 の答えも 1.386）。q02-hazard-function-3 と q06-exponential-distribution-3（指数分布の最小値）。q06-cauchy-distribution-1 と q07-weak-law-of-large-numbers-2（コーシー分布の標本平均）。q02-cumulative-distribution-function-3 と q06-uniform-distribution-2（一様分布の最大値の期待値）。q04-logit-logistic-transformation-3 と q07-delta-method-3（ロジットの標準誤差。n だけが違う）。それぞれ別の用語の問題なので残したが、数値や設定を変えると「今日の復習」で同じ答えばかり出ることが減る。
- q01-conditional-probability-3（和が8のとき6を含む確率）は「応用」だが、数え上げ1回で済むので「標準」でもよい。
- Po と \mathrm{Po}、Bin と \mathrm{Bin} の書体が章によって違う（意味は同じ）。
- ヒントで公式を示している問題（q03-partial-correlation-coefficient-1、q06-beta-distribution-1、q06-gamma-distribution-1 など）は、公式を使うこと自体が問題の主眼なので残した。
- q07-chebyshev-inequality-2・q07-convergence-in-mean-square-1 の「最小の n」は整数なので tolerance 0.5 で問題ない。
