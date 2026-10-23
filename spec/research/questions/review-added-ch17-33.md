# 追加問題の校閲（第17〜33章）

対象：第17〜26章は `git diff cca1e31 HEAD`、第27〜33章は `git diff c349710 HEAD` で追加された問題 61 問（導出・4択）。2026-10-23 校閲。

## 直したもの

- q18-logistic-regression-5：切片 −2.0 では確率の比が 1.79 で、誤答「発症確率は約2倍」が「約2倍」と読めてしまう。切片を −1.0 にし（確率比 1.58、喫煙者のオッズ 0.736、オッズの差 0.368）、解説と検算を更新。
- q20-main-effect-and-interaction-3：選択肢は毎回シャッフルされるのに、解説が「最後の組」「1つ目」「2つ目」「3つ目」と位置で指していた。セル平均の値で指すように修正。
- q21-stratified-sampling-3：問題文の誤記「大きく小さくなる」を「大きく減る」に修正。解説の「1つ目は」を選択肢の内容で指すように修正。
- q21-horvitz-thompson-estimator-4：既存の q21-horvitz-thompson-estimator-1 と包含確率（0.2, 0.5, 0.1）も問い（総計の HT 推定）も同じで、ほぼ重複。値と包含確率を (6,0.3),(15,0.6),(4,0.1) に変え、母平均の推定（$\hat Y_{HT}/N=2.83$）を問う形にした。誤答は標本平均 8.33、$n$ で割った 28.33、総計のまま 85.00。検算を更新。
- q22-pca-on-correlation-matrix-3：解説の例 $\Sigma=\begin{pmatrix}16&6\\6&4\end{pmatrix}$ が q22-pca-on-correlation-matrix-2 と同じ行列で、その答え 0.875 を明かしていた。例を $\begin{pmatrix}9&3\\3&4\end{pmatrix}$（寄与率 0.80 と 0.75）に変更し、検算を更新。
- q25-precision-matrix-5：精度行列の成分（2, 2, −1）が q25-precision-matrix-1 と同じで、誤答の解説がその答え 0.5 を明かしていた。また解説が「最初の選択肢」と位置で指していた。$\Omega$ の中央を 3 に変え（$\Sigma=\frac18(5,2,1;2,4,2;1,2,5)$、周辺相関 0.2、偏相関 +0.41、周辺分散 5/8）、選択肢・解説を内容で指す形に直し、検算を更新（偏相関は Σ からの条件付き共分散でも照合）。
- q30-bias-variance-tradeoff-4：正解の「訓練誤差は減り続け」は、次数が $n-1$ で補間すると 0 で止まるので厳密でない。「訓練誤差は増えないが」に修正。誤答「期待テスト誤差＝バイアス²＋バリアンス」が確実に誤りになるよう、問題文に $\sigma^2>0$ を明記。解説の「2番目の選択肢」を内容で指すように修正。

## 確認したが直さなかった点

- q17-leverage-5：切片を数え忘れた目安 0.20 では A と D が該当し、解説の「点Aが選ばれる」は A が D と並ぶ候補になるという意味。誤答としては成り立つので、そのまま。
- q27-partial-autocorrelation-4：表の偏自己相関（0.667, −0.400, 0.127, 0.067, −0.127）を MA(2)（θ=(1, 0.5)、+符号の流儀は解説に明記）から再計算して一致を確認。
- q28-mcnemar-test-4：誤答 52.83（独立性のカイ二乗）を scipy で確認。
- q29-kaplan-meier-estimator-4：誤答 0.500（打ち切りをリスク集合から除かない）、0.286、0.200 の計算を確認。
- q24-k-means-method-4：誤答「初期中心によらず同じ分割に収束」は q24-k-means-method-3（正誤）と同じ論点だが、4択の誤答として使うだけなので残した。
- q23-sensitivity-and-specificity-4、q26-correspondence-analysis-4、q27-yule-walker-equations-4、q28-relative-risk-4、q30-kullback-leibler-divergence-4、q33-backdoor-criterion-4：それぞれ既存の数値問題と計算の型が近いが、数値が違い、4択の誤答や導出（逆確率重み付け、デルタ法、一般式）が加わっているので重複とはみなさなかった。
- q20-latin-square-design-3：解説の誤差自由度の式 $(p-1)(p-2)$ から q20-latin-square-design-1（5×5）の答えが出せるが、本文にもある基本公式なのでそのまま。
- q32-rejection-sampling-5：ヒントの「1個得るまでの試行回数の期待値は $M$」は方針の範囲（$M$ は自分で求める必要がある）と判断。
- 導出問題（q17-leverage-4、q18-logistic-regression-4、q19-ridge/lasso、q20-one-way-anova-4、q21-SRS-4・neyman-4、q22-loading-4、q23-fisher-4・misclassification-4、q24-ward-4、q25-precision-4、q26-MDS-4、q27-ar/ma-4、q28-relative-risk-4、q29-em-4、q30-aic/KL-4、q31-normal-normal/gamma-poisson-4、q32-inverse/rejection-4、q33-backdoor-4）：steps を1行ずつ追い、飛躍・記号の不統一・answer との食い違いはなかった。ガンマ分布はレート表記、MA は +θ の流儀が問題文に明記されていることを確認。
